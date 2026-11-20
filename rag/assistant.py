"""Assistant de recommandation d'événements : la chaîne RAG complète, de la question à la réponse.

Chaîne LangChain (LCEL), en quatre étapes :
1. repérer les critères de la question (ville, département, période) ;
2. chercher les événements dans l'index Faiss (recherche filtrée, un résultat par événement) ;
3. si aucun événement ne passe les filtres : message fixe, sans appel au LLM ;
   sinon : prompt, puis génération par le LLM Mistral ;
4. assembler la réponse : texte, sources citées, critères, métadonnées.
"""

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
from langchain_mistralai import ChatMistralAI

from rag import config
from rag.criteria import Criteria, extract_criteria
from rag.indexing import load_vectorstore
from rag.prompts import PROMPT, format_event, no_result_answer, prompt_inputs
from rag.retrieval import EventRetriever, RetrievedEvent

# Références citées par le LLM : « [2] », « [1, 3] »
REFERENCE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class AssistantAnswer:
    """Réponse complète à une question."""
    question: str
    answer: str
    sources: list[dict]    # événements cités dans la réponse, liens tirés des métadonnées
    criteria: dict         # critères repérés dans la question
    metadata: dict         # date du jour, modèle, index, durées, tokens
    # Événements trouvés par la recherche et transmis au LLM, cités ou non (évaluation de la recherche)
    retrieved: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)   # textes transmis au LLM (évaluation de la fidélité)

    def to_dict(self, include_contexts: bool = False) -> dict:
        """Dictionnaire prêt pour le JSON ; les textes transmis au LLM seulement sur demande."""
        data = asdict(self)
        if not include_contexts:
            del data["contexts"]
        return data


def make_chat_model(model: str = config.CHAT_MODEL,
                    reasoning_effort: str | None = config.CHAT_REASONING_EFFORT) -> ChatMistralAI:
    """LLM de génération ; la clé MISTRAL_API_KEY est lue dans l'environnement.

    `reasoning_effort` : niveau de réflexion d'un modèle de raisonnement ; None pour un modèle qui ne
    raisonne pas (le paramètre n'est alors pas envoyé). ChatMistralAI réessaie les erreurs réseau,
    pas les erreurs 429 (débit dépassé) : elles remontent.
    """
    # model_kwargs : paramètres transmis tels quels à l'API de Mistral
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    return ChatMistralAI(model=model, temperature=config.CHAT_TEMPERATURE, max_tokens=config.CHAT_MAX_TOKENS,
                         model_kwargs=extra)


def reasoning_tokens(response_metadata: dict) -> int | None:
    """Tokens de réflexion d'une réponse (comptés dans les tokens de sortie), s'ils sont fournis."""
    details = (response_metadata.get("token_usage") or {}).get("completion_tokens_details") or {}
    return details.get("reasoning_tokens")


def cited_refs(answer: str, count: int) -> list[int]:
    """Numéros cités dans la réponse, sans doublon et dans l'ordre ; ceux hors de 1 à `count` sont ignorés."""
    numbers = {int(number) for group in REFERENCE_RE.findall(answer) for number in group.split(",")}
    return sorted(number for number in numbers if 1 <= number <= count)


def make_source(ref: int, event: RetrievedEvent) -> dict:
    """Source d'un événement cité : informations lues dans les métadonnées, jamais dans le texte du LLM."""
    metadata = event.document.metadata
    return {
        "ref": ref,
        **{key: metadata[key] for key in ["uid", "title", "start", "end", "city", "location_name", "url"]},
        "score": round(event.score, 3),
    }


def make_retrieved(ref: int, event: RetrievedEvent) -> dict:
    """Événement trouvé par la recherche : son numéro [n] dans le prompt (= son rang), uid, titre et score."""
    metadata = event.document.metadata
    return {"ref": ref, "uid": metadata["uid"], "title": metadata["title"], "score": round(event.score, 3)}


def criteria_to_dict(criteria: Criteria) -> dict:
    """Critères repérés, avec des dates au format ISO."""
    period = criteria.period
    return {
        "city": criteria.city,
        "department": criteria.department,
        "period": {
            "label": period.label,
            "start": period.start.isoformat() if period.start else None,
            "end": period.end.isoformat() if period.end else None,
        },
    }


def elapsed_ms(start: float) -> float:
    """Millisecondes écoulées depuis `start` (time.perf_counter)."""
    return round((time.perf_counter() - start) * 1000, 1)


class EventAssistant:
    """Répond aux questions sur les événements : recherche filtrée dans Faiss, puis génération par Mistral.

    Les étapes s'échangent un dictionnaire (« état ») qui s'enrichit d'une étape à l'autre.
    """

    def __init__(self, store: FAISS, llm: Runnable | None = None, index_name: str = "",
                 region: str = config.DEFAULT_REGION, model: str = config.CHAT_MODEL):
        self.retriever = EventRetriever(store)
        self.index_name = index_name
        self.region = region
        self.model = model
        llm = llm if llm is not None else make_chat_model(model)
        self.generation = PROMPT | llm   # LCEL : prompt, puis LLM
        self.chain = (
            RunnableLambda(self.find_criteria)
            | RunnableLambda(self.search_events)
            | RunnableBranch(
                (lambda state: not state["events"], RunnableLambda(self.fixed_answer)),
                RunnableLambda(self.generate),
            )
            | RunnableLambda(self.make_answer)
        )

    @classmethod
    def from_index(cls, index_dir: Path, **kwargs) -> "EventAssistant":
        """Assistant sur un index construit par build_index (le nom du dossier identifie l'index)."""
        return cls(load_vectorstore(index_dir), index_name=index_dir.name, **kwargs)

    def ask(self, question: str, today: date | None = None) -> AssistantAnswer:
        """Réponse à une question ; sans date fixée, la vraie date du jour (relue à chaque question)."""
        return self.chain.invoke({"question": question, "today": today or date.today()})

    # --- Étapes de la chaîne ---------------------------------------------------------------------

    def find_criteria(self, state: dict) -> dict:
        """Étape 1 : critères repérés par règles (ville, département, période)."""
        started = time.perf_counter()
        criteria = extract_criteria(state["question"], state["today"], self.retriever.city_index)
        return {**state, "started": started, "criteria": criteria}

    def search_events(self, state: dict) -> dict:
        """Étape 2 : recherche filtrée dans l'index Faiss."""
        start = time.perf_counter()
        events = self.retriever.search(state["question"], state["criteria"])
        return {**state, "events": events, "retrieval_ms": elapsed_ms(start)}

    def fixed_answer(self, state: dict) -> dict:
        """Étape 3, aucun événement : pas d'appel au LLM (le message fixe est écrit à l'étape 4)."""
        return {**state, "message": None, "generation_ms": 0.0}

    def generate(self, state: dict) -> dict:
        """Étape 3 : prompt rempli avec les événements, puis réponse du LLM."""
        start = time.perf_counter()
        inputs = prompt_inputs(state["question"], state["criteria"], state["events"], state["today"], self.region)
        message = self.generation.invoke(inputs)
        return {**state, "message": message, "generation_ms": elapsed_ms(start)}

    def make_answer(self, state: dict) -> AssistantAnswer:
        """Étape 4 : texte, sources citées, critères et métadonnées."""
        events, message, criteria = state["events"], state["message"], state["criteria"]
        answer = message.text if message else no_result_answer(criteria)   # .text : sans le bloc de réflexion
        usage = (message.usage_metadata or {}) if message else {}
        response_metadata = message.response_metadata if message else {}
        return AssistantAnswer(
            question=state["question"],
            answer=answer,
            sources=[make_source(ref, events[ref - 1]) for ref in cited_refs(answer, len(events))],
            criteria=criteria_to_dict(criteria),
            metadata={
                "today": state["today"].isoformat(),
                "model": self.model,
                "index": self.index_name,
                "region": self.region,
                "llm_called": message is not None,
                "search_text": self.retriever.query_text(state["question"], criteria),   # texte vectorisé
                "retrieved_events": len(events),
                "retrieval_ms": state["retrieval_ms"],
                "generation_ms": state["generation_ms"],
                "total_ms": elapsed_ms(state["started"]),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "reasoning_tokens": reasoning_tokens(response_metadata),
                # « length » : plafond de tokens atteint, réponse coupée (voire vide si la réflexion a tout pris)
                "finish_reason": response_metadata.get("finish_reason"),
            },
            retrieved=[make_retrieved(number, event) for number, event in enumerate(events, start=1)],
            contexts=[format_event(number, event, criteria.period) for number, event in enumerate(events, start=1)],
        )
