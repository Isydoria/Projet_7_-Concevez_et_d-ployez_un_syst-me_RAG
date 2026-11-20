"""Juge LLM de l'évaluation : métriques Ragas, avec un modèle Mistral comme juge.

Métriques (une par mot de l'énoncé, plus l'exactitude) :
- Faithfulness : fidélité au contexte (affirmations de la réponse appuyées par les événements transmis) ;
- AnswerRelevancy : pertinence (la réponse répond-elle à la question ?) ;
- ContextRecall : couverture documentaire (la référence se retrouve-t-elle dans les événements transmis ?) ;
- FactualCorrectness : exactitude (mêmes informations que la référence, dans les deux sens).

Branchement : les métriques récentes de Ragas n'ont pas de connecteur Mistral ; le juge passe par l'API de
Mistral compatible avec celle d'OpenAI (client openai pointé sur MISTRAL_OPENAI_URL). Avec la formule gratuite,
chaque modèle de chat est limité à 2 requêtes par minute : un « hook » httpx espace toutes les requêtes,
nouvelles tentatives comprises. Aucune nouvelle tentative automatique : un refus remonte au script.
"""

import os

# Avant tout import de Ragas : pas de statistiques d'usage envoyées aux auteurs de la bibliothèque
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import asyncio  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import httpx  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings import OpenAIEmbeddings  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.llms.base import InstructorBaseRagasLLM  # noqa: E402
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, FactualCorrectness, Faithfulness  # noqa: E402

from rag import config  # noqa: E402
from rag.evaluation import cosine  # noqa: E402

MISTRAL_OPENAI_URL = "https://api.mistral.ai/v1"


class RequestPacer:
    """Espace les requêtes envoyées à Mistral, modèle par modèle ; appelé par httpx avant chaque requête.

    `sleep` et `clock` sont remplaçables pour les tests (attente simulée, sans attente réelle).
    """

    def __init__(self, chat_interval_s: float = config.CHAT_MIN_INTERVAL_S,
                 embed_interval_s: float = config.EMBED_MIN_INTERVAL_S, sleep=asyncio.sleep, clock=time.monotonic):
        self.chat_interval_s, self.embed_interval_s = chat_interval_s, embed_interval_s
        self.sleep, self.clock = sleep, clock
        self.last_request: dict[str, float] = {}   # modèle -> instant de sa dernière requête
        self.counts: Counter = Counter()           # modèle -> nombre de requêtes envoyées

    async def __call__(self, request: httpx.Request) -> None:
        model = json.loads(request.content or b"{}").get("model", "inconnu")
        interval = self.embed_interval_s if "embed" in model else self.chat_interval_s
        wait = interval - (self.clock() - self.last_request.get(model, float("-inf")))
        if wait > 0:
            await self.sleep(wait)
        self.last_request[model] = self.clock()
        self.counts[model] += 1


def make_client(pacer: RequestPacer, api_key: str | None = None) -> AsyncOpenAI:
    """Client OpenAI pointé sur l'API de Mistral ; la clé est lue dans MISTRAL_API_KEY si elle n'est pas donnée."""
    return AsyncOpenAI(api_key=api_key or os.environ["MISTRAL_API_KEY"], base_url=MISTRAL_OPENAI_URL,
                       http_client=httpx.AsyncClient(event_hooks={"request": [pacer]}, timeout=120),
                       max_retries=0)


def make_judge(client: AsyncOpenAI, model: str = config.JUDGE_MODEL) -> InstructorBaseRagasLLM:
    """Modèle juge pour Ragas (sorties structurées en JSON, via Instructor)."""
    return llm_factory(model, provider="mistral", client=client)


def make_embeddings(client: AsyncOpenAI) -> OpenAIEmbeddings:
    """Embeddings Mistral pour la pertinence et la similarité (même modèle que l'index)."""
    return OpenAIEmbeddings(client=client, model=config.EMBEDDING_MODEL)


async def faithfulness_detail(metric: Faithfulness, question: str, answer: str, contexts: list[str]) -> dict:
    """Fidélité avec le détail du juge (affirmation, verdict, motif), en suivant les étapes de Faithfulness.ascore.

    ascore ne renvoie que le score : on enchaîne ses deux étapes (version de Ragas figée, testée par un faux juge).
    """
    statements = await metric._create_statements(question, answer)
    if not statements:
        return {"faithfulness": None, "faithfulness_statements": []}
    verdicts = await metric._create_verdicts(statements, "\n".join(contexts))
    return {
        "faithfulness": float(metric._compute_score(verdicts)),
        "faithfulness_statements": [{"statement": item.statement, "verdict": item.verdict, "reason": item.reason}
                                    for item in verdicts.statements],
    }


async def judge_answer(question: dict, answer: dict, llm: InstructorBaseRagasLLM,
                       embeddings: BaseRagasEmbedding) -> dict:
    """Les quatre métriques Ragas d'une réponse (10 appels au juge), les unes après les autres."""
    user_input, response, contexts = question["question"], answer["answer"], answer["contexts"]
    reference = question["reference"]
    scores = await faithfulness_detail(Faithfulness(llm=llm), user_input, response, contexts)
    relevancy = await AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(user_input=user_input, response=response)
    recall = await ContextRecall(llm=llm).ascore(user_input=user_input, retrieved_contexts=contexts,
                                                 reference=reference)
    correctness = await FactualCorrectness(llm=llm).ascore(response=response, reference=reference)
    return {**scores, "answer_relevancy": relevancy.value, "context_recall": recall.value,
            "factual_correctness": correctness.value}


async def similarity(text: str, reference: str, embeddings: BaseRagasEmbedding) -> float:
    """Similarité cosinus entre les embeddings de la réponse et de la référence (sans juge)."""
    vectors = await embeddings.aembed_texts([text, reference])
    return cosine(vectors[0], vectors[1])
