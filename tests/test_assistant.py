"""Tests unitaires de l'assistant (rag/assistant.py) : chaîne complète avec un faux LLM, sans appel à Mistral.

Aujourd'hui : jeudi 29 octobre 2026 ; le week-end : samedi 31 octobre et dimanche 1er novembre.
"""

import json
from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag import config
from rag.assistant import AssistantAnswer, EventAssistant, cited_refs, make_chat_model
from rag.indexing import build_vectorstore
from tests.factories import KeywordEmbedder, make_chunk

TODAY = date(2026, 10, 29)
QUESTION = "Un concert de jazz à Bordeaux ce week-end"

CATALOGUE = [
    make_chunk("jazz-bordeaux", "Concert jazz jazz", [date(2026, 10, 31)]),
    make_chunk("expo-bordeaux", "Exposition exposition", [date(2026, 11, 1)]),
    make_chunk("theatre-poitiers", "Théâtre", [date(2026, 11, 1)], city="Poitiers", department="Vienne"),
    # Seul événement de Limoges, et il est passé : « à Limoges » ne laisse aucun événement à venir
    make_chunk("expo-limoges", "Exposition", [date(2026, 9, 1)], city="Limoges", department="Haute-Vienne"),
]


class FakeLLM:
    """Faux LLM : renvoie une réponse écrite à l'avance, avec des tokens, et garde les prompts reçus.

    Les métadonnées imitent celles de ChatMistralAI pour un modèle de raisonnement.
    """

    def __init__(self, reply: str, finish_reason: str = "stop"):
        self.reply, self.finish_reason = reply, finish_reason
        self.prompts = []   # un prompt (liste de messages) par appel

    def __call__(self, prompt_value) -> AIMessage:
        self.prompts.append(prompt_value.to_messages())
        return AIMessage(
            content=self.reply,
            usage_metadata={"input_tokens": 1200, "output_tokens": 150, "total_tokens": 1350},
            response_metadata={"finish_reason": self.finish_reason,
                               "token_usage": {"completion_tokens_details": {"reasoning_tokens": 100}}},
        )


def make_assistant(reply: str = "Je vous recommande le concert [1].",
                   finish_reason: str = "stop") -> tuple[EventAssistant, FakeLLM]:
    llm = FakeLLM(reply, finish_reason)
    store = build_vectorstore(CATALOGUE, KeywordEmbedder())
    return EventAssistant(store, llm=RunnableLambda(llm), index_name="test", model="faux-modele"), llm


# --- Références citées --------------------------------------------------------------------------

@pytest.mark.parametrize("answer, expected", [
    ("Le concert [1].", [1]),
    ("D'abord [2], puis [1].", [1, 2]),          # dans l'ordre des numéros
    ("Deux idées [1, 3].", [1, 3]),               # références groupées
    ("Encore [1] et [1].", [1]),                  # sans doublon
    ("Rien de valable : [0], [6], [12].", []),    # hors de 1 à 5 : ignorées
    ("Aucune référence.", []),
])
def test_cited_refs(answer, expected):
    assert cited_refs(answer, count=5) == expected


# --- Réponse complète ---------------------------------------------------------------------------

def test_sources_are_the_cited_events_with_links_from_the_metadata():
    assistant, _ = make_assistant("Pas de jazz annoncé, mais une exposition [2] ; et le concert [1].")
    answer = assistant.ask(QUESTION, today=TODAY)
    assert isinstance(answer, AssistantAnswer)
    assert [source["ref"] for source in answer.sources] == [1, 2]
    assert answer.sources[0]["uid"] == "jazz-bordeaux"   # classé premier pour « jazz »
    assert answer.sources[0]["url"] == "https://openagenda.com/jazz-bordeaux"
    assert set(answer.sources[0]) == {"ref", "uid", "title", "start", "end", "city", "location_name", "url", "score"}


def test_retrieved_lists_every_event_given_to_the_llm_even_if_not_cited():
    # Sans cette liste, une réponse « rien ne correspond » ne dit pas si la recherche a trouvé le bon événement
    assistant, _ = make_assistant("Aucun de ces événements ne correspond à votre demande.")
    answer = assistant.ask(QUESTION, today=TODAY)
    assert answer.sources == []
    assert [event["ref"] for event in answer.retrieved] == [1, 2]   # numéro [n] du prompt = rang de recherche
    assert [event["uid"] for event in answer.retrieved] == ["jazz-bordeaux", "expo-bordeaux"]
    assert set(answer.retrieved[0]) == {"ref", "uid", "title", "score"}
    assert "retrieved" in answer.to_dict()


def test_cited_sources_are_among_the_retrieved_events():
    assistant, _ = make_assistant("Le concert [1].")
    answer = assistant.ask(QUESTION, today=TODAY)
    retrieved = {(event["ref"], event["uid"]) for event in answer.retrieved}
    assert {(source["ref"], source["uid"]) for source in answer.sources} <= retrieved


def test_no_event_means_nothing_retrieved():
    assistant, _ = make_assistant()
    assert assistant.ask("Une exposition à Limoges", today=TODAY).retrieved == []


def test_an_answer_without_reference_has_no_source():
    assistant, _ = make_assistant("Aucun de ces événements ne correspond à votre demande.")
    assert assistant.ask(QUESTION, today=TODAY).sources == []


def test_no_event_gives_the_fixed_answer_without_calling_the_llm():
    assistant, llm = make_assistant()
    answer = assistant.ask("Une exposition à Limoges", today=TODAY)
    assert llm.prompts == []
    assert answer.answer.startswith("Je n'ai trouvé aucun événement correspondant à ces critères : lieu : Limoges")
    assert answer.sources == []
    assert answer.metadata["llm_called"] is False
    assert answer.metadata["retrieved_events"] == 0
    assert answer.metadata["input_tokens"] is None


def test_prompt_received_by_the_llm():
    assistant, llm = make_assistant()
    assistant.ask(QUESTION, today=TODAY)
    system, user = llm.prompts[0]
    assert "Nous sommes le jeudi 29 octobre 2026." in system.content
    assert "lieu : Bordeaux ; période : ce week-end" in user.content
    assert user.content.endswith(f"Question : {QUESTION}")


def test_criteria_and_metadata():
    assistant, _ = make_assistant()
    answer = assistant.ask(QUESTION, today=TODAY)
    assert answer.criteria == {"city": "Bordeaux", "department": None,
                               "period": {"label": "ce week-end", "start": "2026-10-31", "end": "2026-11-01"}}
    metadata = answer.metadata
    assert metadata["today"] == "2026-10-29"
    assert (metadata["model"], metadata["index"], metadata["region"]) == ("faux-modele", "test", "Nouvelle-Aquitaine")
    assert metadata["llm_called"] is True
    assert metadata["retrieved_events"] == 2   # à Bordeaux ce week-end : le concert et l'exposition
    assert (metadata["input_tokens"], metadata["output_tokens"]) == (1200, 150)
    assert (metadata["reasoning_tokens"], metadata["finish_reason"]) == (100, "stop")
    assert 0 <= metadata["retrieval_ms"] <= metadata["total_ms"]
    assert 0 <= metadata["generation_ms"] <= metadata["total_ms"]


def test_a_cut_answer_is_reported():
    # Modèle de raisonnement : si la réflexion consomme tout le plafond, la réponse est vide
    assistant, _ = make_assistant(reply="", finish_reason="length")
    answer = assistant.ask(QUESTION, today=TODAY)
    assert answer.answer == ""
    assert answer.metadata["finish_reason"] == "length"


def test_chat_model_settings(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-factice")   # aucun appel : on lit seulement la configuration
    llm = make_chat_model()
    assert (llm.model, llm.temperature, llm.max_tokens) == (config.CHAT_MODEL, config.CHAT_TEMPERATURE,
                                                            config.CHAT_MAX_TOKENS)
    expected = {"reasoning_effort": config.CHAT_REASONING_EFFORT} if config.CHAT_REASONING_EFFORT else {}
    assert llm.model_kwargs == expected
    # Modèle de raisonnement : le niveau de réflexion est envoyé ; modèle sans raisonnement : il ne l'est pas
    assert make_chat_model("mistral-large-4-0", reasoning_effort="low").model_kwargs == {"reasoning_effort": "low"}
    assert make_chat_model("ministral-8b-2512", reasoning_effort=None).model_kwargs == {}


def test_contexts_are_the_texts_given_to_the_llm():
    assistant, llm = make_assistant()
    answer = assistant.ask(QUESTION, today=TODAY)
    _, user = llm.prompts[0]
    assert len(answer.contexts) == answer.metadata["retrieved_events"]
    assert all(context in user.content for context in answer.contexts)


def test_to_dict_is_json_ready_and_leaves_contexts_out_by_default():
    assistant, _ = make_assistant()
    answer = assistant.ask(QUESTION, today=TODAY)
    data = answer.to_dict()
    assert "contexts" not in data
    json.dumps(data, ensure_ascii=False)   # lève une erreur si une valeur n'est pas convertible
    assert len(answer.to_dict(include_contexts=True)["contexts"]) == 2


def test_today_defaults_to_the_real_date():
    assistant, _ = make_assistant()
    assert assistant.ask("Une sortie").metadata["today"] == date.today().isoformat()


def test_the_chain_answers_several_questions_at_once():
    # batch : une liste de questions, une liste de réponses (utile pour l'évaluation)
    assistant, _ = make_assistant()
    answers = assistant.chain.batch([{"question": QUESTION, "today": TODAY},
                                     {"question": "Une exposition à Limoges", "today": TODAY}])
    assert [answer.metadata["llm_called"] for answer in answers] == [True, False]
