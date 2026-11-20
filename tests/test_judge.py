"""Tests unitaires du juge LLM (rag/judge.py) : vraies métriques Ragas, faux modèle juge, aucun appel à Mistral.

Le faux juge renvoie des réponses écrites à l'avance, au format que chaque métrique de Ragas demande,
et compte les appels.
"""

import asyncio
import os

import httpx
import pytest
from ragas.llms.base import InstructorBaseRagasLLM

from rag import config
from rag.judge import (MISTRAL_OPENAI_URL, RequestPacer, judge_answer, make_client, make_embeddings, make_judge,
                       similarity)
from tests.fake_judge import FakeEmbeddings, FakeJudgeLLM

QUESTION = {"id": 1, "question": "Un concert de jazz ?", "reference": "Le concert A a lieu dimanche.",
            "category": "answer", "expected_behavior": "recommend", "relevant_uids": ["a"], "excluded": {}}
ANSWER = {"answer": "Le concert A a lieu dimanche [1].", "contexts": ["[1] Concert A, dimanche"],
          "sources": [{"uid": "a"}], "retrieved": [{"uid": "a"}], "metadata": {"llm_called": True}}


def test_judge_answer_runs_the_four_metrics_with_ten_calls():
    llm = FakeJudgeLLM()
    scores = asyncio.run(judge_answer(QUESTION, ANSWER, llm, FakeEmbeddings()))
    # Fidélité 2 appels, pertinence 3, couverture documentaire 1, exactitude factuelle 4 (dans les deux sens)
    assert len(llm.calls) == 10
    assert scores["faithfulness"] == 0.5           # 1 affirmation appuyée sur 2
    assert scores["context_recall"] == 0.5
    assert scores["factual_correctness"] == 0.5
    assert scores["answer_relevancy"] == pytest.approx(1.0)


def test_faithfulness_keeps_the_detail_of_the_judge():
    scores = asyncio.run(judge_answer(QUESTION, ANSWER, FakeJudgeLLM(), FakeEmbeddings()))
    assert scores["faithfulness_statements"] == [
        {"statement": "A", "verdict": 1, "reason": "appuyée"},
        {"statement": "B", "verdict": 0, "reason": "absente des événements"},
    ]


def test_similarity_between_answer_and_reference():
    assert asyncio.run(similarity("réponse", "référence", FakeEmbeddings())) == pytest.approx(1.0)


# --- Espacement des requêtes ------------------------------------------------------------------------

class FakeClock:
    """Horloge et attente simulées : attendre fait avancer l'horloge, sans attente réelle."""

    def __init__(self):
        self.now, self.waits = 1000.0, []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.waits.append(round(seconds, 2))
        self.now += seconds


def request_for(model: str) -> httpx.Request:
    return httpx.Request("POST", f"{MISTRAL_OPENAI_URL}/chat/completions", json={"model": model})


def test_pacer_spaces_the_requests_of_each_model():
    clock = FakeClock()
    pacer = RequestPacer(chat_interval_s=31, embed_interval_s=1.1, sleep=clock.sleep, clock=clock)

    async def scenario():
        await pacer(request_for("ministral-14b-2512"))      # première requête : aucune attente
        clock.now += 10
        await pacer(request_for("ministral-14b-2512"))      # 10 s plus tard : attendre les 21 s restantes
        await pacer(request_for("ministral-8b-2512"))       # autre modèle : sa propre limite
        await pacer(request_for("mistral-embed-2312"))      # embeddings : 1 requête par seconde
        await pacer(request_for("mistral-embed-2312"))

    asyncio.run(scenario())
    assert clock.waits == [21.0, 1.1]
    assert pacer.counts == {"ministral-14b-2512": 2, "ministral-8b-2512": 1, "mistral-embed-2312": 2}


def test_client_points_to_mistral_without_automatic_retry():
    client = make_client(RequestPacer(), api_key="cle-factice")
    assert str(client.base_url).rstrip("/") == MISTRAL_OPENAI_URL
    assert client.max_retries == 0          # un refus remonte : le script décide d'attendre ou de s'arrêter


def test_judge_and_embeddings_use_the_configured_models():
    client = make_client(RequestPacer(), api_key="cle-factice")
    judge = make_judge(client)
    assert isinstance(judge, InstructorBaseRagasLLM) and judge.model == config.JUDGE_MODEL
    # Plafond de Ragas par défaut : 1 024 tokens, trop peu pour juger une longue réponse (sortie coupée, échec)
    assert judge.model_args["max_tokens"] == config.JUDGE_MAX_TOKENS > 1024
    assert make_embeddings(client).model == config.EMBEDDING_MODEL


def test_ragas_usage_statistics_are_disabled():
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
