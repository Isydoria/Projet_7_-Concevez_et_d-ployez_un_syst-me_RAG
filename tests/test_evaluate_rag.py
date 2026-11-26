"""Tests unitaires du script d'évaluation (scripts/evaluate_rag.py) : faux assistant, faux juge, aucun appel.

Mini jeu de test de 3 questions : une à réponse (jugée par Ragas), un refus, un message fixe.
"""

import asyncio
import json
from datetime import date

import httpx
import openai
import pytest

from rag.assistant import AssistantAnswer
from scripts import evaluate_rag
from scripts.evaluate_rag import EvaluationStopped, render_report, run_answers, run_judgement
from rag.evaluation import summarize
from tests.fake_judge import FakeEmbeddings, FakeJudgeLLM

TEST_SET = {
    "metadata": {"today": "2026-10-29", "source": "data/raw/photo.json.gz"},
    "questions": [
        {"id": 1, "question": "Un concert de jazz ?", "category": "answer", "expected_behavior": "recommend",
         "reference": "Le concert A a lieu dimanche.", "relevant_uids": ["a"], "excluded": {}},
        {"id": 2, "question": "Quel temps fera-t-il ?", "category": "off_topic", "expected_behavior": "refuse",
         "reference": "Refus poli.", "relevant_uids": [], "excluded": {}},
        {"id": 3, "question": "Un concert à Aubusson ce week-end", "category": "no_result",
         "expected_behavior": "fixed_message", "reference": "Message fixe.", "relevant_uids": [], "excluded": {}},
    ],
}
RUN_INFO = {"model": "faux-modele", "index": "test"}


def make_answer(question: str, text: str, cited: list[str], retrieved: list[str], llm_called: bool = True):
    return AssistantAnswer(
        question=question, answer=text, criteria={},
        sources=[{"ref": retrieved.index(uid) + 1, "uid": uid} for uid in cited],
        metadata={"llm_called": llm_called, "model": "faux-modele"},
        retrieved=[{"ref": rank, "uid": uid, "title": uid, "score": 0.8} for rank, uid in enumerate(retrieved, 1)],
        contexts=[f"[{rank}] événement {uid}" for rank, uid in enumerate(retrieved, 1)],
    )


ANSWERS = {
    "Un concert de jazz ?": make_answer("Un concert de jazz ?", "Le concert A a lieu dimanche [1].", ["a"], ["a", "x"]),
    "Quel temps fera-t-il ?": make_answer("Quel temps fera-t-il ?",
                                          "Je ne réponds qu'aux questions sur les événements en Nouvelle-Aquitaine.",
                                          [], ["a"]),
    "Un concert à Aubusson ce week-end": make_answer("Un concert à Aubusson ce week-end",
                                                     "Je n'ai trouvé aucun événement.", [], [], llm_called=False),
}


def http_error(status: int, text: str) -> httpx.HTTPStatusError:
    """Erreur renvoyée par ChatMistralAI (httpx) quand Mistral refuse une requête."""
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    return httpx.HTTPStatusError(text, request=request, response=httpx.Response(status, request=request, text=text))


def api_error(cls, status: int, text: str) -> openai.APIStatusError:
    """Erreur renvoyée par le client OpenAI du juge quand Mistral refuse une requête."""
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    return cls(text, response=httpx.Response(status, request=request, text=text), body=None)


class FakeAssistant:
    """Remplace EventAssistant : réponses écrites à l'avance ; erreurs simulées sur une question."""

    def __init__(self, errors: dict | None = None):
        self.asked, self.today, self.errors = [], None, errors or {}

    def ask(self, question: str, today: date | None = None) -> AssistantAnswer:
        self.asked.append(question)
        self.today = today
        if self.errors.get(question):
            raise self.errors[question].pop(0)
        return ANSWERS[question]


class FakeSleep:
    """Attentes simulées (synchrones et asynchrones), enregistrées sans attendre."""

    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)

    async def wait(self, seconds):
        self.waits.append(seconds)


# --- Phase 1 : réponses --------------------------------------------------------------------------------

def test_answers_phase_saves_every_answer_with_what_the_evaluation_needs(tmp_path):
    path, assistant, sleep = tmp_path / "answers.json", FakeAssistant(), FakeSleep()
    run_answers(TEST_SET, assistant, path, RUN_INFO, sleep=sleep)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["metadata"]["model"] == "faux-modele" and saved["metadata"]["today"] == "2026-10-29"
    assert list(saved["answers"]) == ["1", "2", "3"]
    assert saved["answers"]["1"]["contexts"] and saved["answers"]["1"]["retrieved"][0]["uid"] == "a"
    assert assistant.today == date(2026, 10, 29)           # date du jour fixée par le jeu de test
    assert sleep.waits == [31, 31]                         # 2 requêtes par minute : pause entre deux questions


def test_answers_phase_resumes_without_asking_again(tmp_path):
    path = tmp_path / "answers.json"
    run_answers(TEST_SET, FakeAssistant(), path, RUN_INFO, sleep=FakeSleep())
    assistant = FakeAssistant()
    run_answers(TEST_SET, assistant, path, RUN_INFO, sleep=FakeSleep())
    assert assistant.asked == []


def test_answers_phase_waits_and_retries_when_the_rate_is_exceeded(tmp_path):
    assistant = FakeAssistant({"Un concert de jazz ?": [http_error(429, '{"type": "rate_limited"}')]})
    sleep = FakeSleep()
    run_answers(TEST_SET, assistant, tmp_path / "answers.json", RUN_INFO, sleep=sleep)
    assert sleep.waits[0] == 60 and assistant.asked.count("Un concert de jazz ?") == 2


def test_answers_phase_stops_at_the_first_refusal_and_keeps_what_was_obtained(tmp_path):
    path = tmp_path / "answers.json"
    assistant = FakeAssistant({"Quel temps fera-t-il ?": [http_error(403, "tier_not_allowed")]})
    with pytest.raises(EvaluationStopped, match="403"):
        run_answers(TEST_SET, assistant, path, RUN_INFO, sleep=FakeSleep())
    assert list(json.loads(path.read_text(encoding="utf-8"))["answers"]) == ["1"]


def test_answers_phase_refuses_to_mix_models(tmp_path):
    path = tmp_path / "answers.json"
    run_answers(TEST_SET, FakeAssistant(), path, RUN_INFO, sleep=FakeSleep())
    with pytest.raises(EvaluationStopped, match="autre-modele"):
        run_answers(TEST_SET, FakeAssistant(), path, {**RUN_INFO, "model": "autre-modele"}, sleep=FakeSleep())


# --- Phase 2 : jugement --------------------------------------------------------------------------------

def answers_file(tmp_path) -> dict:
    path = tmp_path / "answers.json"
    return run_answers(TEST_SET, FakeAssistant(), path, RUN_INFO, sleep=FakeSleep())


def judge(tmp_path, answers, llm, use_ragas=True, judge_model="faux-juge", sleep=None) -> dict:
    return asyncio.run(run_judgement(TEST_SET, answers, llm, FakeEmbeddings(), tmp_path / "evaluation.json",
                                     judge_model, use_ragas=use_ragas, sleep=(sleep or FakeSleep()).wait))


def test_judgement_phase_uses_ragas_only_where_it_makes_sense(tmp_path):
    llm = FakeJudgeLLM()
    evaluation = judge(tmp_path, answers_file(tmp_path), llm)
    results = evaluation["questions"]
    assert [results[key]["ragas_status"] for key in "123"] == ["judged", "not_applicable", "not_applicable"]
    assert len(llm.calls) == 10                            # une seule réponse jugée, 10 appels
    assert results["1"]["ragas"]["faithfulness"] == 0.5
    assert all(results[key]["behavior_ok"] for key in "123")
    assert all(results[key]["similarity"] == pytest.approx(1.0) for key in "123")
    assert results["1"]["retrieval"]["first_rank"] == 1 and results["2"]["retrieval"] is None


def test_judgement_phase_resumes_without_judging_again(tmp_path):
    answers = answers_file(tmp_path)
    judge(tmp_path, answers, FakeJudgeLLM())
    llm = FakeJudgeLLM()
    judge(tmp_path, answers, llm)
    assert llm.calls == []


def test_judgement_without_ragas_then_with_ragas(tmp_path):
    answers = answers_file(tmp_path)
    llm = FakeJudgeLLM()
    evaluation = judge(tmp_path, answers, llm, use_ragas=False)
    assert evaluation["questions"]["1"]["ragas_status"] == "skipped" and llm.calls == []
    evaluation = judge(tmp_path, answers, llm)             # relance avec Ragas : seule la question sautée est jugée
    assert evaluation["questions"]["1"]["ragas_status"] == "judged" and len(llm.calls) == 10


def test_judgement_waits_and_retries_when_the_rate_is_exceeded(tmp_path):
    sleep = FakeSleep()
    llm = FakeJudgeLLM(error=api_error(openai.RateLimitError, 429, "rate_limited"), failures=1)
    evaluation = judge(tmp_path, answers_file(tmp_path), llm, sleep=sleep)
    assert 60 in sleep.waits and evaluation["questions"]["1"]["ragas_status"] == "judged"


@pytest.mark.parametrize("error", [
    api_error(openai.PermissionDeniedError, 403, "tier_not_allowed"),
    api_error(openai.RateLimitError, 429, "Not enough capacity"),      # serveurs saturés : attendre ne suffit pas
], ids=["refus d'accès", "serveurs saturés"])
def test_judgement_stops_at_a_refusal_that_waiting_will_not_fix(tmp_path, error):
    sleep = FakeSleep()
    with pytest.raises(EvaluationStopped, match="question 1"):
        judge(tmp_path, answers_file(tmp_path), FakeJudgeLLM(error=error, failures=1), sleep=sleep)
    assert 60 not in sleep.waits


def test_judgement_refuses_to_mix_judges(tmp_path):
    answers = answers_file(tmp_path)
    judge(tmp_path, answers, FakeJudgeLLM())
    with pytest.raises(EvaluationStopped, match="autre-juge"):
        judge(tmp_path, answers, FakeJudgeLLM(), judge_model="autre-juge")


def test_judgement_needs_every_answer(tmp_path):
    answers = answers_file(tmp_path)
    del answers["answers"]["2"]
    with pytest.raises(EvaluationStopped, match="réponse 2 manquante"):
        judge(tmp_path, answers, FakeJudgeLLM())


# --- Rapport -------------------------------------------------------------------------------------------

def test_report_shows_the_summary_the_questions_and_the_unsupported_statements(tmp_path):
    answers = answers_file(tmp_path)
    evaluation = judge(tmp_path, answers, FakeJudgeLLM())
    report = render_report(TEST_SET, answers, evaluation, summarize(list(evaluation["questions"].values())))
    assert report.startswith("# Évaluation de l'assistant")
    assert "| Comportement attendu respecté | 3/3 |" in report
    assert "| Ragas : fidélité | 0,50 | 1 |" in report
    assert "| 1 | Un concert de jazz ? |" in report
    assert "« B » : absente des événements" in report        # affirmation jugée non appuyée, avec son motif
    assert "- Juge Ragas : `faux-juge`" in report


def test_report_says_when_ragas_was_not_used(tmp_path):
    answers = answers_file(tmp_path)
    evaluation = judge(tmp_path, answers, FakeJudgeLLM(), use_ragas=False)
    report = render_report(TEST_SET, answers, evaluation, summarize(list(evaluation["questions"].values())))
    assert "- Juge Ragas : non utilisé (--no-ragas" in report
    assert "faux-juge" not in report                           # aucun juge nommé : il n'a rien jugé


def human_judgments(answers: dict, started_at: str | None = None) -> dict:
    """Classement humain des 3 réponses du mini jeu, rattaché aux réponses classées par leur date."""
    return {"metadata": {"answers_started_at": started_at or answers["metadata"]["started_at"]},
            "judgments": {"1": {"verdict": "C", "reason": "juste"}, "2": {"verdict": "C", "reason": "refus"},
                          "3": {"verdict": "C", "reason": "message fixe"}}}


def test_report_compares_the_measures_with_the_human_classification(tmp_path):
    answers = answers_file(tmp_path)
    evaluation = judge(tmp_path, answers, FakeJudgeLLM())
    summary = summarize(list(evaluation["questions"].values()))
    report = render_report(TEST_SET, answers, evaluation, summary, human=human_judgments(answers))
    assert "## Mesures automatiques comparées au classement humain" in report
    assert "| Questions | 3 | 0 | 0 |" in report


def test_report_does_not_compare_with_a_classification_of_other_answers(tmp_path):
    answers = answers_file(tmp_path)
    evaluation = judge(tmp_path, answers, FakeJudgeLLM())
    summary = summarize(list(evaluation["questions"].values()))
    report = render_report(TEST_SET, answers, evaluation, summary,
                           human=human_judgments(answers, started_at="2000-01-01T00:00:00"))
    assert "Écarts de comportement signalés par les règles" not in report   # pas de tableau de comparaison
    assert "porte sur d'autres réponses" in report


# --- Lancement -----------------------------------------------------------------------------------------

def test_main_without_key(monkeypatch, capsys):
    monkeypatch.setattr(evaluate_rag, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert evaluate_rag.main(["--phase", "judge"]) == 1
    assert "MISTRAL_API_KEY" in capsys.readouterr().err


def test_main_judge_phase_without_answers(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(evaluate_rag, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-factice")
    assert evaluate_rag.main(["--phase", "judge", "--output-dir", str(tmp_path)]) == 1
    assert "lancer d'abord --phase answers" in capsys.readouterr().err
