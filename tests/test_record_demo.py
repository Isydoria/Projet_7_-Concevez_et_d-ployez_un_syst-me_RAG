"""Tests de l'enregistrement des réponses de démonstration (scripts/record_demo.py), sans appel.

Le dernier test contrôle le fichier versionné data/demo/fallback_answers.json : il doit être complet et valide.
"""

import json
from datetime import date

import httpx
import pytest

from api.main import Service, load_fallback
from api.schemas import AskResponse
from rag import config
from rag.assistant import AssistantAnswer
from scripts.record_demo import DEMO_QUESTIONS, DEMO_TODAY, copy_from_evaluation, record

METADATA = {"today": "2026-10-29", "model": "faux-modele", "index": "test", "region": "Nouvelle-Aquitaine",
            "llm_called": True, "search_text": "Un opéra", "retrieved_events": 0, "retrieval_ms": 1.0,
            "generation_ms": 1.0, "total_ms": 2.0}


class FakeAssistant:
    """Réponse fixée pour chaque question ; erreur possible sur une question donnée."""

    def __init__(self, failing: str | None = None):
        self.failing, self.calls = failing, []

    def ask(self, question: str, today: date | None = None) -> AssistantAnswer:
        self.calls.append((question, today))
        if question == self.failing:
            request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
            raise httpx.HTTPStatusError("refus", request=request, response=httpx.Response(402, request=request))
        return AssistantAnswer(question=question, answer=f"Réponse à : {question}", sources=[],
                               criteria={"place": None, "period": None}, metadata=dict(METADATA))


def make_evaluation_file(path, questions):
    """Fichier de réponses au format de scripts/evaluate_rag.py (réponses indexées par numéro de question)."""
    answers = {str(number): {**FakeAssistant().ask(question).to_dict(include_contexts=True),
                             "contexts": ["texte transmis au LLM"]}
               for number, question in enumerate(questions, start=1)}
    run = {"metadata": {"model": "faux-modele", "today": "2026-10-29", "started_at": "2026-11-20T17:35:15"},
           "answers": answers}
    path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    return path


def test_every_demo_question_is_recorded_with_the_fixed_date(tmp_path):
    path, assistant, waits = tmp_path / "fallback.json", FakeAssistant(), []
    record(assistant, path, sleep=waits.append)
    saved = json.loads(path.read_text(encoding="utf-8"))["answers"]
    assert [entry["question"] for entry in saved] == DEMO_QUESTIONS
    assert all(entry["today"] == "2026-10-29" for entry in saved)
    assert assistant.calls == [(question, DEMO_TODAY) for question in DEMO_QUESTIONS]
    assert waits == [31, 31]                                     # 2 requêtes par minute


def test_answers_obtained_before_a_refusal_are_kept(tmp_path):
    path = tmp_path / "fallback.json"
    with pytest.raises(httpx.HTTPStatusError):
        record(FakeAssistant(failing=DEMO_QUESTIONS[1]), path, sleep=lambda seconds: None)
    assert [entry["question"] for entry in json.loads(path.read_text(encoding="utf-8"))["answers"]] == DEMO_QUESTIONS[:1]


def test_answers_are_copied_from_an_evaluation_without_the_llm_texts(tmp_path):
    eval_path = make_evaluation_file(tmp_path / "rag_answers.json", ["Autre question"] + DEMO_QUESTIONS)
    entries = copy_from_evaluation(eval_path)
    assert [entry["question"] for entry in entries] == DEMO_QUESTIONS             # ordre de la démonstration
    assert all(entry["recorded_at"] == "2026-11-20T17:35:15" for entry in entries)   # date de l'évaluation
    assert all(entry["today"] == "2026-10-29" for entry in entries)
    assert all("contexts" not in entry["answer"] for entry in entries)


def test_a_question_missing_from_the_evaluation_is_reported(tmp_path):
    eval_path = make_evaluation_file(tmp_path / "rag_answers.json", DEMO_QUESTIONS[:2])
    with pytest.raises(ValueError, match="Toulouse"):
        copy_from_evaluation(eval_path)


def test_the_api_reads_the_recorded_file(tmp_path):
    """Le fichier écrit par le script est bien celui que l'API relit au démarrage."""
    path = tmp_path / "fallback.json"
    record(FakeAssistant(), path, sleep=lambda seconds: None)
    service = Service(load_assistant=lambda index_dir: FakeAssistant(), rebuild_index=lambda refresh: tmp_path,
                      index_dir=tmp_path, fallback_file=path)
    service.start()
    recorded = service.recorded_answer(DEMO_QUESTIONS[0].upper(), DEMO_TODAY)   # casse indifférente
    assert recorded["answer"] == f"Réponse à : {DEMO_QUESTIONS[0]}"
    assert recorded["metadata"]["recorded_at"] and recorded["metadata"]["model"] == "faux-modele"
    assert service.recorded_answer(DEMO_QUESTIONS[0], date(2026, 11, 20)) is None   # autre date : pas de secours


def test_the_api_examples_are_recorded_demo_scenarios():
    """L'exemple du Swagger et la question de api_test.py répondent même si Mistral est indisponible."""
    from api.schemas import DEMO_QUESTION, AskRequest
    from api_test import DEMO
    assert DEMO_QUESTION in DEMO_QUESTIONS
    assert DEMO == {"question": DEMO_QUESTION, "today": DEMO_TODAY.isoformat()}
    today_examples = AskRequest.model_json_schema()["properties"]["today"]["examples"]
    assert today_examples == [DEMO_TODAY.isoformat()]


def test_the_fallback_file_is_copied_into_the_docker_image():
    """La démonstration se fait dans le conteneur : le fichier doit entrer dans l'image (liste blanche)."""
    folder = config.DEMO_FALLBACK_FILE.parent.relative_to(config.PROJECT_ROOT).as_posix()   # data/demo
    dockerignore = (config.PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    dockerfile = (config.PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    assert f"!{folder}/" in dockerignore
    assert f"COPY {folder}/ {folder}/" in dockerfile


def test_the_versioned_fallback_file_covers_the_demo_and_matches_the_api_schema():
    """Données produites, tolérance zéro : chaque scénario a une réponse complète, valide pour l'API."""
    fallback = load_fallback(config.DEMO_FALLBACK_FILE)
    service = Service(load_assistant=None, rebuild_index=None, index_dir=config.VECTORSTORE_DIR)
    service.fallback = fallback
    assert len(fallback) == len(DEMO_QUESTIONS)
    for question in DEMO_QUESTIONS:
        recorded = service.recorded_answer(question, DEMO_TODAY)
        assert recorded is not None, question
        response = AskResponse.model_validate(recorded)
        assert response.answer.strip() and response.metadata.recorded_at and response.metadata.model
