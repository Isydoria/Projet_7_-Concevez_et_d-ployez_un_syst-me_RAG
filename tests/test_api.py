"""Tests unitaires de l'API (api/main.py) : TestClient de FastAPI, faux assistant, fausse reconstruction.

Aucun serveur lancé, aucun appel à Mistral. Le test fonctionnel api_test.py est aussi exécuté ici,
contre la même application de test (TestClient se comporte comme un client httpx).
"""

import json
from datetime import date

import httpx
import pytest
from fastapi.testclient import TestClient

import api_test
from api.main import create_app
from rag.assistant import AssistantAnswer

QUESTION = "Un concert de jazz à Bordeaux ce week-end"
ANSWER = AssistantAnswer(
    question=QUESTION,
    answer="Pas de jazz ce week-end, mais un concert de musique de chambre dimanche [1].",
    sources=[{"ref": 1, "uid": "1929716", "title": "Concert du dimanche", "start": "2026-11-01T11:00:00+01:00",
              "end": "2026-11-01T12:00:00+01:00", "city": "Bordeaux", "location_name": "Auditorium",
              "url": "https://openagenda.com/concert", "score": 0.81}],
    criteria={"city": "Bordeaux", "department": None,
              "period": {"label": "ce week-end", "start": "2026-10-31", "end": "2026-11-01"}},
    metadata={"today": "2026-10-29", "model": "faux-modele", "index": "test", "region": "Nouvelle-Aquitaine",
              "llm_called": True, "retrieved_events": 1, "retrieval_ms": 300.0, "generation_ms": 1400.0,
              "total_ms": 1750.0, "input_tokens": 2000, "output_tokens": 100, "reasoning_tokens": None,
              "finish_reason": "stop"},
    retrieved=[{"ref": 1, "uid": "1929716", "title": "Concert du dimanche", "score": 0.81}],
)


def mistral_error(status: int, text: str) -> httpx.HTTPStatusError:
    """Erreur que remonte LangChain quand Mistral refuse une requête."""
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    return httpx.HTTPStatusError(text, request=request, response=httpx.Response(status, request=request, text=text))


class FakeAssistant:
    """Remplace EventAssistant : réponse fixée (ou erreur) ; garde la question et la date reçues."""

    def __init__(self, error: Exception | None = None):
        self.error, self.calls = error, []

    def ask(self, question: str, today: date | None = None) -> AssistantAnswer:
        self.calls.append((question, today))
        if self.error:
            raise self.error
        return ANSWER


@pytest.fixture
def index_dir(tmp_path):
    """Dossier d'index factice, avec son manifeste."""
    directory = tmp_path / "nouvelle-aquitaine_2026-10-29"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "source_file": "events_nouvelle-aquitaine_2026-10-29.json.gz", "reference_date": "2026-10-29",
        "built_at": "2026-11-06T08:32:02+00:00", "embedding_model": "mistral-embed-2312",
        "events": 18771, "chunks": 20644}), encoding="utf-8")
    return directory


def make_client(index_dir, assistant=None, rebuild=None, loader_error: Exception | None = None):
    """Application de test : chargement et reconstruction remplacés par des faux."""
    assistant = assistant or FakeAssistant()
    loaded = []

    def load_assistant(directory):
        loaded.append(directory)
        if loader_error:
            raise loader_error
        return assistant

    app = create_app(load_assistant=load_assistant, rebuild_index=rebuild or (lambda refresh: index_dir),
                     index_dir=index_dir, env_file=None)
    return TestClient(app), assistant, loaded


# --- /ask -----------------------------------------------------------------------------------------------

def test_ask_returns_the_answer_of_the_assistant(index_dir):
    client, assistant, _ = make_client(index_dir)
    with client:
        response = client.post("/ask", json={"question": QUESTION, "today": "2026-10-29"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ANSWER.answer
    assert body["sources"][0]["url"] == "https://openagenda.com/concert"
    assert body["retrieved"][0]["uid"] == "1929716"
    assert body["criteria"]["period"]["label"] == "ce week-end"
    assert assistant.calls == [(QUESTION, date(2026, 10, 29))]       # date du jour transmise


def test_ask_without_date_uses_the_real_date(index_dir):
    client, assistant, _ = make_client(index_dir)
    with client:
        client.post("/ask", json={"question": QUESTION})
    assert assistant.calls == [(QUESTION, None)]                       # None : l'assistant prend la vraie date


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "   "}, {"question": "x" * 501}, {}, {"question": QUESTION, "today": "demain"},
], ids=["vide", "espaces", "trop longue", "absente", "date invalide"])
def test_ask_rejects_a_malformed_request(index_dir, payload):
    client, assistant, _ = make_client(index_dir)
    with client:
        response = client.post("/ask", json=payload)
    assert response.status_code == 422
    assert assistant.calls == []                                       # refusée avant d'atteindre l'assistant


@pytest.mark.parametrize("status, text, retry_after", [
    (402, '{"type": "billing_no_api_budget_configured"}', None),
    (403, '{"type": "tier_not_allowed"}', None),
    (429, '{"type": "rate_limited"}', "60"),
], ids=["sans budget", "modèle refusé", "débit dépassé"])
def test_ask_turns_a_refusal_of_mistral_into_a_clear_503(index_dir, status, text, retry_after):
    client, _, _ = make_client(index_dir, assistant=FakeAssistant(error=mistral_error(status, text)))
    with client:
        response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 503
    assert "Mistral" in response.json()["detail"] and str(status) in response.json()["detail"]
    assert response.headers.get("retry-after") == retry_after


def test_ask_when_the_index_could_not_be_loaded(index_dir):
    client, _, _ = make_client(index_dir, loader_error=FileNotFoundError("index introuvable"))
    with client:
        response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 503
    assert "index" in response.json()["detail"]


# --- /health et /metadata -------------------------------------------------------------------------------

def test_health_when_the_assistant_is_ready(index_dir):
    client, _, _ = make_client(index_dir)
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "index": "nouvelle-aquitaine_2026-10-29", "detail": None}


def test_health_when_the_index_is_missing(index_dir):
    client, _, _ = make_client(index_dir, loader_error=FileNotFoundError("index introuvable"))
    with client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_metadata_describes_the_index_and_the_models(index_dir):
    client, _, _ = make_client(index_dir)
    with client:
        body = client.get("/metadata").json()
    assert body["index"]["name"] == "nouvelle-aquitaine_2026-10-29"
    assert (body["index"]["events"], body["index"]["chunks"]) == (18771, 20644)
    assert body["index"]["reference_date"] == "2026-10-29"
    assert body["chat_model"] and body["embedding_model"] == "mistral-embed-2312"
    assert body["rebuild"]["state"] == "idle"


# --- /rebuild -------------------------------------------------------------------------------------------

def test_rebuild_is_disabled_without_a_configured_token(index_dir, monkeypatch):
    monkeypatch.delenv("REBUILD_TOKEN", raising=False)
    client, _, _ = make_client(index_dir)
    with client:
        response = client.post("/rebuild", headers={"X-Rebuild-Token": "secret"})
    assert response.status_code == 403
    assert "REBUILD_TOKEN" in response.json()["detail"]


@pytest.mark.parametrize("headers", [{}, {"X-Rebuild-Token": "mauvais"}], ids=["sans jeton", "mauvais jeton"])
def test_rebuild_refuses_a_missing_or_wrong_token(index_dir, monkeypatch, headers):
    monkeypatch.setenv("REBUILD_TOKEN", "secret")
    calls = []
    client, _, _ = make_client(index_dir, rebuild=lambda refresh: calls.append(refresh) or index_dir)
    with client:
        response = client.post("/rebuild", headers=headers)
    assert response.status_code == 401 and calls == []


def test_rebuild_runs_in_the_background_then_switches_to_the_new_index(index_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("REBUILD_TOKEN", "secret")
    new_index, calls = tmp_path / "nouvel-index", []
    client, _, loaded = make_client(index_dir, rebuild=lambda refresh: calls.append(refresh) or new_index)
    with client:
        response = client.post("/rebuild", headers={"X-Rebuild-Token": "secret"}, json={"refresh_data": True})
        status = client.get("/metadata").json()["rebuild"]
    assert response.status_code == 202 and response.json()["state"] == "running"
    assert calls == [True]                                             # téléchargement demandé
    assert status["state"] == "succeeded" and status["index"] == "nouvel-index"
    assert loaded == [index_dir, new_index]                            # l'assistant sert désormais le nouvel index


def test_rebuild_uses_the_saved_data_by_default(index_dir, monkeypatch):
    monkeypatch.setenv("REBUILD_TOKEN", "secret")
    calls = []
    client, _, _ = make_client(index_dir, rebuild=lambda refresh: calls.append(refresh) or index_dir)
    with client:
        client.post("/rebuild", headers={"X-Rebuild-Token": "secret"})
    assert calls == [False]


def test_a_failed_rebuild_keeps_the_previous_index(index_dir, monkeypatch):
    monkeypatch.setenv("REBUILD_TOKEN", "secret")

    def failing(refresh):
        raise mistral_error(402, "billing_no_api_budget_configured")

    client, assistant, _ = make_client(index_dir, rebuild=failing)
    with client:
        client.post("/rebuild", headers={"X-Rebuild-Token": "secret"})
        status = client.get("/metadata").json()["rebuild"]
        answer = client.post("/ask", json={"question": QUESTION})
    assert status["state"] == "failed" and "402" in status["message"]
    assert answer.status_code == 200                                   # l'ancien index répond toujours


def test_a_second_rebuild_is_refused_while_one_is_running(index_dir, monkeypatch):
    monkeypatch.setenv("REBUILD_TOKEN", "secret")
    client, _, _ = make_client(index_dir)
    with client:
        client.app.state.service.rebuild.state = "running"            # reconstruction en cours (simulée)
        response = client.post("/rebuild", headers={"X-Rebuild-Token": "secret"})
    assert response.status_code == 409


# --- Documentation et test fonctionnel ------------------------------------------------------------------

def test_every_route_is_documented_in_swagger(index_dir):
    client, _, _ = make_client(index_dir)
    with client:
        schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/ask", "/health", "/metadata", "/rebuild"}
    for path, methods in schema["paths"].items():
        for operation in methods.values():
            assert operation.get("summary") and operation.get("description"), path


def test_functional_script_passes_against_the_application(index_dir, monkeypatch):
    monkeypatch.delenv("REBUILD_TOKEN", raising=False)
    client, _, _ = make_client(index_dir)
    with client:
        results = api_test.run_checks(client)
    assert all(ok for _, ok, _ in results), results
