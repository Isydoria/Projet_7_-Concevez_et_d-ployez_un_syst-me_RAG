"""Tests unitaires du script scripts/ask.py : affichage, options et erreurs, sans appel à Mistral."""

import json
from datetime import date

import httpx
import pytest

from rag.assistant import AssistantAnswer
from scripts import ask

QUESTION = "Un concert de jazz à Bordeaux ce week-end"

ANSWER = AssistantAnswer(
    question=QUESTION,
    answer="Pas de jazz ce week-end, mais un concert de musique de chambre dimanche [1].",
    sources=[{"ref": 1, "uid": "22900016", "title": "Concert du dimanche",
              "start": "2026-11-01T11:00:00+01:00", "end": "2026-11-01T12:00:00+01:00",
              "city": "Bordeaux", "location_name": "Auditorium", "url": "https://openagenda.com/concert",
              "score": 0.806}],
    criteria={"city": "Bordeaux", "department": None,
              "period": {"label": "ce week-end", "start": "2026-10-31", "end": "2026-11-01"}},
    metadata={"today": "2026-10-29", "model": "mistral-large-4-0", "index": "test", "region": "Nouvelle-Aquitaine",
              "llm_called": True, "retrieved_events": 5, "retrieval_ms": 310.4, "generation_ms": 2450.2,
              "total_ms": 2761.0, "input_tokens": 2431, "output_tokens": 210, "reasoning_tokens": 150,
              "finish_reason": "stop"},
    contexts=["[1]\nTitre : Concert du dimanche"],
)

NO_RESULT = AssistantAnswer(
    question="Une exposition à Limoges",
    answer="Je n'ai trouvé aucun événement correspondant à ces critères : …",
    sources=[],
    criteria={"city": "Limoges", "department": None,
              "period": {"label": "à venir", "start": "2026-10-29", "end": None}},
    metadata={"today": "2026-10-29", "model": "mistral-large-4-0", "index": "test", "region": "Nouvelle-Aquitaine",
              "llm_called": False, "retrieved_events": 0, "retrieval_ms": 1.6, "generation_ms": 0.0,
              "total_ms": 2.0, "input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
              "finish_reason": None},
)


class FakeAssistant:
    """Remplace EventAssistant : renvoie une réponse fixée (ou lève une erreur) et garde la date reçue."""

    def __init__(self, answer: AssistantAnswer | None = None, error: Exception | None = None):
        self.answer, self.error, self.today = answer, error, None

    def ask(self, question: str, today: date | None = None) -> AssistantAnswer:
        self.today = today
        if self.error:
            raise self.error
        return self.answer


@pytest.fixture
def ready(monkeypatch, tmp_path):
    """Environnement prêt : une clé (factice) et un dossier d'index existant ; le .env réel n'est pas lu."""
    monkeypatch.setattr(ask, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-factice")
    return tmp_path


def use_assistant(monkeypatch, assistant: FakeAssistant) -> None:
    monkeypatch.setattr(ask.EventAssistant, "from_index", classmethod(lambda cls, index_dir: assistant))


# --- Affichage ----------------------------------------------------------------------------------

def test_format_answer():
    text = ask.format_answer(ANSWER)
    assert text.splitlines()[:2] == [
        f"Question : {QUESTION}",
        "Critères : lieu : Bordeaux ; période : ce week-end, du samedi 31 octobre 2026 au dimanche 1 novembre 2026",
    ]
    assert "\nPas de jazz ce week-end, mais un concert de musique de chambre dimanche [1].\n" in text
    assert "Sources :\n[1] Concert du dimanche — Auditorium, Bordeaux\n    https://openagenda.com/concert" in text
    assert text.endswith("mistral-large-4-0 ; 5 événements trouvés ; recherche 310 ms ; génération 2450 ms ; "
                         "2431 tokens en entrée, 210 en sortie (dont 150 de réflexion)")
    assert "[ATTENTION]" not in text


def test_format_answer_warns_when_the_answer_is_cut():
    cut = AssistantAnswer(question=QUESTION, answer="", sources=[], criteria=ANSWER.criteria,
                          metadata={**ANSWER.metadata, "finish_reason": "length"})
    assert "[ATTENTION] Réponse coupée" in ask.format_answer(cut)


def test_format_answer_without_llm():
    text = ask.format_answer(NO_RESULT)
    assert "Sources :" not in text
    assert text.endswith("Aucun événement trouvé ; recherche 2 ms ; sans appel au LLM")


# --- Options et erreurs -------------------------------------------------------------------------

def test_parse_args():
    args = ask.parse_args([QUESTION, "--today", "2026-10-29"])
    assert (args.question, args.today, args.index, args.json) == (QUESTION, date(2026, 10, 29), ask.DEFAULT_INDEX, False)


def test_missing_key(monkeypatch, capsys):
    monkeypatch.setattr(ask, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert ask.main([QUESTION]) == 1
    assert "[ERREUR] MISTRAL_API_KEY absente" in capsys.readouterr().err


def test_missing_index(ready, capsys):
    assert ask.main([QUESTION, "--index", str(ready / "absent")]) == 1
    assert "[ERREUR] Index introuvable" in capsys.readouterr().err


def test_rate_limit_error(ready, monkeypatch, capsys):
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    error = httpx.HTTPStatusError("429", request=request, response=httpx.Response(429, request=request))
    use_assistant(monkeypatch, FakeAssistant(error=error))
    assert ask.main([QUESTION, "--index", str(ready)]) == 1
    message = capsys.readouterr().err
    assert "[ERREUR] Limite de débit Mistral atteinte" in message
    assert "admin.mistral.ai" in message   # un quota à 0 ne se règle pas en attendant


# --- Sorties ------------------------------------------------------------------------------------

def test_text_output_with_a_fixed_date(ready, monkeypatch, capsys):
    assistant = FakeAssistant(ANSWER)
    use_assistant(monkeypatch, assistant)
    assert ask.main([QUESTION, "--index", str(ready), "--today", "2026-10-29"]) == 0
    assert assistant.today == date(2026, 10, 29)
    assert capsys.readouterr().out == ask.format_answer(ANSWER) + "\n"


def test_json_output(ready, monkeypatch, capsys):
    use_assistant(monkeypatch, FakeAssistant(ANSWER))
    assert ask.main([QUESTION, "--index", str(ready), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["sources"][0]["url"] == "https://openagenda.com/concert"
    assert "contexts" not in data
