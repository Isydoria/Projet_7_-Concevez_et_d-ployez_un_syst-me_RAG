"""Tests unitaires de la récupération des événements Open Agenda (rag/fetch.py).

Aucun appel réseau : l'API est simulée (« mockée ») pour que les tests soient rapides,
reproductibles et indépendants de la disponibilité d'Open Agenda.
"""

import gzip
from datetime import date

import httpx
import pytest

from rag import fetch

FAKE_URL = "https://exemple.test/exports/json"
SAMPLE_EVENTS = [
    {"uid": "1", "title_fr": "Concert de jazz", "keywords_fr": ["Jazz", "Musique"]},
    {"uid": "2", "title_fr": "Théâtre à Périgueux", "keywords_fr": None},
]


def make_response(status_code: int, json_data=None) -> httpx.Response:
    """Fabrique une fausse réponse HTTP, comme si elle venait de l'API."""
    return httpx.Response(status_code, json=json_data, request=httpx.Request("GET", FAKE_URL))


class FakeGet:
    """Remplace httpx.get : renvoie (ou lève) les résultats prévus, dans l'ordre, et note chaque appel."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def sleeps(monkeypatch):
    """Remplace time.sleep pour ne pas attendre pendant les tests, et note les pauses demandées."""
    recorded = []
    monkeypatch.setattr(fetch.time, "sleep", recorded.append)
    return recorded


# --- build_where_clause ---------------------------------------------------------------

def test_where_clause_contains_region_and_one_year_period():
    where = fetch.build_where_clause("Nouvelle-Aquitaine", date(2026, 10, 29))
    assert 'location_region = "Nouvelle-Aquitaine"' in where
    assert "lastdate_end >= date'2025-10-29'" in where
    assert "location_department" not in where


def test_where_clause_adds_department_when_given():
    where = fetch.build_where_clause("Nouvelle-Aquitaine", date(2026, 10, 29), department="Creuse")
    assert 'location_department = "Creuse"' in where


def test_where_clause_keeps_apostrophes_inside_double_quotes():
    where = fetch.build_where_clause("Provence-Alpes-Côte d'Azur", date(2026, 10, 29))
    assert 'location_region = "Provence-Alpes-Côte d\'Azur"' in where


# --- slugify ------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Nouvelle-Aquitaine", "nouvelle-aquitaine"),
    ("Pyrénées-Atlantiques", "pyrenees-atlantiques"),
    ("Provence-Alpes-Côte d'Azur", "provence-alpes-cote-d-azur"),
])
def test_slugify(text, expected):
    assert fetch.slugify(text) == expected


# --- save_raw_events / load_raw_events -----------------------------------------------------

def test_save_then_load_returns_same_data(tmp_path):
    path = tmp_path / "sous_dossier" / "events.json.gz"
    metadata = {"count": 2, "reference_date": "2026-10-29"}

    fetch.save_raw_events(SAMPLE_EVENTS, metadata, path)
    loaded_metadata, loaded_events = fetch.load_raw_events(path)

    assert loaded_metadata == metadata
    assert loaded_events == SAMPLE_EVENTS  # accents et listes conservés


def test_saved_file_is_gzip_with_readable_accents(tmp_path):
    path = tmp_path / "events.json.gz"
    fetch.save_raw_events(SAMPLE_EVENTS, {}, path)
    with gzip.open(path, "rt", encoding="utf-8") as file:
        content = file.read()
    assert "Théâtre à Périgueux" in content  # ensure_ascii=False : pas de é


# --- fetch_events --------------------------------------------------------------------------

def test_fetch_events_returns_json_and_sends_filter(monkeypatch, sleeps):
    fake_get = FakeGet([make_response(200, SAMPLE_EVENTS)])
    monkeypatch.setattr(fetch.httpx, "get", fake_get)

    events = fetch.fetch_events('location_region = "X"', ["uid", "title_fr"])

    assert events == SAMPLE_EVENTS
    assert fake_get.calls[0]["url"].endswith("/exports/json")
    assert fake_get.calls[0]["params"] == {"where": 'location_region = "X"', "select": "uid, title_fr"}
    assert sleeps == []


def test_fetch_events_retries_after_temporary_error(monkeypatch, sleeps):
    fake_get = FakeGet([make_response(503), make_response(200, SAMPLE_EVENTS)])
    monkeypatch.setattr(fetch.httpx, "get", fake_get)

    assert fetch.fetch_events("where", ["uid"]) == SAMPLE_EVENTS
    assert len(fake_get.calls) == 2
    assert sleeps == [1]


def test_fetch_events_retries_after_network_error(monkeypatch, sleeps):
    fake_get = FakeGet([httpx.ConnectError("réseau coupé"), make_response(200, SAMPLE_EVENTS)])
    monkeypatch.setattr(fetch.httpx, "get", fake_get)

    assert fetch.fetch_events("where", ["uid"]) == SAMPLE_EVENTS
    assert sleeps == [1]


def test_fetch_events_does_not_retry_client_error(monkeypatch, sleeps):
    fake_get = FakeGet([make_response(400, {"message": "requête invalide"})])
    monkeypatch.setattr(fetch.httpx, "get", fake_get)

    with pytest.raises(httpx.HTTPStatusError):
        fetch.fetch_events("where", ["uid"])
    assert len(fake_get.calls) == 1
    assert sleeps == []


def test_fetch_events_gives_up_after_max_attempts(monkeypatch, sleeps):
    fake_get = FakeGet([make_response(503)] * 4)
    monkeypatch.setattr(fetch.httpx, "get", fake_get)

    with pytest.raises(httpx.HTTPStatusError):
        fetch.fetch_events("where", ["uid"], max_attempts=4)
    assert len(fake_get.calls) == 4
    assert sleeps == [1, 2, 4]


# --- download_events -----------------------------------------------------------------------

def test_download_events_writes_dated_file_with_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch, "fetch_events", lambda where, fields: SAMPLE_EVENTS)

    path = fetch.download_events(date(2026, 10, 29), department="Creuse", output_dir=tmp_path)

    assert path.name == "events_creuse_2026-10-29.json.gz"
    metadata, events = fetch.load_raw_events(path)
    assert events == SAMPLE_EVENTS
    assert metadata["count"] == 2
    assert metadata["reference_date"] == "2026-10-29"
    assert 'location_department = "Creuse"' in metadata["where"]
