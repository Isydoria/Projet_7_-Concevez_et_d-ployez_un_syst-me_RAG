"""Tests unitaires du nettoyage (rag/preprocess.py), sur des événements fabriqués."""

from datetime import date

import pytest

from rag import preprocess
from rag.fetch import load_raw_events, save_raw_events
from tests.factories import make_raw_event

REFERENCE_DATE = date(2026, 10, 29)


# --- html_to_text (règle 8) ------------------------------------------------------------------

def test_html_to_text_keeps_text_and_line_breaks():
    html = "<p>Préparez vos neurones<br>La Ruelle vous attend</p><p>Entrée&nbsp;libre &amp; gratuite</p>"
    assert preprocess.html_to_text(html) == "Préparez vos neurones\nLa Ruelle vous attend\nEntrée libre & gratuite"


@pytest.mark.parametrize("value", [None, ""])
def test_html_to_text_handles_empty_values(value):
    assert preprocess.html_to_text(value) == ""


# --- clean_text (règle 11) ---------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    (".", ""),
    (" … ", ""),
    (None, ""),
    ("  Concours   de gâteaux ", "Concours de gâteaux"),
])
def test_clean_text(text, expected):
    assert preprocess.clean_text(text) == expected


# --- parse_json_field / extract_label (règle 9) ------------------------------------------------

def test_parse_json_field_decodes_text_and_keeps_lists():
    assert preprocess.parse_json_field('[{"type": "link"}]') == [{"type": "link"}]
    assert preprocess.parse_json_field(["déjà", "une liste"]) == ["déjà", "une liste"]
    assert preprocess.parse_json_field(None) is None
    assert preprocess.parse_json_field("{pas du json") is None


@pytest.mark.parametrize("value, expected", [
    ('{"id": 6, "label": {"fr": "Annulé", "en": "Canceled"}}', (6, "Annulé")),
    ('{"id": 1, "label": "Sur place"}', (1, "Sur place")),
    (None, (None, "")),
])
def test_extract_label(value, expected):
    assert preprocess.extract_label(value) == expected


# --- department_from_postalcode (règle 4) -------------------------------------------------------

@pytest.mark.parametrize("postalcode, declared, expected", [
    ("24800", "Charente", "Dordogne"),                          # Thiviers : nom saisi faux
    ("33290", "gironde", "Gironde"),                            # casse corrigée
    ("64800", "Pyrénées Orientales", "Pyrénées-Atlantiques"),   # Haut-de-Bosdarros
    (None, "Gironde", "Gironde"),                               # pas de code postal : nom saisi
    ("75001", "Paris", "Paris"),                                # hors Nouvelle-Aquitaine : nom saisi
])
def test_department_from_postalcode(postalcode, declared, expected):
    assert preprocess.department_from_postalcode(postalcode, declared) == expected


# --- to_paris_time (règle 10) -------------------------------------------------------------------

@pytest.mark.parametrize("utc, paris", [
    ("2026-11-01T10:00:00+00:00", "2026-11-01T11:00:00+01:00"),  # heure d'hiver
    ("2026-07-01T10:00:00+00:00", "2026-07-01T12:00:00+02:00"),  # heure d'été
])
def test_to_paris_time(utc, paris):
    assert preprocess.to_paris_time(utc) == paris


def test_clean_sessions_converts_each_session_to_paris_time():
    timings = '[{"begin": "2026-11-01T10:00:00+00:00", "end": "2026-11-01T12:00:00+00:00"}]'
    assert preprocess.clean_sessions(timings, "", "") == [
        {"begin": "2026-11-01T11:00:00+01:00", "end": "2026-11-01T13:00:00+01:00"}]


@pytest.mark.parametrize("timings", [None, "[]", "{pas du json"])
def test_clean_sessions_falls_back_on_the_event_range(timings):
    start, end = "2026-11-01T11:00:00+01:00", "2026-11-03T18:00:00+01:00"
    assert preprocess.clean_sessions(timings, start, end) == [{"begin": start, "end": end}]


# --- is_excluded_agenda (règle 5) ---------------------------------------------------------------

@pytest.mark.parametrize("agenda, expected", [
    ("Mes événements France Travail", True),
    ("Semaine de l'industrie 2027", True),        # édition future : couverte
    ("Semaine de l’industrie 2026", True),        # apostrophe typographique
    ("Les balades de l'Alternative Urbaine", False),
    ("Opéra National de Bordeaux", False),
    (None, False),
])
def test_is_excluded_agenda(agenda, expected):
    assert preprocess.is_excluded_agenda(agenda) == expected


# --- clean_event ----------------------------------------------------------------------------------

def test_clean_event_structures_the_event():
    cleaned = preprocess.clean_event(make_raw_event())
    assert cleaned["title"] == "Concert de jazz"
    assert cleaned["long_description"] == "Un quartet"
    assert cleaned["start"] == "2026-11-01T11:00:00+01:00"
    assert (cleaned["status_id"], cleaned["status_label"]) == (1, "Programmé")
    assert cleaned["attendance_label"] == "Sur place"
    assert cleaned["sessions"] == [{"begin": "2026-11-01T11:00:00+01:00", "end": "2026-11-01T13:00:00+01:00"}]
    assert (cleaned["latitude"], cleaned["longitude"]) == (44.84, -0.57)


# --- preprocess_events : toutes les règles sur un petit jeu fabriqué -------------------------------

def test_preprocess_events_applies_all_rules():
    events = [
        make_raw_event(uid="ok"),
        make_raw_event(uid="annule", title_fr="Autre", status='{"id": 6, "label": {"fr": "Annulé"}}'),
        make_raw_event(uid="emploi", title_fr="Job dating", originagenda_title="Mes événements France Travail"),
        make_raw_event(uid="2032", title_fr="Hôtel", firstdate_begin="2032-01-01T01:00:00+00:00"),
        make_raw_event(uid="sans-titre", title_fr=None),
        make_raw_event(uid="doublon", title_fr="  CONCERT de jazz ", updatedat="2025-01-01T00:00:00+00:00"),
    ]
    cleaned, report = preprocess.preprocess_events(events, REFERENCE_DATE)

    assert [event["uid"] for event in cleaned] == ["ok"]
    assert report == {"input": 6, "departments_fixed": 0, "cancelled": 1, "excluded_agendas": 1,
                      "beyond_horizon": 1, "without_title": 1, "duplicates": 1, "output": 1}


def test_preprocess_events_keeps_most_recent_duplicate():
    events = [
        make_raw_event(uid="ancien", updatedat="2025-01-01T00:00:00+00:00"),
        make_raw_event(uid="recent", updatedat="2026-10-20T00:00:00+00:00"),
    ]
    cleaned, _ = preprocess.preprocess_events(events, REFERENCE_DATE)
    assert [event["uid"] for event in cleaned] == ["recent"]


def test_preprocess_events_excludes_cancelled_before_deduplicating():
    events = [
        make_raw_event(uid="annule-recent", status='{"id": 6, "label": {"fr": "Annulé"}}',
                   updatedat="2026-10-20T00:00:00+00:00"),
        make_raw_event(uid="programme-ancien", updatedat="2025-01-01T00:00:00+00:00"),
    ]
    cleaned, _ = preprocess.preprocess_events(events, REFERENCE_DATE)
    assert [event["uid"] for event in cleaned] == ["programme-ancien"]


def test_preprocess_events_counts_fixed_departments():
    events = [make_raw_event(location_postalcode="24800", location_department="Charente")]
    cleaned, report = preprocess.preprocess_events(events, REFERENCE_DATE)
    assert cleaned[0]["department"] == "Dordogne"
    assert report["departments_fixed"] == 1


# --- preprocess_file --------------------------------------------------------------------------------

def test_preprocess_file_writes_processed_file_with_report(tmp_path):
    raw_path = tmp_path / "raw" / "events_test_2026-10-29.json.gz"
    save_raw_events([make_raw_event()], {"reference_date": "2026-10-29"}, raw_path)

    output_path = preprocess.preprocess_file(raw_path, output_dir=tmp_path / "processed")

    assert output_path.name == "events_test_2026-10-29.json.gz"
    metadata, events = load_raw_events(output_path)
    assert metadata["count"] == 1
    assert metadata["report"]["output"] == 1
    assert events[0]["title"] == "Concert de jazz"
