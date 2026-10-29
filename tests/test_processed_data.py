"""Tests des données nettoyées : zéro tolérance sur ce que l'on fournit à l'index.

Le jeu nettoyé n'est pas versionné (il est régénérable) : la fixture rejoue le nettoyage
sur la photo de référence versionnée (quelques secondes, aucun appel réseau).
"""

import re
from datetime import date, datetime, timedelta

import pytest

from rag import config
from rag.fetch import load_raw_events
from rag.preprocess import is_excluded_agenda, normalize_key, preprocess_events

HTML_TAG = re.compile(r"<[a-zA-Z/][^>]*>")


@pytest.fixture(scope="module")
def processed():
    """Nettoie la photo de référence une seule fois pour tous les tests du fichier."""
    metadata, raw_events = load_raw_events(config.REFERENCE_RAW_FILE)
    reference_date = date.fromisoformat(metadata["reference_date"])
    events, report = preprocess_events(raw_events, reference_date)
    return reference_date, events, report


def test_report_is_consistent(processed):
    _, events, report = processed
    removed = sum(report[rule] for rule in
                  ["cancelled", "excluded_agendas", "beyond_horizon", "without_title", "duplicates"])
    assert report["input"] - removed == report["output"] == len(events)


def test_every_event_has_a_title(processed):
    _, events, _ = processed
    assert all(event["title"] for event in events)


def test_no_cancelled_event(processed):
    _, events, _ = processed
    assert all(event["status_id"] != config.CANCELLED_STATUS_ID for event in events)


def test_no_excluded_agenda(processed):
    _, events, _ = processed
    assert not any(is_excluded_agenda(event["source_agenda"]) for event in events)


def test_no_html_left_in_texts(processed):
    _, events, _ = processed
    assert not any(HTML_TAG.search(event[field])
                   for event in events for field in ["title", "description", "long_description"])


def test_no_duplicates(processed):
    _, events, _ = processed
    keys = [(normalize_key(e["title"]), e["start"], normalize_key(e["location_name"])) for e in events]
    assert len(keys) == len(set(keys))


def test_uids_are_unique(processed):
    _, events, _ = processed
    uids = [event["uid"] for event in events]
    assert len(uids) == len(set(uids))


def test_starts_are_within_the_horizon(processed):
    reference_date, events, _ = processed
    horizon = reference_date + timedelta(days=config.MAX_HORIZON_DAYS)
    assert all(datetime.fromisoformat(event["start"]).date() <= horizon for event in events)


def test_dates_are_in_paris_time(processed):
    _, events, _ = processed
    paris_offsets = ("+01:00", "+02:00")
    assert all(event["start"][-6:] in paris_offsets and event["end"][-6:] in paris_offsets
               for event in events)


def test_departments_match_postal_codes(processed):
    _, events, _ = processed
    for event in events:
        prefix = event["postal_code"][:2]
        if prefix in config.NA_DEPARTMENTS:
            assert event["department"] == config.NA_DEPARTMENTS[prefix]
