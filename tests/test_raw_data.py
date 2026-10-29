"""Tests des données brutes récupérées (photo de référence du 29/10/2026).

Contrairement à tests/test_fetch.py, qui teste le CODE avec une API simulée, ces tests
vérifient les DONNÉES réellement récupérées. L'énoncé (étape 2) propose de « s'assurer
d'avoir récupéré les données attendues ». Ils lisent les fichiers versionnés (aucun appel réseau).
"""

from datetime import date, datetime, timedelta

import pytest

from rag import config
from rag.fetch import load_raw_events

REFERENCE_SNAPSHOT = config.RAW_DATA_DIR / "events_nouvelle-aquitaine_2026-10-29.json.gz"
SAMPLE_SNAPSHOT = config.RAW_DATA_DIR / "events_creuse_2026-10-29.json.gz"

# Champs renseignés pour 100 % des événements lors de l'exploration
REQUIRED_FIELDS = ["uid", "canonicalurl", "firstdate_begin", "lastdate_end", "location_name"]
# Titre : la source fournit quelques événements sans texte en français (4 sur la photo du 29/10,
# publiés dans une autre langue). Le nettoyage les exclut ; ici, on surveille seulement leur proportion.
MAX_SHARE_WITHOUT_TITLE = 0.001  # 0,1 %
# Données personnelles volontairement exclues (minimisation, RGPD)
PERSONAL_DATA_FIELDS = ["contributor_email", "contributor_contactnumber",
                        "contributor_contactname", "contributor_contactposition"]


@pytest.fixture(scope="module")
def snapshot():
    """Charge la photo de référence une seule fois pour tous les tests de ce fichier."""
    return load_raw_events(REFERENCE_SNAPSHOT)


def test_snapshot_is_not_empty_and_matches_metadata(snapshot):
    metadata, events = snapshot
    assert len(events) > 0
    assert metadata["count"] == len(events)


def test_all_events_are_in_the_requested_region(snapshot):
    _, events = snapshot
    assert {event["location_region"] for event in events} == {config.DEFAULT_REGION}


def test_all_events_end_within_the_period(snapshot):
    metadata, events = snapshot
    period_start = date.fromisoformat(metadata["reference_date"]) - timedelta(days=config.HISTORY_DAYS)
    too_old = [event["uid"] for event in events
               if datetime.fromisoformat(event["lastdate_end"]).date() < period_start]
    assert too_old == []


def test_uids_are_unique(snapshot):
    _, events = snapshot
    uids = [event["uid"] for event in events]
    assert len(uids) == len(set(uids))


def test_exactly_the_configured_fields_are_present(snapshot):
    _, events = snapshot
    assert all(set(event) == set(config.OPENAGENDA_FIELDS) for event in events)


def test_no_personal_data_fields(snapshot):
    _, events = snapshot
    assert not any(field in event for event in events for field in PERSONAL_DATA_FIELDS)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_required_fields_are_filled(snapshot, field):
    _, events = snapshot
    empty = [event["uid"] for event in events if not str(event.get(field) or "").strip()]
    assert empty == []


def test_events_without_title_are_rare(snapshot):
    _, events = snapshot
    without_title = [event["uid"] for event in events if not str(event.get("title_fr") or "").strip()]
    assert len(without_title) / len(events) < MAX_SHARE_WITHOUT_TITLE


def test_sample_contains_only_creuse():
    metadata, events = load_raw_events(SAMPLE_SNAPSHOT)
    assert len(events) > 0
    assert metadata["count"] == len(events)
    assert {event["location_department"] for event in events} == {"Creuse"}
