"""Tests unitaires du découpage en chunks (rag/chunking.py)."""

import pytest

from rag import chunking, config
from rag.preprocess import clean_event
from tests.factories import make_raw_event


def make_clean_event(**raw_overrides) -> dict:
    """Événement nettoyé, fabriqué à partir d'un événement brut modifiable."""
    return clean_event(make_raw_event(**raw_overrides))


# --- Dates et âges en français --------------------------------------------------------------------

def test_format_datetime_fr_writes_day_and_month_in_french():
    assert chunking.format_datetime_fr("2026-10-31T20:30:00+01:00") == "samedi 31 octobre 2026 à 20h30"


@pytest.mark.parametrize("start, end, expected", [
    ("2026-10-31T20:30:00+01:00", "2026-10-31T23:30:00+01:00",
     "le samedi 31 octobre 2026, de 20h30 à 23h30"),                       # même jour
    ("2026-11-01T10:00:00+01:00", "2026-11-03T18:00:00+01:00",
     "du dimanche 1 novembre 2026 à 10h00 au mardi 3 novembre 2026 à 18h00"),  # plusieurs jours
    ("2026-11-01T10:00:00+01:00", "2026-11-01T10:00:00+01:00",
     "le dimanche 1 novembre 2026 à 10h00"),                                # début = fin
    ("", "", ""),
])
def test_format_dates(start, end, expected):
    assert chunking.format_dates(start, end) == expected


@pytest.mark.parametrize("age_min, age_max, expected", [
    (3.0, 16.0, "de 3 à 16 ans"),
    (6.0, None, "à partir de 6 ans"),
    (None, 12.0, "jusqu'à 12 ans"),
    (None, None, ""),
])
def test_format_age(age_min, age_max, expected):
    assert chunking.format_age(age_min, age_max) == expected


# --- En-tête et corps ------------------------------------------------------------------------------

def test_build_header_lists_available_information():
    event = make_clean_event(conditions_fr="Gratuit", age_min=6, keywords_fr=["Jazz", "Concert"])
    assert chunking.build_header(event) == (
        "Titre : Concert de jazz\n"
        "Dates : le dimanche 1 novembre 2026, de 11h00 à 13h00\n"
        "Lieu : Auditorium, Bordeaux (Gironde)\n"
        "Mots-clés : Jazz, Concert\n"
        "Tarifs : Gratuit\n"
        "Âge : à partir de 6 ans"
    )


def test_build_header_shows_status_and_attendance_only_when_unusual():
    usual = chunking.build_header(make_clean_event())
    assert "Statut" not in usual and "Participation" not in usual

    unusual = chunking.build_header(make_clean_event(
        status='{"id": 5, "label": {"fr": "Complet"}}',
        attendancemode='{"id": 2, "label": {"fr": "En ligne"}}',
    ))
    assert "Statut : Complet" in unusual
    assert "Participation : En ligne" in unusual


@pytest.mark.parametrize("description, long_html, expected", [
    ("Un quartet", "<p>Un quartet de jazz manouche</p>", "Un quartet de jazz manouche"),        # doublon évité
    ("Fais-moi peur", "<p>Un concert pour Halloween</p>", "Fais-moi peur\nUn concert pour Halloween"),
    ("Un quartet", None, "Un quartet"),                                                         # pas de description longue
])
def test_build_body_avoids_repeating_the_short_description(description, long_html, expected):
    event = make_clean_event(description_fr=description, longdescription_fr=long_html)
    assert chunking.build_body(event) == expected


# --- Découpage ---------------------------------------------------------------------------------------

def test_short_event_gives_one_chunk_with_header_and_metadata():
    documents = chunking.chunk_event(make_clean_event(uid="42"), chunking.make_splitter())

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "42#0"
    assert document.page_content.startswith("Titre : Concert de jazz\n")
    assert "\n---\nUn quartet" in document.page_content
    assert document.metadata["uid"] == "42"
    assert document.metadata["city"] == "Bordeaux"
    assert (document.metadata["chunk_index"], document.metadata["chunk_count"]) == (0, 1)
    assert set(chunking.METADATA_FIELDS) <= set(document.metadata)
    assert document.metadata["sessions"] == [
        {"begin": "2026-11-01T11:00:00+01:00", "end": "2026-11-01T13:00:00+01:00"}]


def numbered_description(sentences: int) -> str:
    """Description HTML faite de phrases numérotées (toutes différentes), pour tester le découpage."""
    return "<p>" + " ".join(f"Phrase numéro {i} du programme du festival." for i in range(sentences)) + "</p>"


def test_long_description_is_split_with_header_on_every_chunk():
    event = make_clean_event(uid="7", description_fr="Festival", longdescription_fr=numbered_description(120))
    documents = chunking.chunk_event(event, chunking.make_splitter())

    assert len(documents) > 1
    assert [document.id for document in documents] == [f"7#{i}" for i in range(len(documents))]
    for document in documents:
        assert document.page_content.startswith("Titre : Concert de jazz\n")
        body = document.page_content.split("\n---\n", 1)[1]
        assert len(body) <= config.CHUNK_SIZE
        assert document.metadata["chunk_count"] == len(documents)


def test_consecutive_chunks_overlap():
    # Sans description courte : une ligne courte suivie d'un paragraphe trop long forme un chunk à part
    # (comportement normal du découpeur récursif), ce qui n'est pas ce que l'on vérifie ici.
    event = make_clean_event(description_fr=None, longdescription_fr=numbered_description(120))
    first, second = chunking.chunk_event(event, chunking.make_splitter())[:2]
    first_body = first.page_content.split("\n---\n", 1)[1]
    second_body = second.page_content.split("\n---\n", 1)[1]
    assert second_body[:40] in first_body  # le début du 2e chunk reprend la fin du 1er


def test_event_without_description_gives_a_header_only_chunk():
    event = make_clean_event(description_fr=None, longdescription_fr=None)
    documents = chunking.chunk_event(event, chunking.make_splitter())
    assert len(documents) == 1
    assert documents[0].page_content == chunking.build_header(event)


def test_chunk_events_reports_counts():
    events = [make_clean_event(uid="court"),
              make_clean_event(uid="long", longdescription_fr=numbered_description(120))]
    documents, report = chunking.chunk_events(events)
    assert report["events"] == 2
    assert report["chunks"] == len(documents) >= 3
    assert report["split_events"] == 1
