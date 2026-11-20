"""Tests unitaires de la recherche filtrée (rag/retrieval.py), sur un petit index construit sans Mistral.

Aujourd'hui : jeudi 5 novembre 2026 ; le week-end : samedi 7 et dimanche 8 novembre.
"""

from datetime import date

import pytest

from rag import config
from rag.criteria import Criteria, Period
from rag.indexing import build_vectorstore
from rag.retrieval import EventRetriever, overlaps, sessions_in_period
from tests.factories import KeywordEmbedder, make_chunk, session

TODAY = date(2026, 11, 5)
UPCOMING = Period("à venir", TODAY, None)
WEEKEND = Period("ce week-end", date(2026, 11, 7), date(2026, 11, 8))


CATALOGUE = [
    make_chunk("jazz-bordeaux", "Concert jazz jazz jazz", [date(2026, 11, 7)]),
    make_chunk("jazz-passe", "Concert jazz jazz jazz", [date(2026, 10, 10)]),
    # Le mardi seulement : sa plage (3 au 10 novembre) couvre le week-end, pas ses séances
    make_chunk("atelier-mardi", "Atelier jazz jazz pour enfants", [date(2026, 11, 3), date(2026, 11, 10)]),
    make_chunk("theatre-chatellerault", "Théâtre théâtre théâtre", [date(2026, 11, 7)],
               city="Châtellerault", department="Vienne"),
    make_chunk("theatre-poitiers", "Théâtre et exposition", [date(2026, 11, 8)],
               city="POITIERS", department="Vienne"),
    # Événement en deux chunks : pour « jazz », le meilleur est le second
    make_chunk("festival", "Exposition exposition", [date(2026, 11, 7), date(2026, 11, 8)],
               chunk_index=0, chunk_count=2),
    make_chunk("festival", "Festival jazz jazz", [date(2026, 11, 7), date(2026, 11, 8)],
               chunk_index=1, chunk_count=2),
]


@pytest.fixture
def embedder():
    return KeywordEmbedder()


@pytest.fixture
def retriever(embedder):
    return EventRetriever(build_vectorstore(CATALOGUE, embedder))


def uids_of(results) -> list[str]:
    return [result.document.metadata["uid"] for result in results]


# --- Dates et séances ---------------------------------------------------------------------------

@pytest.mark.parametrize("begin, end, expected", [
    (date(2026, 11, 6), date(2026, 11, 9), True),    # séance de plusieurs jours qui englobe le week-end
    (date(2026, 11, 8), date(2026, 11, 8), True),
    (date(2026, 11, 9), date(2026, 11, 9), False),
    (date(2026, 11, 6), date(2026, 11, 6), False),
])
def test_overlaps(begin, end, expected):
    assert overlaps(WEEKEND, begin, end) is expected


def test_overlaps_with_an_open_period():
    assert overlaps(UPCOMING, date(2030, 1, 1), date(2030, 1, 1))
    assert not overlaps(UPCOMING, date(2026, 11, 4), date(2026, 11, 4))


def test_sessions_in_period_keeps_only_overlapping_sessions():
    sessions = [session(date(2026, 11, 3)), session(date(2026, 11, 7)), session(date(2026, 11, 10))]
    assert sessions_in_period(sessions, WEEKEND) == [session(date(2026, 11, 7))]


# --- Recherche ----------------------------------------------------------------------------------

def test_city_index_is_built_from_the_index_metadata(retriever):
    assert retriever.city_index["bordeaux"] == "Bordeaux"
    assert retriever.city_index["chatellerault"] == "Châtellerault"


def test_one_result_per_event_with_its_best_chunk(retriever):
    results = retriever.search("jazz", Criteria(period=UPCOMING))
    uids = uids_of(results)
    assert len(uids) == len(set(uids))
    festival = next(result for result in results if result.document.metadata["uid"] == "festival")
    assert festival.document.id == "festival#1"


def test_returns_at_most_k_events_sorted_by_score(retriever):
    results = retriever.search("jazz", Criteria(period=UPCOMING), k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_city_filter_compares_normalized_names(retriever):
    # Châtellerault est plus proche de « théâtre », mais n'est pas à Poitiers ; « POITIERS » = « Poitiers »
    results = retriever.search("théâtre", Criteria(period=UPCOMING, city="Poitiers"))
    assert uids_of(results) == ["theatre-poitiers"]


def test_department_filter(retriever):
    results = retriever.search("théâtre", Criteria(period=UPCOMING, department="Vienne"))
    assert set(uids_of(results)) == {"theatre-chatellerault", "theatre-poitiers"}


def test_period_filter_uses_the_real_sessions(retriever):
    uids = uids_of(retriever.search("jazz", Criteria(period=WEEKEND)))
    assert "jazz-bordeaux" in uids and "festival" in uids
    assert "atelier-mardi" not in uids  # sa plage couvre le week-end, mais aucune séance ces jours-là


def test_upcoming_period_excludes_past_events(retriever):
    assert "jazz-passe" not in uids_of(retriever.search("jazz", Criteria(period=UPCOMING)))
    last_month = Period("le mois dernier", date(2026, 10, 1), date(2026, 10, 31))
    assert uids_of(retriever.search("jazz", Criteria(period=last_month))) == ["jazz-passe"]


def test_filters_apply_to_all_chunks_not_only_the_closest(embedder):
    # 30 chunks très proches de la question mais hors critères, puis 1 chunk éloigné qui les respecte :
    # avec le réglage par défaut de LangChain (filtre sur les 20 plus proches), il serait perdu.
    decoys = [make_chunk(f"jazz-{number}", "jazz jazz jazz", [date(2026, 11, 7)]) for number in range(30)]
    target = make_chunk("expo-pau", "exposition", [date(2026, 11, 7)], city="Pau",
                        department="Pyrénées-Atlantiques")
    retriever = EventRetriever(build_vectorstore(decoys + [target], embedder))
    assert uids_of(retriever.search("jazz", Criteria(period=UPCOMING, city="Pau"))) == ["expo-pau"]


def test_no_matching_event_returns_nothing_without_calling_the_model(retriever, embedder):
    assert retriever.search("jazz", Criteria(period=UPCOMING, city="Limoges")) == []
    assert embedder.query_calls == 0


def test_the_model_receives_the_question_without_the_place_and_period_already_filtered(retriever, embedder):
    retriever.search("Du théâtre à Poitiers", Criteria(period=UPCOMING, city="Poitiers"))
    assert embedder.queries == ["Du théâtre"]


def test_the_full_question_can_still_be_searched_for_comparison(retriever, embedder, monkeypatch):
    monkeypatch.setattr(config, "SEARCH_REDUCED_QUESTION", False)
    retriever.search("Du théâtre à Poitiers", Criteria(period=UPCOMING, city="Poitiers"))
    assert embedder.queries == ["Du théâtre à Poitiers"]
