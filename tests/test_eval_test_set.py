"""Contrôle du jeu de test annoté (data/eval/test_set.json) sur la photo de référence : zéro tolérance.

Le jeu de test est produit par le projet. Chaque événement retenu doit exister dans les données nettoyées,
respecter les critères repérés dans la question (ville, département) et avoir une séance pendant la période.
Les catégories doivent être cohérentes avec les données : une question « sans résultat » ne trouve aucun
événement (l'assistant renvoie alors son message fixe) ; toute autre question en trouve au moins un
(l'assistant appelle alors le LLM). Si la photo de référence ou les règles de nettoyage changent,
ce test signale les questions à revoir.
"""

import json
from datetime import date

import pytest

from rag import config
from rag.criteria import build_city_index, extract_criteria, normalize
from rag.fetch import load_raw_events
from rag.preprocess import preprocess_events
from rag.retrieval import sessions_in_period

TEST_SET = json.loads(config.TEST_SET_FILE.read_text(encoding="utf-8"))
QUESTIONS = TEST_SET["questions"]
TODAY = date.fromisoformat(TEST_SET["metadata"]["today"])
IDS = [f"q{question['id']}" for question in QUESTIONS]

FIELDS = {"id", "question", "category", "expected_behavior", "reference", "relevant_uids", "excluded"}
# Comportement attendu pour chaque catégorie
BEHAVIOR_OF = {"answer": "recommend", "past": "recommend", "vague": "recommend", "no_result": "fixed_message",
               "absent_genre": "say_none", "off_topic": "refuse", "other_region": "redirect"}


@pytest.fixture(scope="module")
def reference_data():
    """Nettoie la photo de référence une seule fois ; index des villes construit comme celui de la recherche."""
    metadata, raw_events = load_raw_events(config.REFERENCE_RAW_FILE)
    events, _ = preprocess_events(raw_events, date.fromisoformat(metadata["reference_date"]))
    return {event["uid"]: event for event in events}, build_city_index(event["city"] for event in events)


def respects(event: dict, criteria) -> bool:
    """Même règle que la recherche filtrée : ville, département, au moins une séance pendant la période."""
    return ((not criteria.city or normalize(event["city"]) == normalize(criteria.city))
            and (not criteria.department or event["department"] == criteria.department)
            and bool(sessions_in_period(event["sessions"], criteria.period)))


# --- Structure ------------------------------------------------------------------------------------

def test_metadata_points_to_the_reference_photo():
    assert TEST_SET["metadata"]["source"] == config.REFERENCE_RAW_FILE.relative_to(config.PROJECT_ROOT).as_posix()
    metadata, _ = load_raw_events(config.REFERENCE_RAW_FILE)
    assert TEST_SET["metadata"]["today"] == metadata["reference_date"]


def test_twenty_questions_numbered_from_1_to_20():
    assert [question["id"] for question in QUESTIONS] == list(range(1, 21))
    assert len({question["question"] for question in QUESTIONS}) == 20   # pas de doublon


@pytest.mark.parametrize("question", QUESTIONS, ids=IDS)
def test_fields_and_values(question):
    assert set(question) == FIELDS
    assert question["question"].strip() and question["reference"].strip()
    assert question["category"] in TEST_SET["metadata"]["categories"]
    assert question["expected_behavior"] == BEHAVIOR_OF[question["category"]]
    if question["category"] == "vague":
        assert question["relevant_uids"] is None     # toute sortie de la période convient
    elif question["expected_behavior"] == "recommend":
        assert question["relevant_uids"], "une question à réponse retient au moins un événement"
    else:
        assert question["relevant_uids"] == []       # message fixe, absence, refus, autre région : rien d'attendu


# --- Cohérence avec les données -------------------------------------------------------------------

@pytest.mark.parametrize("question", QUESTIONS, ids=IDS)
def test_relevant_events_exist_and_match_the_question(question, reference_data):
    events, city_index = reference_data
    criteria = extract_criteria(question["question"], TODAY, city_index)
    problems = [f"{uid} : absent des données" if uid not in events else f"{uid} : hors des critères"
                for uid in question["relevant_uids"] or []
                if uid not in events or not respects(events[uid], criteria)]
    assert not problems, f"question {question['id']} ({criteria}) : {problems}"


@pytest.mark.parametrize("question", QUESTIONS, ids=IDS)
def test_excluded_events_exist_and_are_not_relevant(question, reference_data):
    events, _ = reference_data
    excluded = set(question["excluded"]) - {"*"}     # « * » : motif commun à plusieurs événements
    assert excluded <= set(events)
    assert not excluded & set(question["relevant_uids"] or [])


def test_manual_judgments_cover_every_question():
    """Classement humain des réponses : un verdict C / P / I motivé pour chaque question du jeu de test."""
    judgments = json.loads(config.MANUAL_JUDGMENTS_FILE.read_text(encoding="utf-8"))["judgments"]
    assert list(judgments) == [str(question["id"]) for question in QUESTIONS]
    assert all(judgment["verdict"] in {"C", "P", "I"} and judgment["reason"].strip()
               for judgment in judgments.values())


@pytest.mark.parametrize("question", QUESTIONS, ids=IDS)
def test_category_is_consistent_with_the_data(question, reference_data):
    """« Sans résultat » : aucun événement ne respecte les critères ; sinon, au moins un (appel au LLM)."""
    events, city_index = reference_data
    criteria = extract_criteria(question["question"], TODAY, city_index)
    found = sum(respects(event, criteria) for event in events.values())
    if question["category"] == "no_result":
        assert found == 0
    else:
        assert found > 0
