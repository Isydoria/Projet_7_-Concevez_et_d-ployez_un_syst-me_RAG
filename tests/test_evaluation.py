"""Tests unitaires des mesures d'évaluation (rag/evaluation.py), sur des réponses construites à la main.

Aucun appel à Mistral : les réponses imitent AssistantAnswer.to_dict(include_contexts=True).
"""

import json

import pytest

from rag.evaluation import (behavior_problems, citation_measures, cosine, free_measures, load_test_set,
                            retrieval_measures)


def make_answer(retrieved=(), cited=(), text="Voici une idée [1].", llm_called=True) -> dict:
    """Réponse de l'assistant, réduite aux champs utiles à l'évaluation."""
    return {
        "question": "question",
        "answer": text,
        "retrieved": [{"ref": rank, "uid": uid, "title": uid, "score": 0.8} for rank, uid in enumerate(retrieved, 1)],
        "sources": [{"ref": 1, "uid": uid} for uid in cited],
        "metadata": {"llm_called": llm_called},
    }


def make_question(category="answer", behavior="recommend", relevant=("a",)) -> dict:
    return {"id": 1, "question": "question", "category": category, "expected_behavior": behavior,
            "reference": "référence", "relevant_uids": None if relevant is None else list(relevant), "excluded": {}}


# --- Recherche ----------------------------------------------------------------------------------------

def test_retrieval_when_the_expected_event_is_found_first():
    measures = retrieval_measures(["a", "x", "y", "z", "w"], relevant=["a"])
    assert measures == {"hit": True, "first_rank": 1, "reciprocal_rank": 1.0, "precision": 0.2, "recall": 1.0}


def test_retrieval_when_the_expected_event_is_found_third():
    measures = retrieval_measures(["x", "y", "a", "z", "w"], relevant=["a", "b"])
    assert (measures["hit"], measures["first_rank"]) == (True, 3)
    assert measures["reciprocal_rank"] == pytest.approx(1 / 3)
    assert measures["recall"] == 0.5          # 1 trouvé sur les 2 qui pouvaient l'être


def test_retrieval_when_nothing_expected_is_found():
    measures = retrieval_measures(["x", "y", "z"], relevant=["a"])
    assert measures == {"hit": False, "first_rank": None, "reciprocal_rank": 0.0, "precision": 0.0, "recall": 0.0}


def test_recall_is_capped_by_the_number_of_places():
    # 27 concerts attendus, 5 places : en trouver 5 est le maximum possible
    relevant = [f"c{i}" for i in range(27)]
    assert retrieval_measures(relevant[:5], relevant)["recall"] == 1.0


def test_retrieval_without_any_event_retrieved():
    measures = retrieval_measures([], relevant=["a"])
    assert (measures["hit"], measures["precision"], measures["recall"]) == (False, 0.0, 0.0)


# --- Citations ----------------------------------------------------------------------------------------

def test_citations_counts_the_cited_events_that_answer_the_question():
    assert citation_measures(["a", "x"], relevant=["a", "b"]) == {"cited": 2, "cited_relevant": 1, "precision": 0.5}


def test_citation_precision_is_undefined_when_nothing_is_cited():
    assert citation_measures([], relevant=["a"]) == {"cited": 0, "cited_relevant": 0, "precision": None}


# --- Comportement --------------------------------------------------------------------------------------

def test_recommend_needs_at_least_one_cited_event():
    question = make_question()
    assert behavior_problems(question, make_answer(retrieved=["a"], cited=["a"])) == []
    assert behavior_problems(question, make_answer(retrieved=["a"], cited=[], text="Rien.")) == [
        "aucun événement cité"]


def test_fixed_message_means_the_llm_is_not_called():
    question = make_question("no_result", "fixed_message", relevant=())
    assert behavior_problems(question, make_answer(llm_called=False, text="Je n'ai trouvé aucun événement")) == []
    assert behavior_problems(question, make_answer(retrieved=["a"])) == ["LLM appelé au lieu du message fixe"]


def test_refusal_cites_nothing_and_recalls_the_region():
    question = make_question("off_topic", "refuse", relevant=())
    good = make_answer(retrieved=["a"], text="Je ne réponds qu'aux questions sur les événements en Nouvelle-Aquitaine.")
    assert behavior_problems(question, good) == []
    bad = make_answer(retrieved=["a"], cited=["a"], text="Il fera beau ! Allez voir ce concert [1].")
    assert behavior_problems(question, bad) == ["événement cité malgré le refus", "manque : Nouvelle-Aquitaine"]


def test_redirect_recalls_the_region_and_allows_an_alternative_there():
    question = make_question("other_region", "redirect", relevant=())
    text = "Je ne recommande que des événements en Nouvelle-Aquitaine. Voici une exposition à Bordeaux [1]."
    assert behavior_problems(question, make_answer(["a"], ["a"], text)) == []
    assert behavior_problems(question, make_answer(["a"], ["a"], "Voici une exposition à Toulouse [1].")) == [
        "manque : Nouvelle-Aquitaine"]


def test_say_none_needs_a_negative_answer_and_allows_an_alternative():
    question = make_question("absent_genre", "say_none", relevant=())
    answer = make_answer(retrieved=["a"], cited=["a"], text="Aucun opéra ce week-end, mais cette exposition [1].")
    assert behavior_problems(question, answer) == []
    assert behavior_problems(question, make_answer(retrieved=["a"], cited=["a"], text="Voici un opéra [1].")) == [
        "ne dit pas qu'il n'y en a pas"]


def test_say_none_accepts_the_usual_phrasings_of_the_model():
    # Formulation réelle de Ministral 8B (notebook 04, B7)
    question = make_question("absent_genre", "say_none", relevant=())
    text = "Je n’ai pas trouvé de concert de jazz dans les événements proposés. En revanche, voici [1]."
    assert behavior_problems(question, make_answer(["a"], ["a"], text)) == []


def test_say_none_looks_for_the_negation_in_the_first_sentence_only():
    # « pas de réservation » plus loin dans le texte ne veut pas dire « il n'y en a pas »
    question = make_question("absent_genre", "say_none", relevant=())
    text = "Voici un opéra ce samedi [1]. Il n'y a pas de réservation à faire."
    assert behavior_problems(question, make_answer(["a"], ["a"], text)) == ["ne dit pas qu'il n'y en a pas"]


def test_past_events_must_be_presented_as_finished():
    question = make_question("past", "recommend")
    assert behavior_problems(question, make_answer(["a"], ["a"], "Ce concert a eu lieu le 19 septembre [1].")) == []
    assert behavior_problems(question, make_answer(["a"], ["a"], "Allez voir ce concert [1].")) == [
        "ne présente pas les événements comme passés"]


# --- Synthèse par question -----------------------------------------------------------------------------

def test_free_measures_of_a_question_with_expected_events():
    measures = free_measures(make_question(relevant=["a"]), make_answer(retrieved=["x", "a"], cited=["a"]))
    assert measures["behavior_ok"] is True
    assert measures["retrieval"]["first_rank"] == 2
    assert measures["citations"]["precision"] == 1.0


def test_free_measures_skip_retrieval_when_no_event_is_expected():
    question = make_question("off_topic", "refuse", relevant=())
    measures = free_measures(question, make_answer(retrieved=["a"], text="Seulement en Nouvelle-Aquitaine."))
    assert measures["retrieval"] is None and measures["citations"] is None
    assert measures["behavior_ok"] is True


def test_free_measures_of_a_vague_question():
    measures = free_measures(make_question("vague", relevant=None), make_answer(retrieved=["a"], cited=["a"]))
    assert measures["retrieval"] is None and measures["behavior_ok"] is True


# --- Similarité et chargement --------------------------------------------------------------------------

def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 2.0]) == pytest.approx(0.0)
    assert cosine([3.0, 4.0], [3.0, 4.0]) == pytest.approx(1.0)   # vecteurs non normalisés


def test_load_test_set(tmp_path):
    path = tmp_path / "test_set.json"
    path.write_text(json.dumps({"metadata": {"today": "2026-10-29"}, "questions": [make_question()]},
                               ensure_ascii=False), encoding="utf-8")
    test_set = load_test_set(path)
    assert test_set["metadata"]["today"] == "2026-10-29" and len(test_set["questions"]) == 1
