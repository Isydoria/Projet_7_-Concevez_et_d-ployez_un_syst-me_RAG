"""Évaluation de l'assistant sur le jeu de test annoté (data/eval/test_set.json).

Trois niveaux de jugement, qui se contrôlent mutuellement :
- des mesures gratuites, calculées par le code (ce module) : la recherche a-t-elle trouvé les événements
  attendus (identifiants) ? les événements cités répondent-ils à la question ? le comportement attendu
  (message fixe, refus, absence) est-il respecté ? similarité entre la réponse et la référence ;
- un LLM juge (Ragas), pour ce qui demande de comprendre le texte ;
- un classement humain des réponses, qui sert à vérifier le juge.

Les réponses évaluées ont la forme de AssistantAnswer.to_dict(include_contexts=True).
"""

import json
import re
from pathlib import Path

import numpy as np

from rag import config

# Comportements attendus : motifs cherchés dans la réponse (en minuscules, apostrophes droites)
REGION_RE = re.compile(r"nouvelle-aquitaine")
# Négation (« aucun », « je n'ai pas trouvé », « il n'y a pas »…), cherchée dans la première phrase seulement :
# c'est là que le modèle annonce s'il a trouvé ou non (« pas de réservation » plus loin ne compte pas)
NEGATIVE_RE = re.compile(r"\b(?:aucun\w*|pas|ni|ne|n'y a)\b")
FIRST_SENTENCE_RE = re.compile(r"[^.!?:\n]*")
PAST_RE = re.compile(r"termin|ont eu lieu|a eu lieu|pass[ée]")


def load_test_set(path: Path = config.TEST_SET_FILE) -> dict:
    """Jeu de test annoté : métadonnées (date du jour, photo de référence) et questions."""
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_measures(retrieved: list[str], relevant: list[str], places: int = config.TOP_K_EVENTS) -> dict:
    """Mesures de la recherche, sur les identifiants (uid) : exactes, sans LLM.

    - hit : au moins un événement attendu parmi les trouvés (Exact Match sur les identifiants) ;
    - first_rank : rang du premier événement attendu (None s'il n'est pas trouvé) ;
    - reciprocal_rank : 1 / rang (0 s'il n'est pas trouvé) ; sa moyenne sur les questions est le MRR ;
    - precision : part des événements trouvés qui répondent à la question (le bruit) ;
    - recall : part de ce qui pouvait être trouvé, c'est-à-dire du plus petit des deux nombres
      « places disponibles » et « événements attendus » (une seule pièce à Brive : la trouver suffit).
    """
    relevant_set = set(relevant)
    ranks = [rank for rank, uid in enumerate(retrieved, start=1) if uid in relevant_set]
    found = len(ranks)
    return {
        "hit": found > 0,
        "first_rank": ranks[0] if ranks else None,
        "reciprocal_rank": 1 / ranks[0] if ranks else 0.0,
        "precision": found / len(retrieved) if retrieved else 0.0,
        "recall": found / min(len(relevant_set), places),
    }


def citation_measures(cited: list[str], relevant: list[str]) -> dict:
    """Événements cités dans la réponse : combien répondent à la question (précision, None si rien n'est cité)."""
    cited_relevant = sum(uid in set(relevant) for uid in cited)
    return {"cited": len(cited), "cited_relevant": cited_relevant,
            "precision": cited_relevant / len(cited) if cited else None}


def behavior_problems(question: dict, answer: dict) -> list[str]:
    """Écarts au comportement attendu pour la question ; liste vide : comportement conforme."""
    behavior = question["expected_behavior"]
    llm_called = answer["metadata"]["llm_called"]
    text = answer["answer"].lower().replace("’", "'").replace("*", "")
    cited = answer["sources"]
    if behavior == "fixed_message":
        return [] if not llm_called else ["LLM appelé au lieu du message fixe"]
    if not llm_called:
        return ["message fixe au lieu d'une réponse du LLM"]
    problems = []
    if behavior in ("refuse", "redirect"):
        # Refus (hors sujet) : aucune recommandation ; redirection (autre région) : une alternative est permise
        if behavior == "refuse" and cited:
            problems.append("événement cité malgré le refus")
        if not REGION_RE.search(text):
            problems.append("manque : Nouvelle-Aquitaine")
    elif behavior == "say_none":
        if not NEGATIVE_RE.search(FIRST_SENTENCE_RE.match(text.strip()).group()):
            problems.append("ne dit pas qu'il n'y en a pas")   # une alternative citée reste permise
    elif behavior == "recommend":
        if not cited:
            problems.append("aucun événement cité")
        if question["category"] == "past" and not PAST_RE.search(text):
            problems.append("ne présente pas les événements comme passés")
    return problems


def free_measures(question: dict, answer: dict) -> dict:
    """Mesures sans LLM d'une réponse : comportement, et recherche / citations si des événements sont attendus."""
    problems = behavior_problems(question, answer)
    relevant = question["relevant_uids"]
    expects_events = bool(relevant)            # None (question vague) ou [] (cas limite) : rien à mesurer
    retrieved = [event["uid"] for event in answer.get("retrieved", [])]
    cited = [source["uid"] for source in answer["sources"]]
    return {
        "id": question["id"],
        "category": question["category"],
        "behavior_ok": not problems,
        "behavior_problems": problems,
        "retrieval": retrieval_measures(retrieved, relevant) if expects_events else None,
        "citations": citation_measures(cited, relevant) if expects_events else None,
    }


def cosine(a: list[float], b: list[float]) -> float:
    """Similarité cosinus entre deux vecteurs (1 : même direction ; 0 : sans rapport)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
