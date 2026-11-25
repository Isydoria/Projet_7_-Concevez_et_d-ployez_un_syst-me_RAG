"""Enregistre les vraies réponses des scénarios de démonstration (plan B si Mistral est indisponible).

Pose chaque scénario à l'assistant (date du jour fixée) et enregistre la réponse complète dans
data/demo/fallback_answers.json. Pendant la démonstration, si Mistral refuse ou est injoignable, l'API
renvoie ces réponses, signalées par metadata.recorded_at et l'en-tête X-Answer-Source: recorded.

Depuis la racine du projet :
    python -m scripts.record_demo                 # 3 appels à Mistral, espacés de 31 s
    python -m scripts.record_demo --from-eval data/eval/question_reduite/rag_answers.json
                                                  # sans appel : réponses déjà obtenues lors d'une évaluation
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from rag import config
from rag.assistant import EventAssistant
from rag.indexing import vectorstore_dir_for
from scripts.ask import use_utf8_output

DEMO_TODAY = date(2026, 10, 29)   # date des données de référence : réponses reproductibles
# Trois questions du jeu de test annoté (data/eval/test_set.json), une par comportement à montrer
DEMO_QUESTIONS = [
    "Un opéra à Limoges ce week-end",          # Q16 honnêteté : aucun opéra, il le dit et propose autre chose
    "Du théâtre à Brive-la-Gaillarde",         # Q4 amélioration mesurée : recherche sans le lieu ni la période
    "Une exposition à Toulouse en novembre",   # Q18 périmètre : seule la Nouvelle-Aquitaine est couverte
]


def save(entries: list[dict], path: Path) -> None:
    """Écrit les réponses enregistrées au format relu par l'API."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"answers": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record(assistant, path: Path, questions: list[str] = DEMO_QUESTIONS, today: date = DEMO_TODAY,
           sleep=time.sleep, pause_s: float = config.CHAT_MIN_INTERVAL_S) -> list[dict]:
    """Pose chaque question et enregistre les réponses obtenues (arrêt au premier refus de Mistral)."""
    entries = []
    try:
        for position, question in enumerate(questions):
            if position:
                sleep(pause_s)   # formule gratuite : 2 requêtes par minute
            answer = assistant.ask(question, today=today)
            entries.append({"question": question, "today": today.isoformat(),
                            "recorded_at": datetime.now().isoformat(timespec="seconds"), "answer": answer.to_dict()})
            print(f"[OK] {question}", flush=True)
    finally:
        if entries:   # les réponses obtenues sont gardées, même si une question suivante échoue
            save(entries, path)
    return entries


def copy_from_evaluation(eval_path: Path, questions: list[str] = DEMO_QUESTIONS) -> list[dict]:
    """Reprend les réponses déjà obtenues lors d'une évaluation (scripts/evaluate_rag.py), sans appel.

    La date d'enregistrement est celle du lancement de l'évaluation ; les textes transmis au LLM, qui ne
    servent qu'au jugement, sont retirés.
    """
    run = json.loads(eval_path.read_text(encoding="utf-8"))
    by_question = {answer["question"]: answer for answer in run["answers"].values()}
    missing = [question for question in questions if question not in by_question]
    if missing:
        raise ValueError(f"Questions absentes de {eval_path} : {missing}")
    entries = []
    for question in questions:
        answer = {key: value for key, value in by_question[question].items() if key != "contexts"}
        entries.append({"question": question, "today": answer["metadata"]["today"],
                        "recorded_at": run["metadata"]["started_at"], "answer": answer})
    return entries


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description="Enregistre les réponses de secours de la démonstration.")
    parser.add_argument("--from-eval", type=Path, default=None,
                        help="fichier de réponses d'une évaluation à reprendre, sans appel à Mistral")
    args = parser.parse_args(argv)
    if args.from_eval:
        entries = copy_from_evaluation(args.from_eval)
        save(entries, config.DEMO_FALLBACK_FILE)
    else:
        load_dotenv(config.PROJECT_ROOT / ".env")
        assistant = EventAssistant.from_index(vectorstore_dir_for(config.REFERENCE_RAW_FILE))
        try:
            entries = record(assistant, config.DEMO_FALLBACK_FILE)
        except httpx.HTTPError as error:
            print(f"[ERREUR] Mistral indisponible : {error}", file=sys.stderr)
            return 1
    print(f"[OK] {len(entries)} réponses enregistrées dans {config.DEMO_FALLBACK_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
