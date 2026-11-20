"""Évalue l'assistant sur le jeu de test annoté : les réponses, puis leur jugement (mesures sans LLM et Ragas).

Exemples, depuis la racine du projet :
    python -m scripts.evaluate_rag                    # les deux phases (≈ 1 h 20 avec la formule gratuite)
    python -m scripts.evaluate_rag --phase answers    # phase 1 : les réponses de l'assistant (≈ 10 min)
    python -m scripts.evaluate_rag --phase judge      # phase 2 : mesures et Ragas sur les réponses enregistrées
    python -m scripts.evaluate_rag --no-ragas         # mesures sans juge seulement (≈ 1 min après les réponses)

Résultats dans data/eval/ : rag_answers.json (réponses complètes), rag_evaluation.json (mesures, détail du
juge), rag_evaluation.md (résumé lisible). Chaque réponse et chaque jugement est enregistré dès qu'il est
obtenu : une relance reprend là où le calcul s'est arrêté. Pour tout recommencer, supprimer ces fichiers.

Refus de Mistral : limite de débit dépassée → attente de 60 s et nouvel essai (3 essais au plus) ;
refus d'accès (403) ou serveurs saturés → arrêt immédiat, les résultats déjà obtenus restent enregistrés.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import httpx
import openai
from dotenv import load_dotenv

from rag import config
from rag.assistant import EventAssistant
from rag.evaluation import RAGAS_METRICS, free_measures, is_judged, load_test_set, summarize
from rag.indexing import vectorstore_dir_for
from rag.judge import RequestPacer, judge_answer, make_client, make_embeddings, make_judge, similarity
from scripts.ask import use_utf8_output

DEFAULT_INDEX = vectorstore_dir_for(config.REFERENCE_RAW_FILE)
ANSWERS_FILE, EVALUATION_FILE, REPORT_FILE = "rag_answers.json", "rag_evaluation.json", "rag_evaluation.md"
RATE_LIMIT_ATTEMPTS = 3   # essais au plus quand la limite de débit est dépassée
RATE_LIMIT_WAIT_S = 60    # attente avant un nouvel essai


class EvaluationStopped(Exception):
    """Refus de Mistral qu'une attente ne résoudra pas, ou donnée manquante : le calcul s'arrête."""


def is_rate_limited(status: int, text: str) -> bool:
    """429 « débit dépassé » : attendre résout. 429 « serveurs saturés » (capacity) ou 403 (accès refusé) : non."""
    return status == 429 and "capacity" not in text.lower()


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- Phase 1 : réponses ----------------------------------------------------------------------------------

def ask_with_retry(assistant, question: dict, today: date, sleep):
    """Réponse de l'assistant ; attente puis nouvel essai si la limite de débit est dépassée, arrêt sinon."""
    for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
        try:
            return assistant.ask(question["question"], today=today)
        except httpx.HTTPStatusError as error:   # erreur de l'API Mistral, remontée par ChatMistralAI
            status, text = error.response.status_code, error.response.text
            if is_rate_limited(status, text) and attempt < RATE_LIMIT_ATTEMPTS:
                sleep(RATE_LIMIT_WAIT_S)
                continue
            raise EvaluationStopped(f"question {question['id']} : refus de Mistral ({status}) : {text[:200]}") from error


def run_answers(test_set: dict, assistant, path: Path, run_info: dict, sleep=time.sleep,
                pause_s: float = config.CHAT_MIN_INTERVAL_S) -> dict:
    """Phase 1 : pose chaque question à l'assistant (date du jour fixée) et enregistre la réponse complète."""
    today = date.fromisoformat(test_set["metadata"]["today"])
    answers = load_json(path) or {"metadata": {**run_info, "today": today.isoformat(), "started_at": now()},
                                  "answers": {}}
    if answers["metadata"]["model"] != run_info["model"]:
        raise EvaluationStopped(f"{path.name} contient les réponses de {answers['metadata']['model']}, pas de "
                                f"{run_info['model']} : supprimer le fichier pour recommencer")
    asked = 0
    for question in test_set["questions"]:
        key = str(question["id"])
        if key in answers["answers"]:
            continue   # réponse déjà obtenue
        if asked:
            sleep(pause_s)   # formule gratuite : 2 requêtes par minute
        answer = ask_with_retry(assistant, question, today, sleep)
        answers["answers"][key] = answer.to_dict(include_contexts=True)
        save_json(path, answers)
        asked += 1
        print(f"[OK] réponse {key}/{len(test_set['questions'])}", flush=True)
    return answers


# --- Phase 2 : jugement ----------------------------------------------------------------------------------

def status_error(error: BaseException | None) -> openai.APIStatusError | None:
    """Erreur HTTP de l'API dans la chaîne des exceptions (Instructor peut envelopper l'erreur d'origine)."""
    while error is not None:
        if isinstance(error, openai.APIStatusError):
            return error
        error = error.__cause__ or error.__context__
    return None


async def with_retry(make_call, label: str, sleep):
    """Appel au juge ; attente puis nouvel essai si la limite de débit est dépassée, arrêt sinon."""
    for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
        try:
            return await make_call()
        except Exception as error:
            api_error = status_error(error)
            if api_error is None:
                raise   # erreur du programme, pas un refus de Mistral : elle doit se voir
            if is_rate_limited(api_error.status_code, str(api_error)) and attempt < RATE_LIMIT_ATTEMPTS:
                await sleep(RATE_LIMIT_WAIT_S)
                continue
            raise EvaluationStopped(f"{label} : refus de Mistral ({api_error.status_code}) : "
                                    f"{str(api_error)[:200]}") from error


async def run_judgement(test_set: dict, answers: dict, judge_llm, embeddings, path: Path, judge_model: str,
                        use_ragas: bool = True, sleep=asyncio.sleep) -> dict:
    """Phase 2 : mesures sans LLM, similarité et, là où elles ont un sens, métriques Ragas ; enregistre tout."""
    evaluation = load_json(path) or {"metadata": {"judge_model": judge_model, "started_at": now()}, "questions": {}}
    if evaluation["metadata"]["judge_model"] != judge_model:
        raise EvaluationStopped(f"{path.name} contient les jugements de {evaluation['metadata']['judge_model']}, "
                                f"pas de {judge_model} : supprimer le fichier pour changer de juge")
    for question in test_set["questions"]:
        key = str(question["id"])
        done = evaluation["questions"].get(key)
        if done and (done["ragas_status"] != "skipped" or not use_ragas):
            continue   # déjà jugée (une question sautée sans Ragas est rejugée si Ragas est demandé)
        answer = answers["answers"].get(key)
        if answer is None:
            raise EvaluationStopped(f"réponse {key} manquante : lancer d'abord --phase answers")
        result = free_measures(question, answer)
        result["similarity"] = await with_retry(
            lambda: similarity(answer["answer"], question["reference"], embeddings), f"question {key}, similarité",
            sleep)
        if not is_judged(question, answer):
            result.update(ragas_status="not_applicable", ragas=None)
        elif not use_ragas:
            result.update(ragas_status="skipped", ragas=None)
        else:
            ragas = await with_retry(lambda: judge_answer(question, answer, judge_llm, embeddings),
                                     f"question {key}, Ragas", sleep)
            result.update(ragas_status="judged", ragas=ragas)
        evaluation["questions"][key] = result
        evaluation["metadata"]["updated_at"] = now()
        save_json(path, evaluation)
        print(f"[OK] jugement {key}/{len(test_set['questions'])} ({result['ragas_status']})", flush=True)
    return evaluation


# --- Rapport ---------------------------------------------------------------------------------------------

def number(value) -> str:
    """Nombre à deux décimales, virgule décimale ; tiret si la mesure manque."""
    return "—" if value is None else f"{value:.2f}".replace(".", ",")


def render_report(test_set: dict, answers: dict, evaluation: dict, summary: dict) -> str:
    """Résumé lisible : conditions, synthèse, tableau par question, écarts, affirmations jugées non appuyées."""
    results = evaluation["questions"]
    retrieval, ragas = summary["retrieval"], summary["ragas"]
    metric_labels = {"faithfulness": "fidélité", "answer_relevancy": "pertinence",
                     "context_recall": "couverture documentaire", "factual_correctness": "exactitude factuelle"}
    lines = [
        "# Évaluation de l'assistant sur le jeu de test annoté", "",
        f"- Jeu de test : {len(test_set['questions'])} questions ; date du jour fixée : "
        f"{test_set['metadata']['today']} ; données : `{test_set['metadata']['source']}`",
        f"- Réponses : modèle `{answers['metadata']['model']}`, index `{answers['metadata']['index']}` "
        f"(obtenues à partir du {answers['metadata']['started_at']})",
        f"- Juge Ragas : `{evaluation['metadata']['judge_model']}` (jugement mis à jour le "
        f"{evaluation['metadata'].get('updated_at', '—')})", "",
        "## Synthèse", "",
        "| Mesure | Valeur | Questions |", "|---|---|---|",
        f"| Comportement attendu respecté | {summary['behavior_ok']}/{summary['questions']} | {summary['questions']} |",
        f"| Recherche : événement attendu trouvé (Exact Match sur les identifiants) | {number(retrieval['hit_rate'])} "
        f"| {retrieval['questions']} |",
        f"| Recherche : MRR (moyenne de 1 / rang du premier événement attendu) | {number(retrieval['mrr'])} "
        f"| {retrieval['questions']} |",
        f"| Recherche : précision (part des événements trouvés qui sont pertinents) | {number(retrieval['precision'])} "
        f"| {retrieval['questions']} |",
        f"| Recherche : rappel (part de ce qui pouvait être trouvé) | {number(retrieval['recall'])} "
        f"| {retrieval['questions']} |",
        f"| Citations : part des événements cités qui sont pertinents | {number(summary['citations']['precision'])} "
        f"| {summary['citations']['questions']} |",
        f"| Similarité entre la réponse et la référence (mistral-embed) | {number(summary['similarity'])} "
        f"| {summary['questions']} |",
        *[f"| Ragas : {metric_labels[metric]} | {number(ragas[metric])} | {ragas['questions']} |"
          for metric in RAGAS_METRICS],
        "", "## Par question", "",
        "| n° | Question | Catégorie | Comportement | Rang du 1er attendu | Citations pertinentes | Similarité "
        "| Fidélité | Pertinence | Couverture | Exactitude |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for question in test_set["questions"]:
        result = results.get(str(question["id"]))
        if result is None:
            lines.append(f"| {question['id']} | {question['question']} | {question['category']} | non jugée "
                         "| | | | | | | |")
            continue
        rank = result["retrieval"]["first_rank"] if result["retrieval"] else None
        citations = result["citations"]
        cited = f"{citations['cited_relevant']}/{citations['cited']}" if citations else "—"
        scores = result["ragas"] or {}
        lines.append(f"| {question['id']} | {question['question']} | {question['category']} "
                     f"| {'conforme' if result['behavior_ok'] else 'écart'} "
                     f"| {rank if rank else ('non trouvé' if result['retrieval'] else '—')} | {cited} "
                     f"| {number(result['similarity'])} | "
                     + " | ".join(number(scores.get(metric)) for metric in RAGAS_METRICS) + " |")
    problems = [(question, results[str(question["id"])]["behavior_problems"]) for question in test_set["questions"]
                if str(question["id"]) in results and results[str(question["id"])]["behavior_problems"]]
    lines += ["", "## Écarts de comportement", ""]
    lines += [f"- Question {question['id']} ({question['question']}) : {' ; '.join(found)}"
              for question, found in problems] or ["Aucun écart."]
    lines += ["", "## Affirmations jugées non appuyées par les événements transmis (fidélité)", "",
              "À lire avant d'interpréter la fidélité : une affirmation juste peut être jugée non appuyée "
              "(absence d'un événement, offre d'élargir la recherche).", ""]
    for question in test_set["questions"]:
        scores = (results.get(str(question["id"])) or {}).get("ragas") or {}
        rejected = [item for item in scores.get("faithfulness_statements", []) if not item["verdict"]]
        if rejected:
            lines += [f"**Question {question['id']}. {question['question']}**", ""]
            lines += [f"- « {item['statement']} » : {item['reason']}" for item in rejected] + [""]
    return "\n".join(lines).rstrip() + "\n"


# --- Lancement -------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évalue l'assistant sur le jeu de test annoté.")
    parser.add_argument("--phase", choices=["answers", "judge", "all"], default="all",
                        help="answers : réponses ; judge : jugement des réponses enregistrées ; all : les deux.")
    parser.add_argument("--no-ragas", action="store_true", help="Mesures sans juge LLM seulement (rapide).")
    parser.add_argument("--judge-model", default=config.JUDGE_MODEL, help="Modèle Mistral juge pour Ragas.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Dossier de l'index.")
    parser.add_argument("--output-dir", type=Path, default=config.EVAL_DIR, help="Dossier des résultats.")
    return parser.parse_args(argv)


async def judge_phase(test_set: dict, answers: dict, args: argparse.Namespace, path: Path) -> dict:
    """Phase 2 avec un vrai juge : client Mistral aux requêtes espacées, fermé à la fin."""
    pacer = RequestPacer()
    client = make_client(pacer)
    try:
        return await run_judgement(test_set, answers, make_judge(client, args.judge_model), make_embeddings(client),
                                   path, args.judge_model, use_ragas=not args.no_ragas)
    finally:
        await client.close()
        print(f"Requêtes envoyées à Mistral : {dict(pacer.counts)}")


def main(argv: list[str] | None = None) -> int:
    """Lance la ou les phases demandées. Renvoie 0 si tout s'est bien passé, 1 sinon."""
    use_utf8_output()
    args = parse_args(argv)
    load_dotenv(config.PROJECT_ROOT / ".env")   # chemin explicite : indépendant du dossier de lancement
    if not os.getenv("MISTRAL_API_KEY"):
        print("[ERREUR] MISTRAL_API_KEY absente : copier .env.example en .env et y renseigner la clé", file=sys.stderr)
        return 1
    test_set = load_test_set()
    answers_path, evaluation_path, report_path = (args.output_dir / name
                                                  for name in (ANSWERS_FILE, EVALUATION_FILE, REPORT_FILE))
    try:
        if args.phase in ("answers", "all"):
            if not args.index.is_dir():
                raise EvaluationStopped(f"index introuvable : {args.index} (python -m scripts.build_index)")
            run_answers(test_set, EventAssistant.from_index(args.index), answers_path,
                        {"model": config.CHAT_MODEL, "index": args.index.name})
        if args.phase in ("judge", "all"):
            answers = load_json(answers_path)
            if answers is None:
                raise EvaluationStopped(f"{answers_path.name} introuvable : lancer d'abord --phase answers")
            evaluation = asyncio.run(judge_phase(test_set, answers, args, evaluation_path))
            summary = summarize(list(evaluation["questions"].values()))
            report_path.write_text(render_report(test_set, answers, evaluation, summary), encoding="utf-8")
            print(f"[OK] résumé écrit dans {report_path}")
    except EvaluationStopped as error:
        print(f"[ERREUR] {error}\nLes résultats déjà obtenus sont enregistrés : relancer la même commande "
              "pour reprendre.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
