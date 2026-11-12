"""Pose une question à l'assistant et affiche la réponse, ses sources et ses métadonnées.

Exemples, depuis la racine du projet :
    python -m scripts.ask "Un concert de jazz à Bordeaux ce week-end"
    python -m scripts.ask "Du théâtre à Poitiers" --today 2026-10-29    # date fixée : résultats reproductibles
    python -m scripts.ask "Une exposition à Limoges" --json              # réponse complète en JSON
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import httpx
from dotenv import load_dotenv

from rag import config
from rag.assistant import AssistantAnswer, EventAssistant
from rag.criteria import Criteria, Period
from rag.indexing import vectorstore_dir_for
from rag.prompts import format_criteria

DEFAULT_INDEX = vectorstore_dir_for(config.REFERENCE_RAW_FILE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lit la question et les options de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Pose une question à l'assistant Puls-Events.")
    parser.add_argument("question", help="Question en français, entre guillemets.")
    parser.add_argument("--today", type=date.fromisoformat, default=None,
                        help="Date du jour au format AAAA-MM-JJ (défaut : la vraie date).")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                        help="Dossier de l'index (défaut : l'index de la photo de référence).")
    parser.add_argument("--json", action="store_true", help="Affiche la réponse complète en JSON.")
    return parser.parse_args(argv)


def use_utf8_output() -> None:
    """Sortie en UTF-8 : sous Windows, une sortie redirigée vers un fichier ne l'est pas, et certains
    titres (« ◯ ~ Ode à la douceur ») feraient planter l'affichage."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def criteria_text(criteria: dict) -> str:
    """Critères de la réponse, rédigés comme dans le prompt."""
    period = criteria["period"]
    start, end = (date.fromisoformat(value) if value else None for value in (period["start"], period["end"]))
    return format_criteria(Criteria(Period(period["label"], start, end), criteria["city"], criteria["department"]))


def metadata_line(metadata: dict) -> str:
    """Modèle, nombre d'événements, durées et tokens, sur une ligne."""
    if not metadata["llm_called"]:
        return f"Aucun événement trouvé ; recherche {metadata['retrieval_ms']:.0f} ms ; sans appel au LLM"
    reasoning = f" (dont {metadata['reasoning_tokens']} de réflexion)" if metadata.get("reasoning_tokens") else ""
    return (f"{metadata['model']} ; {metadata['retrieved_events']} événements trouvés ; "
            f"recherche {metadata['retrieval_ms']:.0f} ms ; génération {metadata['generation_ms']:.0f} ms ; "
            f"{metadata['input_tokens']} tokens en entrée, {metadata['output_tokens']} en sortie{reasoning}")


def format_answer(answer: AssistantAnswer) -> str:
    """Réponse lisible : question, critères, texte, sources avec leurs liens, puis les métadonnées."""
    lines = [f"Question : {answer.question}", f"Critères : {criteria_text(answer.criteria)}", "", answer.answer]
    if answer.metadata.get("finish_reason") == "length":
        lines += ["", "[ATTENTION] Réponse coupée : le plafond de tokens (réflexion comprise) a été atteint."]
    if answer.sources:
        lines += ["", "Sources :"]
        for source in answer.sources:
            place = ", ".join(part for part in [source["location_name"], source["city"]] if part)
            lines += [f"[{source['ref']}] {source['title']} — {place}", f"    {source['url']}"]
    lines += ["", metadata_line(answer.metadata)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Pose la question et affiche la réponse. Renvoie 0 si tout s'est bien passé, 1 sinon."""
    use_utf8_output()
    args = parse_args(argv)
    load_dotenv(config.PROJECT_ROOT / ".env")  # chemin explicite : indépendant du dossier de lancement
    if not os.getenv("MISTRAL_API_KEY"):
        print("[ERREUR] MISTRAL_API_KEY absente : copier .env.example en .env et y renseigner la clé",
              file=sys.stderr)
        return 1
    if not args.index.is_dir():
        print(f"[ERREUR] Index introuvable : {args.index}. Le construire avec : python -m scripts.build_index",
              file=sys.stderr)
        return 1

    assistant = EventAssistant.from_index(args.index)
    try:
        answer = assistant.ask(args.question, today=args.today)
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 429:
            raise
        print("[ERREUR] Limite de débit Mistral atteinte : réessayez dans une minute. Si l'erreur persiste, "
              "vérifiez les limites du modèle sur https://admin.mistral.ai (page Limits).", file=sys.stderr)
        return 1
    print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2) if args.json else format_answer(answer))
    return 0


if __name__ == "__main__":
    sys.exit(main())
