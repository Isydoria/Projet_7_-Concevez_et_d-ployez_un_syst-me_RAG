"""Construit l'index vectoriel : nettoyage -> découpage -> vectorisation Mistral -> index Faiss.

Exemples, depuis la racine du projet :
    python -m scripts.build_index                                                    # photo de référence (≈ 10 min, ≈ 0,45 $)
    python -m scripts.build_index --input data/raw/events_creuse_2026-10-29.json.gz  # échantillon (≈ 10 s)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag import config
from rag.indexing import build_index


def parse_args() -> argparse.Namespace:
    """Lit les options de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Construit l'index vectoriel Faiss dans vectorstore/.")
    parser.add_argument("--input", type=Path, default=config.REFERENCE_RAW_FILE,
                        help="Fichier brut à indexer (défaut : la photo de référence).")
    return parser.parse_args()


def main() -> int:
    """Configure les logs, vérifie la clé, puis construit l'index. Renvoie 0 si tout s'est bien passé."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # évite une ligne de log par requête HTTP
    load_dotenv(config.PROJECT_ROOT / ".env")  # chemin explicite : indépendant du dossier de lancement
    if not os.getenv("MISTRAL_API_KEY"):
        logging.error("MISTRAL_API_KEY absente : copier .env.example en .env et y renseigner la clé")
        return 1
    build_index(args.input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
