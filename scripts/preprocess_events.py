"""Nettoie les événements bruts (data/raw/) et enregistre le résultat dans data/processed/.

Exemples, depuis la racine du projet :
    python -m scripts.preprocess_events                                                  # photo de référence
    python -m scripts.preprocess_events --input data/raw/events_creuse_2026-10-29.json.gz
"""

import argparse
import logging
from pathlib import Path

from rag import config
from rag.preprocess import preprocess_file


def parse_args() -> argparse.Namespace:
    """Lit les options de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Nettoie les événements bruts dans data/processed/.")
    parser.add_argument("--input", type=Path, default=config.REFERENCE_RAW_FILE,
                        help="Fichier brut à nettoyer (défaut : la photo de référence).")
    return parser.parse_args()


def main() -> None:
    """Configure les logs, lit les options et lance le nettoyage."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    preprocess_file(parse_args().input)


if __name__ == "__main__":
    main()
