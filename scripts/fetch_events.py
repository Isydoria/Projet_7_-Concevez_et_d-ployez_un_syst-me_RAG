"""Télécharge les événements Open Agenda (1 an d'historique + à venir) dans data/raw/.

Exemples, depuis la racine du projet :
    python -m scripts.fetch_events                              # Nouvelle-Aquitaine, date du jour
    python -m scripts.fetch_events --department Creuse          # petit échantillon pour développer
    python -m scripts.fetch_events --reference-date 2026-10-29  # date de référence fixée
"""

import argparse
import logging
from datetime import date

from rag import config
from rag.fetch import download_events


def parse_args() -> argparse.Namespace:
    """Lit les options de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Télécharge les événements Open Agenda dans data/raw/.")
    parser.add_argument("--region", default=config.DEFAULT_REGION,
                        help="Région (défaut : %(default)s).")
    parser.add_argument("--department",
                        help="Limite à un département, pour un échantillon (ex. Creuse).")
    parser.add_argument("--reference-date", type=date.fromisoformat, default=date.today(),
                        help="Date de référence AAAA-MM-JJ (défaut : aujourd'hui).")
    return parser.parse_args()


def main() -> None:
    """Configure les logs, lit les options et lance le téléchargement."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    download_events(args.reference_date, region=args.region, department=args.department)


if __name__ == "__main__":
    main()
