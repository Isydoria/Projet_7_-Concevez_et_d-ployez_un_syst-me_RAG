"""Récupération des événements Open Agenda via l'API OpenDataSoft (Explore v2.1).

On utilise l'endpoint /exports, qui renvoie tout le résultat filtré en une seule requête :
l'endpoint /records est limité à 10 000 résultats (offset + limit <= 10 000).
"""

import gzip
import json
import logging
import re
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from rag import config

logger = logging.getLogger(__name__)

# Codes HTTP pour lesquels une nouvelle tentative a un sens (erreurs temporaires)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def build_where_clause(region: str, reference_date: date, department: str | None = None) -> str:
    """Construit le filtre ODSQL : zone + période (1 an d'historique et événements à venir).

    Les valeurs sont entre guillemets doubles : une apostrophe dans un nom
    (« Provence-Alpes-Côte d'Azur ») casserait un filtre écrit entre apostrophes.
    """
    start_date = reference_date - timedelta(days=config.HISTORY_DAYS)
    clauses = [
        f'location_region = "{region}"',
        f"lastdate_end >= date'{start_date.isoformat()}'",
    ]
    if department:
        clauses.append(f'location_department = "{department}"')
    return " and ".join(clauses)


def fetch_events(where: str, fields: list[str], max_attempts: int = 4,
                 timeout: float = 120.0) -> list[dict]:
    """Télécharge tous les événements correspondant au filtre, avec nouvelles tentatives.

    Une erreur temporaire (réseau, 429, 5xx) entraîne une nouvelle tentative après une pause
    de plus en plus longue (1 s, 2 s, 4 s). Une erreur de requête (autre 4xx) est levée tout de suite.
    """
    url = f"{config.OPENAGENDA_DATASET_URL}/exports/json"
    params = {"where": where, "select": ", ".join(fields)}

    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            is_retryable = (
                isinstance(error, httpx.TransportError)
                or error.response.status_code in RETRYABLE_STATUS_CODES
            )
            if not is_retryable or attempt == max_attempts:
                raise
            wait_seconds = 2 ** (attempt - 1)
            logger.warning("Tentative %d/%d échouée (%s) : nouvel essai dans %d s",
                           attempt, max_attempts, error, wait_seconds)
            time.sleep(wait_seconds)


def slugify(text: str) -> str:
    """Transforme un nom en morceau de nom de fichier : « Pyrénées-Atlantiques » -> « pyrenees-atlantiques »."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def save_raw_events(events: list[dict], metadata: dict, output_path: Path) -> None:
    """Enregistre les événements bruts et leurs métadonnées dans un fichier JSON compressé (gzip)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as file:
        json.dump({"metadata": metadata, "events": events}, file, ensure_ascii=False)


def load_raw_events(path: Path) -> tuple[dict, list[dict]]:
    """Relit un fichier brut et renvoie (métadonnées, événements)."""
    with gzip.open(path, "rt", encoding="utf-8") as file:
        payload = json.load(file)
    return payload["metadata"], payload["events"]


def download_events(reference_date: date, region: str = config.DEFAULT_REGION,
                    department: str | None = None,
                    output_dir: Path = config.RAW_DATA_DIR) -> Path:
    """Récupère les événements de la zone, les enregistre dans data/raw/ et renvoie le chemin du fichier."""
    where = build_where_clause(region, reference_date, department)
    logger.info("Récupération des événements : %s", where)
    events = fetch_events(where, config.OPENAGENDA_FIELDS)
    logger.info("%d événements récupérés", len(events))

    metadata = {
        "source": f"{config.OPENAGENDA_DATASET_URL}/exports/json",
        "where": where,
        "fields": config.OPENAGENDA_FIELDS,
        "reference_date": reference_date.isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(events),
    }
    zone = slugify(department or region)
    output_path = output_dir / f"events_{zone}_{reference_date.isoformat()}.json.gz"
    save_raw_events(events, metadata, output_path)
    logger.info("Données brutes enregistrées : %s", output_path)
    return output_path
