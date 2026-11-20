"""Lance l'API : prépare l'index s'il manque, puis démarre le serveur Uvicorn.

Index absent : téléchargement de l'index publié (≈ 1 min, gratuit, empreinte SHA-256 vérifiée) ; en cas
d'échec, construction à partir des données de référence (≈ 12 min, ≈ 6,6 millions de tokens d'embeddings
sur la clé MISTRAL_API_KEY). C'est la commande lancée par le conteneur Docker ; elle sert aussi hors Docker.

Depuis la racine du projet :
    python -m scripts.serve                     # http://127.0.0.1:8000 (documentation : /docs)
    python -m scripts.serve --host 0.0.0.0      # joignable depuis l'extérieur (conteneur)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv

from rag import config
from rag.indexing import MANIFEST_NAME, build_index, download_index, vectorstore_dir_for

logger = logging.getLogger("serve")
DEFAULT_INDEX = vectorstore_dir_for(config.REFERENCE_RAW_FILE)


def prepare_index(index_dir: Path = DEFAULT_INDEX, download=download_index, build=build_index) -> str:
    """Rend l'index disponible et dit comment : « présent », « téléchargé » ou « construit »."""
    if (index_dir / MANIFEST_NAME).exists():
        return "présent"
    try:
        download(config.INDEX_ARCHIVE_URL, config.INDEX_ARCHIVE_SHA256, index_dir)
        return "téléchargé"
    except (httpx.HTTPError, ValueError, OSError) as error:
        logger.warning("Téléchargement de l'index impossible (%s) : construction à partir des données", error)
    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError("index absent et non téléchargeable : renseigner MISTRAL_API_KEY pour le construire")
    build(config.REFERENCE_RAW_FILE)   # écrit dans vectorstore/<zone>_<date>/, c'est-à-dire index_dir
    return "construit"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prépare l'index s'il manque, puis lance l'API.")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute (0.0.0.0 dans un conteneur).")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv(config.PROJECT_ROOT / ".env")   # sans effet dans le conteneur : les variables y sont déjà
    try:
        logger.info("Index %s : %s", DEFAULT_INDEX.name, prepare_index())
    except RuntimeError as error:
        logger.error("%s", error)
        return 1
    uvicorn.run("api.main:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
