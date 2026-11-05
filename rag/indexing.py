"""Vectorisation des chunks avec Mistral et construction de l'index Faiss (Flat, produit scalaire).

Pipeline complet d'indexation, rejoué à chaque reconstruction :
données brutes -> nettoyage -> découpage -> vectorisation (par lots) -> index Faiss + manifeste.
"""

import json
import logging
import time
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_mistralai import MistralAIEmbeddings

from rag import config
from rag.chunking import chunk_events
from rag.fetch import load_raw_events
from rag.preprocess import preprocess_events

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def make_embedder() -> MistralAIEmbeddings:
    """Modèle d'embedding Mistral ; la clé est lue dans la variable d'environnement MISTRAL_API_KEY."""
    return MistralAIEmbeddings(model=config.EMBEDDING_MODEL)


def make_batches(texts: list[str], max_chars: int) -> list[list[str]]:
    """Regroupe les textes, dans l'ordre, en lots d'au plus max_chars caractères (un lot = une requête)."""
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if current and size + len(text) > max_chars:
            batches.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        batches.append(current)
    return batches


def embed_in_batches(texts: list[str], embedder: Embeddings,
                     max_chars: int = config.EMBED_BATCH_MAX_CHARS,
                     min_interval: float = config.EMBED_MIN_INTERVAL_S,
                     sleep: Callable[[float], None] = time.sleep,
                     clock: Callable[[], float] = time.monotonic) -> list[list[float]]:
    """Vectorise les textes par lots, en espaçant les requêtes d'au moins min_interval secondes.

    sleep et clock sont des paramètres pour que les tests puissent simuler le temps.
    """
    batches = make_batches(texts, max_chars)
    vectors: list[list[float]] = []
    last_call: float | None = None
    for number, batch in enumerate(batches, start=1):
        if last_call is not None:
            remaining = min_interval - (clock() - last_call)
            if remaining > 0:
                sleep(remaining)
        last_call = clock()
        vectors.extend(embedder.embed_documents(batch))
        if number % 50 == 0 or number == len(batches):
            logger.info("Vectorisation : lot %d/%d (%d textes)", number, len(batches), len(vectors))
    return vectors


def build_vectorstore(documents: list[Document], embedder: Embeddings) -> FAISS:
    """Vectorise les chunks et construit l'index Faiss exact (Flat, produit scalaire = cosinus)."""
    texts = [document.page_content for document in documents]
    vectors = embed_in_batches(texts, embedder)
    return FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embedder,
        metadatas=[document.metadata for document in documents],
        ids=[document.id for document in documents],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def vectorstore_dir_for(raw_path: Path, root: Path = config.VECTORSTORE_DIR) -> Path:
    """Dossier de l'index d'un fichier brut : events_creuse_2026-10-29.json.gz -> <root>/creuse_2026-10-29."""
    return root / raw_path.name.removeprefix("events_").removesuffix(".json.gz")


def build_index(raw_path: Path = config.REFERENCE_RAW_FILE, output_root: Path = config.VECTORSTORE_DIR,
                embedder: Embeddings | None = None) -> Path:
    """Pipeline complet : nettoyage -> découpage -> vectorisation -> index Faiss sauvegardé avec son manifeste."""
    raw_metadata, raw_events = load_raw_events(raw_path)
    events, cleaning_report = preprocess_events(raw_events, date.fromisoformat(raw_metadata["reference_date"]))
    documents, chunking_report = chunk_events(events)
    logger.info("%d événements nettoyés -> %d chunks", len(events), len(documents))

    started = time.monotonic()
    vectorstore = build_vectorstore(documents, embedder or make_embedder())
    embedding_seconds = round(time.monotonic() - started, 1)

    output_dir = vectorstore_dir_for(raw_path, output_root)
    vectorstore.save_local(str(output_dir))
    manifest = {
        "source_file": raw_path.name,
        "reference_date": raw_metadata["reference_date"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": config.EMBEDDING_MODEL,
        "dimension": vectorstore.index.d,
        "index_type": type(vectorstore.index).__name__,
        "distance": "produit scalaire (= cosinus, vecteurs normalisés)",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "events": len(events),
        "chunks": vectorstore.index.ntotal,
        "embedding_seconds": embedding_seconds,
        "cleaning_report": cleaning_report,
        "chunking_report": chunking_report,
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Index enregistré : %s (%d vecteurs, %.0f s de vectorisation)",
                output_dir, vectorstore.index.ntotal, embedding_seconds)
    return output_dir


def load_vectorstore(index_dir: Path, embedder: Embeddings | None = None) -> FAISS:
    """Recharge un index construit par build_index.

    Le docstore de LangChain est au format pickle, qui peut exécuter du code à la relecture :
    on ne recharge que des index produits par ce projet.
    """
    return FAISS.load_local(
        str(index_dir),
        embedder or make_embedder(),
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def load_manifest(index_dir: Path) -> dict:
    """Relit le manifeste d'un index."""
    return json.loads((index_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
