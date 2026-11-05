"""Tests des données de l'index de référence : tous les événements sont indexés (énoncé, étape 3).

L'index n'est pas versionné (sa construction appelle l'API Mistral) : ces tests sont ignorés
tant qu'il n'a pas été construit avec « python -m scripts.build_index ».
"""

from datetime import date

import numpy as np
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag import config, indexing
from rag.chunking import METADATA_FIELDS
from rag.fetch import load_raw_events
from rag.preprocess import preprocess_events

INDEX_DIR = indexing.vectorstore_dir_for(config.REFERENCE_RAW_FILE)
pytestmark = pytest.mark.skipif(not (INDEX_DIR / indexing.MANIFEST_NAME).exists(),
                                reason="index de référence non construit (python -m scripts.build_index)")


@pytest.fixture(scope="module")
def store():
    # Aucune recherche n'est faite : un faux modèle suffit pour relire l'index (pas de clé nécessaire)
    return indexing.load_vectorstore(INDEX_DIR, embedder=DeterministicFakeEmbedding(size=1024))


@pytest.fixture(scope="module")
def manifest():
    return indexing.load_manifest(INDEX_DIR)


@pytest.fixture(scope="module")
def chunks(store):
    return {doc_id: store.docstore.search(doc_id) for doc_id in store.index_to_docstore_id.values()}


def test_one_vector_per_chunk(store, manifest, chunks):
    assert store.index.ntotal == len(chunks) == manifest["chunks"]


def test_every_cleaned_event_is_indexed(chunks):
    metadata, raw_events = load_raw_events(config.REFERENCE_RAW_FILE)
    events, _ = preprocess_events(raw_events, date.fromisoformat(metadata["reference_date"]))
    assert {chunk.metadata["uid"] for chunk in chunks.values()} == {event["uid"] for event in events}


def test_chunk_ids_are_stable(chunks):
    assert all(doc_id == f"{chunk.metadata['uid']}#{chunk.metadata['chunk_index']}"
               for doc_id, chunk in chunks.items())


def test_every_chunk_has_its_metadata(chunks):
    expected = set(METADATA_FIELDS) | {"chunk_index", "chunk_count"}
    assert all(expected <= set(chunk.metadata) for chunk in chunks.values())


def test_index_is_exact_inner_product_with_mistral_dimension(store, manifest):
    assert type(store.index).__name__ == "IndexFlatIP"
    assert store.index.d == manifest["dimension"] == 1024
    assert manifest["embedding_model"] == config.EMBEDDING_MODEL


def test_vectors_are_normalized(store):
    sample = store.index.reconstruct_n(0, min(500, store.index.ntotal))
    assert np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-3)
