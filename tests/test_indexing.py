"""Tests unitaires de la vectorisation et de l'index (rag/indexing.py), sans aucun appel à Mistral."""

from pathlib import Path

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from rag import config, indexing
from rag.fetch import save_raw_events
from tests.factories import make_raw_event


class FakeEmbedder(Embeddings):
    """Remplace MistralAIEmbeddings : vecteurs déterministes, et trace de chaque appel (= une requête)."""

    def __init__(self, size: int = 8):
        self._inner = DeterministicFakeEmbedding(size=size)
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


# --- make_batches -------------------------------------------------------------------------------------

def test_make_batches_respects_the_size_limit_and_the_order():
    texts = ["a" * 30, "b" * 30, "c" * 30]
    assert indexing.make_batches(texts, max_chars=70) == [["a" * 30, "b" * 30], ["c" * 30]]


def test_make_batches_puts_a_too_long_text_alone():
    assert indexing.make_batches(["x" * 100, "y" * 10], max_chars=50) == [["x" * 100], ["y" * 10]]


# --- embed_in_batches ---------------------------------------------------------------------------------

def test_embed_in_batches_sends_one_request_per_batch_and_waits_between_them():
    embedder, sleeps = FakeEmbedder(), []
    texts = [f"texte {i} " + "t" * 25 for i in range(5)]   # 33 caractères chacun -> lots de 2, 2 et 1

    vectors = indexing.embed_in_batches(texts, embedder, max_chars=70, min_interval=1.1,
                                        sleep=sleeps.append, clock=lambda: 100.0)  # horloge figée

    assert [len(call) for call in embedder.calls] == [2, 2, 1]
    assert sleeps == [1.1, 1.1]                               # une pause avant chaque requête, sauf la 1re
    assert vectors == FakeEmbedder().embed_documents(texts)   # un vecteur par texte, dans l'ordre


def test_embed_in_batches_does_not_wait_when_requests_are_already_spaced():
    embedder, sleeps = FakeEmbedder(), []
    texts = [f"texte {i} " + "t" * 25 for i in range(5)]
    clock_values = iter([0.0, 5.0, 5.0, 10.0, 10.0])         # l'horloge avance de 5 s entre deux lots

    indexing.embed_in_batches(texts, embedder, max_chars=70, min_interval=1.1,
                              sleep=sleeps.append, clock=lambda: next(clock_values))

    assert sleeps == []


# --- build_vectorstore ------------------------------------------------------------------------------

def test_build_vectorstore_creates_an_exact_inner_product_index_with_our_ids():
    documents = [Document(id=f"{i}#0", page_content=f"Événement numéro {i}", metadata={"uid": str(i)})
                 for i in range(3)]
    store = indexing.build_vectorstore(documents, FakeEmbedder())

    assert type(store.index).__name__ == "IndexFlatIP"
    assert store.index.ntotal == 3
    assert sorted(store.index_to_docstore_id.values()) == ["0#0", "1#0", "2#0"]
    assert store.docstore.search("1#0").metadata == {"uid": "1"}


# --- vectorstore_dir_for, build_index, load_vectorstore --------------------------------------------------

def test_vectorstore_dir_is_named_after_the_raw_file(tmp_path):
    raw_path = Path("data/raw/events_creuse_2026-10-29.json.gz")
    assert indexing.vectorstore_dir_for(raw_path, tmp_path) == tmp_path / "creuse_2026-10-29"


def test_build_index_saves_index_and_manifest_then_reloads(tmp_path):
    raw_path = tmp_path / "events_test_2026-10-29.json.gz"
    long_html = "<p>" + " ".join(f"Phrase numéro {i} du programme." for i in range(150)) + "</p>"
    save_raw_events([make_raw_event(uid="court"),
                     make_raw_event(uid="long", title_fr="Festival", longdescription_fr=long_html)],
                    {"reference_date": "2026-10-29"}, raw_path)

    index_dir = indexing.build_index(raw_path, output_root=tmp_path / "vectorstore", embedder=FakeEmbedder())

    assert index_dir == tmp_path / "vectorstore" / "test_2026-10-29"
    assert {path.name for path in index_dir.iterdir()} == {"index.faiss", "index.pkl", "manifest.json"}
    manifest = indexing.load_manifest(index_dir)
    assert manifest["events"] == 2
    assert manifest["chunks"] == manifest["chunking_report"]["chunks"] > 2
    assert manifest["embedding_model"] == config.EMBEDDING_MODEL
    assert manifest["index_type"] == "IndexFlatIP"

    store = indexing.load_vectorstore(index_dir, embedder=FakeEmbedder())
    assert store.index.ntotal == manifest["chunks"]
    assert store.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT
