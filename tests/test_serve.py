"""Tests unitaires de la préparation de l'index (rag/indexing.download_index, scripts/serve.py).

Aucun accès réseau : le téléchargement passe par un transport httpx simulé.
"""

import hashlib
import io
import zipfile

import httpx
import pytest

from rag.indexing import MANIFEST_NAME, download_index
from scripts import serve

FILES = {"index.faiss": b"vecteurs", "index.pkl": b"documents", MANIFEST_NAME: b'{"events": 3}'}


def make_archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def client_serving(content: bytes, status: int = 200) -> httpx.Client:
    """Client httpx dont toutes les requêtes reçoivent `content` (aucun accès réseau)."""
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, content=content)))


# --- Téléchargement vérifié -----------------------------------------------------------------------------

def test_download_extracts_the_index_when_the_checksum_matches(tmp_path):
    archive = make_archive(FILES)
    index_dir = tmp_path / "vectorstore" / "zone_2026-10-29"
    download_index("https://example.org/index.zip", hashlib.sha256(archive).hexdigest(), index_dir,
                   client=client_serving(archive))
    assert {path.name: path.read_bytes() for path in index_dir.iterdir()} == FILES


def test_download_refuses_an_archive_whose_checksum_differs(tmp_path):
    # Archive corrompue ou modifiée : index.pkl est un pickle, on ne charge que le fichier publié
    archive = make_archive(FILES)
    index_dir = tmp_path / "index"
    with pytest.raises(ValueError, match="empreinte"):
        download_index("https://example.org/index.zip", "0" * 64, index_dir, client=client_serving(archive))
    assert not index_dir.exists()


def test_download_refuses_unexpected_files_in_the_archive(tmp_path):
    archive = make_archive({**FILES, "../autre.py": b"code"})
    index_dir = tmp_path / "index"
    with pytest.raises(ValueError, match="contenu"):
        download_index("https://example.org/index.zip", hashlib.sha256(archive).hexdigest(), index_dir,
                       client=client_serving(archive))
    assert not index_dir.exists()


def test_download_reports_an_http_error(tmp_path):
    with pytest.raises(httpx.HTTPStatusError):
        download_index("https://example.org/index.zip", "0" * 64, tmp_path / "index",
                       client=client_serving(b"introuvable", status=404))


# --- Préparation avant le lancement de l'API ------------------------------------------------------------

def existing_index(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    return index_dir


def test_an_existing_index_is_used_as_is(tmp_path):
    calls = []
    status = serve.prepare_index(existing_index(tmp_path), download=lambda *args: calls.append("download"),
                                 build=lambda *args: calls.append("build"))
    assert status == "présent" and calls == []


def test_a_missing_index_is_downloaded(tmp_path):
    calls = []
    status = serve.prepare_index(tmp_path / "index", download=lambda *args: calls.append("download"),
                                 build=lambda *args: calls.append("build"))
    assert status == "téléchargé" and calls == ["download"]


def test_the_index_is_built_when_the_download_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "cle-factice")
    calls = []

    def failing_download(*args):
        raise ValueError("empreinte différente")

    status = serve.prepare_index(tmp_path / "index", download=failing_download,
                                 build=lambda *args: calls.append("build"))
    assert status == "construit" and calls == ["build"]


def test_no_index_without_download_nor_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    def failing_download(*args):
        raise httpx.ConnectError("hors ligne")

    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        serve.prepare_index(tmp_path / "index", download=failing_download, build=lambda *args: None)
