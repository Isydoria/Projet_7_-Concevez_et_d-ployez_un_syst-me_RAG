"""API REST de l'assistant Puls-Events (FastAPI) : la porte d'entrée HTTP du système RAG.

La logique métier reste dans rag/ (EventAssistant) : l'API reçoit une requête, la vérifie (modèles Pydantic),
la confie à l'assistant et renvoie du JSON. Routes : POST /ask, GET /health, GET /metadata, POST /rebuild.

Lancement, depuis la racine du projet :
    uvicorn api.main:app
Documentation interactive (Swagger) : http://127.0.0.1:8000/docs
"""

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response

from api.schemas import (AskRequest, AskResponse, ErrorResponse, HealthResponse, IndexInfo, MetadataResponse,
                         RebuildRequest, RebuildStatus)
from rag import config
from rag.assistant import EventAssistant
from rag.fetch import download_events
from rag.indexing import MANIFEST_NAME, build_index, load_manifest, vectorstore_dir_for

logger = logging.getLogger(__name__)
DEFAULT_INDEX = vectorstore_dir_for(config.REFERENCE_RAW_FILE)
RETRY_AFTER_S = "60"   # limite de débit de Mistral : la requête peut être relancée une minute plus tard


def load_event_assistant(index_dir: Path) -> EventAssistant:
    """Assistant sur un index existant ; erreur explicite si le dossier n'existe pas."""
    if not index_dir.is_dir():
        raise FileNotFoundError(f"index introuvable : {index_dir} (le construire avec python -m scripts.build_index)")
    return EventAssistant.from_index(index_dir)


def build_new_index(refresh_data: bool) -> Path:
    """Reconstruit l'index : depuis la photo enregistrée, ou après téléchargement des événements du jour."""
    raw_path = download_events(date.today()) if refresh_data else config.REFERENCE_RAW_FILE
    return build_index(raw_path)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def describe(error: Exception) -> str:
    """Cause d'une erreur en une ligne ; pour un refus de Mistral, son code HTTP."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"Mistral a refusé la requête (code {error.response.status_code}) : {error.response.text[:150]}"
    return f"{type(error).__name__} : {error}"[:300]


def mistral_unavailable(error: httpx.HTTPStatusError) -> HTTPException:
    """Refus de Mistral → 503 avec un message clair (le client de l'API n'a rien à corriger dans sa requête)."""
    status, text = error.response.status_code, error.response.text
    if status == 429 and "capacity" not in text.lower():
        return HTTPException(503, detail="Mistral : limite de débit atteinte (code 429) ; réessayer dans une minute.",
                             headers={"Retry-After": RETRY_AFTER_S})
    return HTTPException(503, detail=f"Mistral refuse la requête (code {status}) : le service de génération est "
                                     "indisponible pour l'instant.")


@dataclass
class RebuildState:
    """Avancement de la dernière reconstruction (lu par /metadata)."""
    state: str = "idle"
    refresh_data: bool | None = None
    started_at: str | None = None
    finished_at: str | None = None
    index: str | None = None
    message: str | None = None


class Service:
    """État partagé de l'API : l'assistant servi, son index et la reconstruction en cours."""

    def __init__(self, load_assistant: Callable[[Path], EventAssistant], rebuild_index: Callable[[bool], Path],
                 index_dir: Path):
        self.load_assistant, self.rebuild_index = load_assistant, rebuild_index
        self.index_dir, self.assistant, self.error = index_dir, None, None
        self.rebuild = RebuildState()
        self.lock = threading.Lock()

    def start(self) -> None:
        """Charge l'assistant une fois, au démarrage ; en cas d'échec, l'API démarre et /health le signale."""
        try:
            self.assistant, self.error = self.load_assistant(self.index_dir), None
        except Exception as error:   # index absent, clé manquante… : signalé, pas fatal
            self.assistant, self.error = None, describe(error)
            logger.error("Assistant non chargé : %s", self.error)

    def begin_rebuild(self, refresh_data: bool) -> bool:
        """Marque une reconstruction comme lancée ; faux si une autre est déjà en cours."""
        with self.lock:
            if self.rebuild.state == "running":
                return False
            self.rebuild = RebuildState(state="running", refresh_data=refresh_data, started_at=now())
            return True

    def run_rebuild(self, refresh_data: bool) -> None:
        """Reconstruit puis bascule sur le nouvel index ; en cas d'échec, l'ancien index continue de répondre."""
        try:
            index_dir = self.rebuild_index(refresh_data)
            assistant = self.load_assistant(index_dir)
        except Exception as error:
            self.rebuild.state, self.rebuild.message = "failed", describe(error)
            logger.error("Reconstruction échouée : %s", self.rebuild.message)
        else:
            self.index_dir, self.assistant, self.error = index_dir, assistant, None
            self.rebuild.state, self.rebuild.index = "succeeded", index_dir.name
            self.rebuild.message = "index reconstruit et chargé"
        finally:
            self.rebuild.finished_at = now()


def create_app(load_assistant: Callable[[Path], EventAssistant] = load_event_assistant,
               rebuild_index: Callable[[bool], Path] = build_new_index, index_dir: Path = DEFAULT_INDEX,
               env_file: Path | None = config.PROJECT_ROOT / ".env") -> FastAPI:
    """Application FastAPI ; le chargement de l'assistant et la reconstruction sont remplaçables (tests)."""
    service = Service(load_assistant, rebuild_index, index_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if env_file is not None:
            load_dotenv(env_file)   # MISTRAL_API_KEY, REBUILD_TOKEN (jamais affichés)
        service.start()
        yield

    app = FastAPI(
        title="Puls-Events : assistant de recommandation d'événements culturels",
        description=("Système RAG sur les événements Open Agenda de Nouvelle-Aquitaine : recherche filtrée dans "
                     "un index Faiss (embeddings Mistral), puis réponse rédigée par un modèle Mistral, avec les "
                     "événements cités et leurs liens."),
        version="1.0.0", lifespan=lifespan)
    app.state.service = service

    @app.post("/ask", response_model=AskResponse, tags=["Assistant"], summary="Poser une question à l'assistant",
              description=("Repère le lieu et la période de la question, cherche les 5 événements les plus proches "
                           "qui respectent ces critères, puis fait rédiger la réponse par le modèle. Renvoie le "
                           "texte, les événements cités (avec leurs liens), les critères repérés, les événements "
                           "trouvés et les métadonnées (modèle, durées, tokens)."),
              responses={422: {"description": "Requête mal formée : question vide ou trop longue, date invalide"},
                         503: {"model": ErrorResponse,
                               "description": "Index non chargé, ou Mistral indisponible (refus, débit dépassé)"}})
    def ask(request: AskRequest) -> dict:
        if service.assistant is None:
            raise HTTPException(503, detail=f"Assistant indisponible : index non chargé ({service.error})")
        try:
            answer = service.assistant.ask(request.question, today=request.today)
        except httpx.HTTPStatusError as error:   # refus de Mistral, remonté par LangChain
            raise mistral_unavailable(error) from error
        return answer.to_dict()

    @app.get("/health", response_model=HealthResponse, tags=["Service"], summary="État de l'API",
             description="« ok » si l'assistant est chargé et prêt à répondre ; sinon 503 et la cause.",
             responses={503: {"model": HealthResponse, "description": "Assistant non chargé"}})
    def health(response: Response) -> HealthResponse:
        if service.assistant is None:
            response.status_code = 503
            return HealthResponse(status="unavailable", index=service.index_dir.name, detail=service.error)
        return HealthResponse(status="ok", index=service.index_dir.name)

    @app.get("/metadata", response_model=MetadataResponse, tags=["Service"], summary="Ce que sert l'API",
             description=("Index servi (date des données, nombre d'événements et de chunks, date de construction), "
                          "modèles utilisés, région, et avancement de la dernière reconstruction."))
    def metadata() -> MetadataResponse:
        index = None
        if (service.index_dir / MANIFEST_NAME).exists():
            manifest = load_manifest(service.index_dir)
            index = IndexInfo(name=service.index_dir.name,
                              **{key: manifest.get(key) for key in ("reference_date", "built_at", "events", "chunks")})
        return MetadataResponse(index=index, chat_model=config.CHAT_MODEL, embedding_model=config.EMBEDDING_MODEL,
                                region=config.DEFAULT_REGION, rebuild=RebuildStatus(**asdict(service.rebuild)))

    @app.post("/rebuild", status_code=202, response_model=RebuildStatus, tags=["Administration"],
              summary="Reconstruire la base vectorielle",
              description=("Reconstruit l'index Faiss en arrière-plan (≈ 12 min, ≈ 6,6 millions de tokens "
                           "d'embeddings) : depuis les données enregistrées par défaut, ou après téléchargement des "
                           "événements du jour (`refresh_data`). Réponse immédiate (202) ; avancement dans "
                           "/metadata ; l'ancien index répond jusqu'à la bascule. Protégé : en-tête "
                           "`X-Rebuild-Token`, comparé à la variable REBUILD_TOKEN du serveur."),
              responses={401: {"model": ErrorResponse, "description": "Jeton absent ou invalide"},
                         403: {"model": ErrorResponse, "description": "Reconstruction désactivée (REBUILD_TOKEN absent)"},
                         409: {"model": ErrorResponse, "description": "Une reconstruction est déjà en cours"}})
    def rebuild(background: BackgroundTasks, request: RebuildRequest | None = None,
                x_rebuild_token: str | None = Header(default=None, description="Jeton de reconstruction")
                ) -> RebuildStatus:
        expected = os.getenv("REBUILD_TOKEN")
        if not expected:
            raise HTTPException(403, detail="Reconstruction désactivée : définir REBUILD_TOKEN dans .env pour l'activer.")
        if not x_rebuild_token or not secrets.compare_digest(x_rebuild_token.encode(), expected.encode()):
            raise HTTPException(401, detail="Jeton de reconstruction absent ou invalide (en-tête X-Rebuild-Token).")
        refresh_data = bool(request and request.refresh_data)
        if not service.begin_rebuild(refresh_data):
            raise HTTPException(409, detail="Une reconstruction est déjà en cours : suivre son avancement dans /metadata.")
        background.add_task(service.run_rebuild, refresh_data)
        return RebuildStatus(**asdict(service.rebuild))

    return app


app = create_app()
