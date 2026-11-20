"""Modèles Pydantic de l'API : forme des requêtes et des réponses.

Ils servent deux fois : FastAPI refuse automatiquement une requête qui ne les respecte pas (code 422),
et la documentation Swagger (/docs) est générée à partir d'eux.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DEMO_QUESTION = "Un concert de jazz à Bordeaux ce week-end"


# --- /ask -----------------------------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Question posée à l'assistant."""
    question: str = Field(..., min_length=1, max_length=500, description="Question en français (1 à 500 caractères).",
                          examples=[DEMO_QUESTION])
    today: date | None = Field(None, description=(
        "Date du jour (AAAA-MM-JJ), qui fixe le sens de « ce week-end », « demain »… Par défaut : la vraie date. "
        "Avec 2026-10-29, date des données de référence, les réponses sont reproductibles."),
        examples=["2026-10-29"])

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Une question faite d'espaces est refusée comme une question vide."""
        if not value.strip():
            raise ValueError("la question est vide")
        return value.strip()


class Source(BaseModel):
    """Événement cité dans la réponse ; ses informations viennent des données, jamais du texte du modèle."""
    ref: int = Field(..., description="Référence [n] citée dans la réponse")
    uid: str
    title: str
    start: str | None = None
    end: str | None = None
    city: str | None = None
    location_name: str | None = None
    url: str | None = Field(None, description="Page de l'événement sur Open Agenda")
    score: float = Field(..., description="Similarité avec la question (cosinus)")


class RetrievedEvent(BaseModel):
    """Événement trouvé par la recherche et transmis au modèle, cité ou non."""
    ref: int = Field(..., description="Numéro [n] dans le prompt (= rang dans la recherche)")
    uid: str
    title: str
    score: float


class Period(BaseModel):
    label: str
    start: str | None = None
    end: str | None = None


class Criteria(BaseModel):
    """Critères repérés dans la question (par des règles) et appliqués comme filtres."""
    city: str | None = None
    department: str | None = None
    period: Period


class AnswerMetadata(BaseModel):
    """Conditions et coût de la réponse."""
    today: str
    model: str
    index: str
    region: str
    llm_called: bool = Field(..., description="Faux : aucun événement trouvé, message fixe sans appel au modèle")
    search_text: str | None = Field(None, description=(
        "Texte de la recherche par similarité : la question sans le lieu ni la période, déjà appliqués comme "
        "filtres"))
    retrieved_events: int
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = Field(None, description="« length » : réponse coupée par le plafond de tokens")


class AskResponse(BaseModel):
    """Réponse de l'assistant."""
    question: str
    answer: str = Field(..., description="Réponse rédigée, avec les références [n] des événements cités")
    sources: list[Source]
    criteria: Criteria
    metadata: AnswerMetadata
    retrieved: list[RetrievedEvent]


class ErrorResponse(BaseModel):
    """Erreur expliquée."""
    detail: str


# --- /health et /metadata -------------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    index: str | None = Field(None, description="Index servi par l'API")
    detail: str | None = Field(None, description="Cause, si l'assistant n'est pas disponible")


class IndexInfo(BaseModel):
    name: str
    reference_date: str | None = None
    built_at: str | None = None
    events: int | None = None
    chunks: int | None = None


class RebuildRequest(BaseModel):
    refresh_data: bool = Field(False, description=(
        "Faux (par défaut) : reconstruire l'index depuis les données enregistrées (photo de référence). "
        "Vrai : télécharger d'abord les événements du jour sur Open Agenda."))


class RebuildStatus(BaseModel):
    """Avancement de la dernière reconstruction."""
    state: Literal["idle", "running", "succeeded", "failed"]
    refresh_data: bool | None = None
    started_at: str | None = None
    finished_at: str | None = None
    index: str | None = Field(None, description="Index construit, en cas de réussite")
    message: str | None = None


class MetadataResponse(BaseModel):
    index: IndexInfo | None = Field(None, description="Index servi (manifeste)")
    chat_model: str
    embedding_model: str
    region: str
    rebuild: RebuildStatus
