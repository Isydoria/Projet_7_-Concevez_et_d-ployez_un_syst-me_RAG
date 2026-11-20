"""Fabriques de données de test, partagées par plusieurs fichiers de tests."""

from datetime import date

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.criteria import normalize


def make_raw_event(**overrides) -> dict:
    """Fabrique un événement brut Open Agenda valide ; chaque test ne modifie que ce qui l'intéresse."""
    event = {
        "uid": "1", "title_fr": "Concert de jazz", "description_fr": "Un quartet",
        "longdescription_fr": "<p>Un quartet</p>", "keywords_fr": ["Jazz"],
        "firstdate_begin": "2026-11-01T10:00:00+00:00", "lastdate_end": "2026-11-01T12:00:00+00:00",
        "timings": '[{"begin": "2026-11-01T11:00:00+01:00", "end": "2026-11-01T13:00:00+01:00"}]',
        "daterange_fr": "Dimanche 1 novembre, 11h00", "location_name": "Auditorium",
        "location_address": "9 cours Clemenceau", "location_city": "Bordeaux",
        "location_postalcode": "33000", "location_department": "Gironde",
        "location_region": "Nouvelle-Aquitaine", "location_coordinates": {"lat": 44.84, "lon": -0.57},
        "conditions_fr": None, "age_min": None, "age_max": None, "registration": None,
        "accessibility_label_fr": None,
        "status": '{"id": 1, "label": {"fr": "Programmé", "en": "Scheduled"}}',
        "attendancemode": '{"id": 1, "label": {"fr": "Sur place"}}',
        "canonicalurl": "https://openagenda.com/exemple", "originagenda_title": "Opéra National de Bordeaux",
        "updatedat": "2026-10-01T00:00:00+00:00",
    }
    event.update(overrides)
    return event


class KeywordEmbedder(Embeddings):
    """Faux modèle d'embedding lisible : un axe par mot-clé, vecteurs normalisés comme ceux de Mistral.

    « jazz jazz » est très proche de la question « jazz », « théâtre » en est éloigné.
    Compte les vectorisations de questions (= appels au modèle).
    """

    KEYWORDS = ["jazz", "theatre", "exposition", "enfants"]

    def __init__(self):
        self.query_calls = 0
        self.queries = []   # textes vectorisés comme questions, dans l'ordre

    def vectorize(self, text: str) -> list[float]:
        words = normalize(text).split()
        counts = np.array([words.count(keyword) for keyword in self.KEYWORDS], dtype=float) + 0.01
        return list(counts / np.linalg.norm(counts))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        self.queries.append(text)
        return self.vectorize(text)


def session(day: date) -> dict:
    """Séance de 20h30 à 23h00 (heure de Paris)."""
    return {"begin": f"{day.isoformat()}T20:30:00+01:00", "end": f"{day.isoformat()}T23:00:00+01:00"}


def make_chunk(uid: str, text: str, sessions: list[date], city: str = "Bordeaux",
               department: str = "Gironde", chunk_index: int = 0, chunk_count: int = 1) -> Document:
    """Chunk avec les métadonnées utilisées par la recherche et par les sources de la réponse."""
    session_list = [session(day) for day in sessions]
    return Document(id=f"{uid}#{chunk_index}", page_content=text, metadata={
        "uid": uid, "title": uid, "city": city, "department": department, "location_name": f"Salle {uid}",
        "start": session_list[0]["begin"], "end": session_list[-1]["end"],
        "url": f"https://openagenda.com/{uid}", "sessions": session_list,
        "chunk_index": chunk_index, "chunk_count": chunk_count,
    })
