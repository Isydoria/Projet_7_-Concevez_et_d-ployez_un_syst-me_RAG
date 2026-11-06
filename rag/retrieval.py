"""Recherche filtrée dans l'index : critères appliqués à tous les chunks, un résultat par événement.

Le FAISS de LangChain n'applique un filtre qu'aux `fetch_k` chunks les plus proches (20 par défaut) :
un événement qui respecte les critères sans être parmi eux serait perdu. On filtre donc tous les
chunks (index exact Flat : le tri complet reste rapide).
"""

from dataclasses import dataclass
from datetime import date, datetime

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag import config
from rag.criteria import Criteria, Period, build_city_index, normalize


@dataclass(frozen=True)
class RetrievedEvent:
    """Un événement trouvé : son meilleur chunk et la similarité (cosinus) de ce chunk avec la question."""
    document: Document
    score: float


@dataclass(frozen=True)
class EventFilterData:
    """Ce qu'il faut savoir d'un événement pour le filtrer, préparé une seule fois au chargement."""
    city_key: str                             # ville normalisée (« POITIERS » -> « poitiers »)
    department: str
    sessions: tuple[tuple[date, date], ...]   # dates de début et de fin de chaque séance


def overlaps(period: Period, begin: date, end: date) -> bool:
    """Vrai si les dates [begin, end] chevauchent la période (une borne None = pas de limite)."""
    return (period.end is None or begin <= period.end) and (period.start is None or end >= period.start)


def session_dates(session: dict) -> tuple[date, date]:
    """Dates de début et de fin d'une séance (les heures sont déjà en heure de Paris)."""
    return datetime.fromisoformat(session["begin"]).date(), datetime.fromisoformat(session["end"]).date()


def sessions_in_period(sessions: list[dict], period: Period) -> list[dict]:
    """Séances qui chevauchent la période (par exemple, celles du week-end demandé)."""
    return [session for session in sessions if overlaps(period, *session_dates(session))]


class EventRetriever:
    """Recherche filtrée : critères appliqués à tous les chunks, un résultat par événement."""

    def __init__(self, store: FAISS):
        self.store = store
        documents = [store.docstore.search(doc_id) for doc_id in store.index_to_docstore_id.values()]
        # Pour criteria.py : chaque ville avec son écriture la plus fréquente
        self.city_index = build_city_index(document.metadata["city"] for document in documents)
        # Une entrée par événement : tous ses chunks portent les mêmes métadonnées d'événement
        self.events = {
            document.metadata["uid"]: EventFilterData(
                city_key=normalize(document.metadata["city"]),
                department=document.metadata["department"],
                sessions=tuple(session_dates(session) for session in document.metadata["sessions"]),
            )
            for document in documents if document.metadata["chunk_index"] == 0
        }

    def allowed_uids(self, criteria: Criteria) -> set[str]:
        """Événements qui respectent les critères : ville, département, au moins une séance dans la période.

        Calculé une fois par question sur les événements : le filtre de LangChain, appelé pour
        chaque chunk, n'a plus qu'à tester l'appartenance d'un uid à cet ensemble.
        """
        city_key = normalize(criteria.city)
        return {uid for uid, event in self.events.items()
                if (not city_key or event.city_key == city_key)
                and (not criteria.department or event.department == criteria.department)
                and any(overlaps(criteria.period, begin, end) for begin, end in event.sessions)}

    def search(self, question: str, criteria: Criteria, k: int = config.TOP_K_EVENTS) -> list[RetrievedEvent]:
        """Les k événements les plus proches de la question parmi ceux qui respectent les critères."""
        allowed = self.allowed_uids(criteria)
        if not allowed:
            return []  # rien à chercher : pas d'appel au modèle d'embedding
        total = self.store.index.ntotal
        matches = self.store.similarity_search_with_score(
            question, k=total, fetch_k=total, filter=lambda metadata: metadata["uid"] in allowed)
        best: dict[str, RetrievedEvent] = {}
        for document, score in matches:  # du plus au moins similaire : le premier chunk est le meilleur
            uid = document.metadata["uid"]
            if uid not in best:
                best[uid] = RetrievedEvent(document, float(score))
                if len(best) == k:
                    break
        return list(best.values())
