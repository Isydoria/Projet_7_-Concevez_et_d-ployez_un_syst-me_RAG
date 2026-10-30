"""Découpage des événements nettoyés en chunks (documents LangChain) prêts à être vectorisés.

Un événement = un document ; seule une description trop longue est découpée (découpage récursif).
Chaque chunk commence par un en-tête de contexte (titre, dates, lieu…) et porte des métadonnées
(dates, ville, URL…) qui servent aux filtres et à la citation des sources.
"""

from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config

# Noms écrits « en dur » : les noms fournis par le système dépendent de sa langue
# (« Sunday » sur un Windows en anglais).
DAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
             "septembre", "octobre", "novembre", "décembre"]

# Métadonnées conservées pour chaque chunk (filtres, sources, vérification de l'indexation)
METADATA_FIELDS = ["uid", "title", "start", "end", "city", "department", "location_name", "url",
                   "status_label", "attendance_label", "age_min", "age_max", "keywords"]

SEPARATOR = "\n---\n"  # sépare l'en-tête du morceau de description


def format_date_fr(moment: datetime) -> str:
    """« samedi 31 octobre 2026 »."""
    return f"{DAYS_FR[moment.weekday()]} {moment.day} {MONTHS_FR[moment.month - 1]} {moment.year}"


def format_time_fr(moment: datetime) -> str:
    """« 20h30 »."""
    return f"{moment.hour}h{moment.minute:02d}"


def format_datetime_fr(iso_datetime: str) -> str:
    """« 2026-10-31T20:30:00+01:00 » -> « samedi 31 octobre 2026 à 20h30 »."""
    moment = datetime.fromisoformat(iso_datetime)
    return f"{format_date_fr(moment)} à {format_time_fr(moment)}"


def format_dates(start: str, end: str) -> str:
    """Période lisible, toujours avec l'année (le champ date_text d'Open Agenda l'omet parfois)."""
    if not start:
        return ""
    if not end or end == start:
        return f"le {format_datetime_fr(start)}"
    begin, finish = datetime.fromisoformat(start), datetime.fromisoformat(end)
    if begin.date() == finish.date():
        return f"le {format_date_fr(begin)}, de {format_time_fr(begin)} à {format_time_fr(finish)}"
    return f"du {format_datetime_fr(start)} au {format_datetime_fr(end)}"


def format_age(age_min: float | None, age_max: float | None) -> str:
    """« de 3 à 16 ans », « à partir de 6 ans », « jusqu'à 12 ans », ou "" si l'âge n'est pas renseigné."""
    if age_min is not None and age_max is not None:
        return f"de {int(age_min)} à {int(age_max)} ans"
    if age_min is not None:
        return f"à partir de {int(age_min)} ans"
    if age_max is not None:
        return f"jusqu'à {int(age_max)} ans"
    return ""


def build_header(event: dict) -> str:
    """En-tête de contexte répété en tête de chaque chunk (une ligne par information disponible)."""
    location = ", ".join(part for part in [event["location_name"], event["city"]] if part)
    if event["department"]:
        location += f" ({event['department']})"
    lines = [
        ("Titre", event["title"]),
        ("Dates", format_dates(event["start"], event["end"])),
        ("Lieu", location),
        ("Mots-clés", ", ".join(str(keyword) for keyword in event["keywords"])),
        ("Tarifs", event["conditions"]),
        ("Âge", format_age(event["age_min"], event["age_max"])),
        # Statut et mode de participation : seulement s'ils sont inhabituels
        ("Statut", "" if event["status_label"] in ("", "Programmé") else event["status_label"]),
        ("Participation", "" if event["attendance_label"] in ("", "Sur place") else event["attendance_label"]),
    ]
    return "\n".join(f"{label} : {value}" for label, value in lines if value)


def build_body(event: dict) -> str:
    """Description courte + description longue, sans répéter la courte quand la longue commence par elle."""
    short, long_text = event["description"], event["long_description"]
    if short and long_text.startswith(short.rstrip(".… ")):
        return long_text
    return "\n".join(part for part in [short, long_text] if part)


def make_splitter() -> RecursiveCharacterTextSplitter:
    """Découpeur récursif : coupe de préférence entre paragraphes, puis lignes, puis phrases, puis mots."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",  # le point reste à la fin de la phrase
    )


def chunk_event(event: dict, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    """Transforme un événement en un ou plusieurs chunks : en-tête + morceau de description."""
    header = build_header(event)
    body = build_body(event)
    pieces = splitter.split_text(body) if body else [""]
    metadata = {field: event[field] for field in METADATA_FIELDS}
    return [
        Document(
            id=f"{event['uid']}#{index}",  # identifiant stable : prépare une mise à jour incrémentale
            page_content=f"{header}{SEPARATOR}{piece}" if piece else header,
            metadata={**metadata, "chunk_index": index, "chunk_count": len(pieces)},
        )
        for index, piece in enumerate(pieces)
    ]


def chunk_events(events: list[dict]) -> tuple[list[Document], dict]:
    """Découpe tous les événements ; renvoie (chunks, rapport chiffré du découpage)."""
    splitter = make_splitter()
    documents = [document for event in events for document in chunk_event(event, splitter)]
    report = {
        "events": len(events),
        "chunks": len(documents),
        "split_events": sum(1 for document in documents if document.metadata["chunk_index"] == 1),
        "characters": sum(len(document.page_content) for document in documents),
    }
    return documents, report
