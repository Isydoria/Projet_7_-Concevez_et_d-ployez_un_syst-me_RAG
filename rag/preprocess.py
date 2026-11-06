"""Nettoyage et structuration des événements Open Agenda (règles justifiées dans notebooks/01_exploration.ipynb, section 8).

Les règles sont rejouées à chaque reconstruction de l'index : elles sont donc génériques.
Déroulé : 1) transformations de chaque événement (décodage, textes, lieu, dates en heure de Paris) ;
2) filtres (annulés, agendas exclus, dates aberrantes, sans titre) ; 3) dédoublonnage, EN DERNIER,
pour qu'entre deux copies d'un même événement on garde une copie valide.
"""

import json
import logging
import re
from datetime import date, datetime, time, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from rag import config
from rag.fetch import load_raw_events, save_raw_events

logger = logging.getLogger(__name__)

PARIS = ZoneInfo(config.TIMEZONE)
ONLY_PUNCTUATION = re.compile(r"^[\W_]*$")  # aucun caractère de mot (lettre, chiffre)


class _TextExtractor(HTMLParser):
    """Récupère le texte d'un fragment HTML ; les balises de bloc deviennent des retours à la ligne."""

    BLOCK_TAGS = {"p", "br", "div", "li", "ul", "ol", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)  # décode les entités : &nbsp;, &amp;…
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(html: str | None) -> str:
    """Convertit un fragment HTML en texte brut lisible (règle 8)."""
    if not html:
        return ""
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    text = "".join(extractor.parts).replace("\xa0", " ")  # espace insécable -> espace
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def clean_text(text: str | None) -> str:
    """Réduit les espaces ; un texte fait uniquement de ponctuation devient vide (règle 11)."""
    if not text:
        return ""
    text = " ".join(str(text).split())
    return "" if ONLY_PUNCTUATION.match(text) else text


def parse_json_field(value):
    """Décode un champ stocké comme texte JSON (règle 9) ; listes et dictionnaires sont renvoyés tels quels."""
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Champ JSON illisible ignoré : %.60s", value)
        return None


def extract_label(value) -> tuple[int | None, str]:
    """Extrait (identifiant, libellé français) d'un champ {"id": …, "label": {"fr": …}} : statut, mode."""
    data = parse_json_field(value)
    if not isinstance(data, dict):
        return None, ""
    label = data.get("label")
    if isinstance(label, dict):
        label = label.get("fr", "")
    return data.get("id"), label or ""


def department_from_postalcode(postalcode: str | None, declared: str | None) -> str:
    """Déduit le département des 2 premiers chiffres du code postal (règle 4) ; sinon garde le nom saisi."""
    prefix = (postalcode or "").strip()[:2]
    return config.NA_DEPARTMENTS.get(prefix, declared or "")


def to_paris_time(iso_datetime: str | None) -> str:
    """Convertit une date ISO en heure de Paris (règle 10) : « …T10:00:00+00:00 » -> « …T11:00:00+01:00 »."""
    if not iso_datetime:
        return ""
    return datetime.fromisoformat(iso_datetime).astimezone(PARIS).isoformat()


def clean_sessions(timings: str | None, start: str, end: str) -> list[dict]:
    """Séances en heure de Paris (règle 10) ; sans séance, la plage début-fin de l'événement.

    Le filtre de dates s'appuie sur les séances : tout événement doit en avoir au moins une.
    """
    sessions = [{"begin": to_paris_time(session.get("begin")), "end": to_paris_time(session.get("end"))}
                for session in parse_json_field(timings) or []]
    return sessions or [{"begin": start, "end": end}]


def is_excluded_agenda(agenda_title: str | None) -> bool:
    """Vrai si l'agenda d'origine fait partie des agendas emploi et formation exclus (règle 5)."""
    title = (agenda_title or "").replace("’", "'").strip()
    return any(title.startswith(prefix) for prefix in config.EXCLUDED_AGENDA_PREFIXES)


def normalize_key(text: str | None) -> str:
    """Forme comparable pour le dédoublonnage : minuscules, espaces réduits (règle 1)."""
    return " ".join((text or "").lower().split())


def clean_event(event: dict) -> dict:
    """Transforme un événement brut en événement structuré, avec des noms de champs clairs."""
    status_id, status_label = extract_label(event.get("status"))
    _, attendance_label = extract_label(event.get("attendancemode"))
    coordinates = event.get("location_coordinates") or {}
    start, end = to_paris_time(event.get("firstdate_begin")), to_paris_time(event.get("lastdate_end"))
    return {
        "uid": event["uid"],
        "title": clean_text(event.get("title_fr")),
        "description": clean_text(event.get("description_fr")),
        "long_description": html_to_text(event.get("longdescription_fr")),
        "keywords": event.get("keywords_fr") or [],
        "start": start,
        "end": end,
        "date_text": clean_text(event.get("daterange_fr")),
        "sessions": clean_sessions(event.get("timings"), start, end),
        "location_name": clean_text(event.get("location_name")),
        "address": clean_text(event.get("location_address")),
        "city": clean_text(event.get("location_city")),
        "postal_code": clean_text(event.get("location_postalcode")),
        "department": department_from_postalcode(event.get("location_postalcode"),
                                                 event.get("location_department")),
        "region": event.get("location_region") or "",
        "latitude": coordinates.get("lat"),
        "longitude": coordinates.get("lon"),
        "conditions": clean_text(event.get("conditions_fr")),
        "age_min": event.get("age_min"),
        "age_max": event.get("age_max"),
        "registration": parse_json_field(event.get("registration")) or [],
        "accessibility": event.get("accessibility_label_fr") or [],
        "status_id": status_id,
        "status_label": status_label,
        "attendance_label": attendance_label,
        "url": event.get("canonicalurl") or "",
        "source_agenda": event.get("originagenda_title") or "",
        "updated_at": event.get("updatedat") or "",
    }


def preprocess_events(events: list[dict], reference_date: date) -> tuple[list[dict], dict]:
    """Applique toutes les règles ; renvoie (événements nettoyés, rapport : nombre traité par règle)."""
    cleaned = [clean_event(event) for event in events]
    report = {
        "input": len(events),
        "departments_fixed": sum(1 for raw, clean in zip(events, cleaned)
                                 if clean["department"] != (raw.get("location_department") or "")),
    }
    # pandas sert à calculer les filtres ; les événements eux-mêmes restent des dictionnaires Python
    df = pd.DataFrame(cleaned)
    horizon = datetime.combine(reference_date, time(), tzinfo=PARIS) + timedelta(days=config.MAX_HORIZON_DAYS)
    filters = {
        "cancelled": df["status_id"] == config.CANCELLED_STATUS_ID,                       # règle 3
        "excluded_agendas": df["source_agenda"].map(is_excluded_agenda),                  # règle 5
        "beyond_horizon": df["start"].map(                                                # règle 2
            lambda start: not start or datetime.fromisoformat(start) > horizon),
        "without_title": df["title"] == "",                                               # règle 13
    }
    for rule, mask in filters.items():
        remaining = mask.loc[df.index]       # le filtre, restreint aux lignes encore présentes
        report[rule] = int(remaining.sum())
        df = df[~remaining]

    # Règle 1 : dédoublonnage, en gardant la version mise à jour le plus récemment
    df = df.assign(title_key=df["title"].map(normalize_key),
                   location_key=df["location_name"].map(normalize_key))
    df = df.sort_values(["updated_at", "uid"], ascending=[False, True], kind="stable")
    duplicated = df.duplicated(subset=["title_key", "start", "location_key"], keep="first")
    report["duplicates"] = int(duplicated.sum())
    df = df[~duplicated]

    # Ordre stable du résultat (reproductibilité) : par date de début, puis par identifiant
    df = df.sort_values(["start", "uid"], kind="stable")
    report["output"] = len(df)
    return [cleaned[index] for index in df.index], report


def preprocess_file(raw_path: Path, output_dir: Path = config.PROCESSED_DATA_DIR) -> Path:
    """Nettoie un fichier de data/raw/ et enregistre le résultat (même nom) dans data/processed/."""
    raw_metadata, raw_events = load_raw_events(raw_path)
    events, report = preprocess_events(raw_events, date.fromisoformat(raw_metadata["reference_date"]))
    for rule, count in report.items():
        logger.info("%-18s : %d", rule, count)

    metadata = {
        "source_file": raw_path.name,
        "reference_date": raw_metadata["reference_date"],
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report": report,
        "count": len(events),
    }
    output_path = output_dir / raw_path.name
    save_raw_events(events, metadata, output_path)  # même format .json.gz que les données brutes
    logger.info("Données nettoyées enregistrées : %s", output_path)
    return output_path
