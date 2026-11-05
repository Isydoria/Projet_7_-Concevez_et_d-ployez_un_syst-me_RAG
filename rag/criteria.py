"""Repérage, par des règles, des critères d'une question : lieu (ville, département) et période.

Les règles sont déterministes et testables. Une question qui ne contient aucun critère reconnu
est traitée par la seule recherche par le sens ; la période vaut alors « à venir ».
"""

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from rag import config
from rag.chunking import DAYS_FR, MONTHS_FR, format_date_fr


@dataclass(frozen=True)
class Period:
    """Période demandée : dates incluses ; None = pas de borne."""
    label: str           # « ce week-end », « à venir »… (repris dans le prompt et les messages)
    start: date | None
    end: date | None


@dataclass(frozen=True)
class Criteria:
    """Critères repérés dans une question (None = pas de filtre)."""
    period: Period       # toujours renseignée : « à venir » par défaut
    city: str | None = None
    department: str | None = None


def normalize(text: str | None) -> str:
    """Forme comparable : minuscules, sans accents, tirets et ponctuation remplacés par des espaces.

    « Châtellerault » et « CHATELLERAULT » -> « chatellerault » ; « week-end » -> « week end ».
    """
    text = (text or "").lower().replace("œ", "oe").replace("æ", "ae")
    text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


# --- Lieu ---------------------------------------------------------------------------------------

# Prépositions de lieu, en texte normalisé (« à » -> « a »). Une ville n'est repérée qu'après l'une
# d'elles : « dégustation de cognac » ou « vins de Bordeaux » ne désignent pas un lieu.
# « dans » est réservé aux départements : « dans le noir », « dans la nature » pourraient
# tomber sur une commune.
CITY_PREPOSITIONS = ["a", "au", "aux", "sur", "vers", "pres de", "proche de", "autour de",
                     "a cote de", "du cote de", "aux alentours de", "aux environs de"]
DEPARTMENT_PREPOSITIONS = ["en", "dans le", "dans la", "dans les", "dans l"]
CONTRACTED_ARTICLES = {"au": "le", "aux": "les"}  # « au Bouscat » = « à Le Bouscat »
DEPARTMENTS = {normalize(name): name for name in config.NA_DEPARTMENTS.values()}


def build_city_index(cities: Iterable[str | None]) -> dict[str, str]:
    """Forme normalisée -> écriture de référence (« bordeaux » -> « Bordeaux »).

    `cities` contient une ville par événement ou par chunk, avec répétitions : on garde l'écriture
    la plus fréquente, en préférant une écriture qui n'est pas tout en majuscules.
    """
    spellings = defaultdict(Counter)
    for city in cities:
        if key := normalize(city):
            spellings[key][city] += 1
    return {key: max(counter, key=lambda spelling: (not spelling.isupper(), counter[spelling]))
            for key, counter in spellings.items()}


def find_place(text: str, prepositions: list[str], names: dict[str, str]) -> str | None:
    """Premier nom de `names` cité juste après une des `prepositions` (`text` déjà normalisé).

    Le nom le plus long l'emporte : « à Saint-Jean-de-Luz » donne Saint-Jean-de-Luz, pas Saint-Jean.
    """
    words = text.split()
    max_words = max((key.count(" ") + 1 for key in names), default=0)
    longest_first = sorted(prepositions, key=lambda preposition: -len(preposition.split()))
    for position in range(len(words)):
        for preposition in longest_first:
            preposition_words = preposition.split()
            if words[position:position + len(preposition_words)] != preposition_words:
                continue
            following = words[position + len(preposition_words):]
            article = CONTRACTED_ARTICLES.get(preposition)
            for count in range(min(max_words, len(following)), 0, -1):
                candidate = " ".join(following[:count])
                for key in ([f"{article} {candidate}"] if article else []) + [candidate]:
                    if key in names:
                        return names[key]
    return None


# --- Période ------------------------------------------------------------------------------------

WEEKDAYS = [normalize(day) for day in DAYS_FR]       # « lundi »… « dimanche »
MONTHS = [normalize(month) for month in MONTHS_FR]   # « fevrier », « aout », « decembre »…
WEEKDAY_RE = "|".join(WEEKDAYS)
MONTH_RE = "|".join(MONTHS)
PAST_RE = "(?:dernier|derniere|passe|passee)"        # « la semaine dernière », « le mois passé »
NEXT_RE = "(?:prochain|prochaine)"
MONTH_MARKERS_RE = "(?:en|debut|debut d|fin|fin d|mi|courant|mois de|mois d)"  # « en mars », « fin novembre »

# Mots qui situent toute la question dans le passé. « se passe » (présent) n'en fait pas partie.
PAST_MARKERS = re.compile(
    r"\b(?:passes|passees|s est passee?s?|se sont passee?s?|il y a eu|il y avait|a eu lieu|"
    r"ont eu lieu|avait lieu|avaient lieu|l an dernier|l annee derniere|l annee passee)\b"
)

# (expression normalisée, décalage en jours, libellé) : « après-demain » avant « demain »
RELATIVE_DAYS = [
    ("apres demain", 2, "après-demain"),
    ("avant hier", -2, "avant-hier"),
    ("aujourd hui", 0, "aujourd'hui"),
    ("ce soir", 0, "ce soir"),
    ("ce matin", 0, "ce matin"),
    ("cet apres midi", 0, "cet après-midi"),
    ("demain", 1, "demain"),
    ("hier", -1, "hier"),
]


def month_period(label: str, year: int, month: int) -> Period:
    """Le mois entier, du 1er au dernier jour."""
    first = date(year, month, 1)
    next_first = date(year + month // 12, month % 12 + 1, 1)
    return Period(label, first, next_first - timedelta(days=1))


def resolve_day(day: int, month: int, year: int | None, today: date, past: bool) -> date | None:
    """Date complète : sans année, la prochaine occurrence (la dernière, au passé) ; None si impossible."""
    try:
        if year:
            return date(year, month, day)
        candidate = date(today.year, month, day)
        if past and candidate > today:
            return date(today.year - 1, month, day)
        if not past and candidate < today:
            return date(today.year + 1, month, day)
        return candidate
    except ValueError:  # « le 31 février »
        return None


def explicit_date(question: str, text: str, today: date, past: bool) -> Period | None:
    """« le 14 novembre », « le 1er octobre 2027 », « le 14/11 », « le 3/10/2026 »."""
    match = re.search(rf"\b(\d{{1,2}})(?:er)? ({MONTH_RE})(?: (\d{{4}}))?( {PAST_RE})?\b", text)
    if match:
        day = resolve_day(int(match[1]), MONTHS.index(match[2]) + 1,
                          int(match[3]) if match[3] else None, today, past or bool(match[4]))
    else:
        # Dates en chiffres : cherchées dans la question d'origine, car la normalisation supprime
        # les « / ». « 3/6 ans » est une tranche d'âge, pas une date.
        match = re.search(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2}))?(?![\d/])(?!\s*ans\b)",
                          question)
        if not match:
            return None
        year = int(match[3]) if match[3] else None
        if year and year < 100:
            year += 2000
        day = resolve_day(int(match[1]), int(match[2]), year, today, past)
    return Period(f"le {format_date_fr(day)}", day, day) if day else None


def relative_day(text: str, today: date) -> Period | None:
    """« aujourd'hui », « ce soir », « demain », « après-demain », « hier », « avant-hier »."""
    for expression, offset, label in RELATIVE_DAYS:
        if re.search(rf"\b{expression}\b", text):
            day = today + timedelta(days=offset)
            return Period(label, day, day)
    return None


def weekend(text: str, today: date) -> Period | None:
    """« ce week-end », « le week-end prochain », « le week-end dernier » (singulier seulement)."""
    saturday = today + timedelta(days=5 - today.weekday())  # samedi de la semaine en cours
    if re.search(rf"\bweek ?end {PAST_RE}\b", text):
        return Period("le week-end dernier", saturday - timedelta(days=7), saturday - timedelta(days=6))
    if re.search(rf"\bweek ?end {NEXT_RE}\b|\b{NEXT_RE} week ?end\b", text):
        if today.weekday() >= 5:  # samedi ou dimanche : le week-end suivant
            saturday += timedelta(days=7)
        return Period("le week-end prochain", saturday, saturday + timedelta(days=1))
    if re.search(r"\bweek ?end\b", text):
        return Period("ce week-end", max(saturday, today), saturday + timedelta(days=1))
    return None


def week(text: str, today: date) -> Period | None:
    """« cette semaine », « la semaine prochaine », « la semaine dernière »."""
    monday = today - timedelta(days=today.weekday())
    if re.search(rf"\bsemaine {PAST_RE}\b", text):
        return Period("la semaine dernière", monday - timedelta(days=7), monday - timedelta(days=1))
    if re.search(rf"\bsemaine {NEXT_RE}\b|\b{NEXT_RE} semaine\b", text):
        return Period("la semaine prochaine", monday + timedelta(days=7), monday + timedelta(days=13))
    if re.search(r"\bcette semaine\b", text):
        return Period("cette semaine", today, monday + timedelta(days=6))
    return None


def relative_month(text: str, today: date) -> Period | None:
    """« ce mois-ci », « le mois prochain », « le mois dernier »."""
    first = today.replace(day=1)
    if re.search(rf"\bmois {PAST_RE}\b", text):
        previous = first - timedelta(days=1)
        return month_period("le mois dernier", previous.year, previous.month)
    if re.search(rf"\bmois {NEXT_RE}\b|\b{NEXT_RE} mois\b", text):
        following = first + timedelta(days=32)
        return month_period("le mois prochain", following.year, following.month)
    if re.search(r"\bce mois(?: ci)?\b(?! de)", text):  # « ce mois de décembre » : nom de mois
        return Period("ce mois-ci", today, month_period("", today.year, today.month).end)
    return None


def month_name(text: str, today: date, past: bool) -> Period | None:
    """« en décembre », « fin novembre », « au mois d'août », « mars 2027 », « en octobre dernier ».

    Le nom du mois doit être annoncé (« en », « fin »…) ou suivi d'une année : sinon, « la planète
    Mars » deviendrait le mois de mars.
    """
    match = (re.search(rf"\b{MONTH_MARKERS_RE} (?P<month>{MONTH_RE})(?: (?P<year>\d{{4}}))?"
                       rf"(?P<past> {PAST_RE})?\b", text)
             or re.search(rf"\b(?P<month>{MONTH_RE}) (?P<year>\d{{4}})\b", text))
    if not match:
        return None
    month = MONTHS.index(match["month"]) + 1
    if match["year"]:
        year = int(match["year"])
    elif past or match.groupdict().get("past"):
        year = today.year if month <= today.month else today.year - 1
    else:
        year = today.year if month >= today.month else today.year + 1
    return month_period(f"en {MONTHS_FR[month - 1]} {year}", year, month)


def weekdays(text: str, today: date, past: bool) -> Period | None:
    """« samedi », « dimanche prochain », « samedi dernier » ; « samedi ou dimanche » -> les deux jours.

    Le pluriel (« les samedis ») désigne une habitude : il ne déclenche pas de filtre.
    """
    days, labels = [], []
    for match in re.finditer(rf"\b({WEEKDAY_RE})(?: ({NEXT_RE}|{PAST_RE}))?\b", text):
        weekday, modifier = WEEKDAYS.index(match[1]), match[2] or ""
        if past or modifier.startswith(("dernier", "passe")):
            days.append(today - timedelta(days=(today.weekday() - weekday) % 7 or 7))
        else:
            offset = (weekday - today.weekday()) % 7  # aujourd'hui compris
            if offset == 0 and modifier.startswith("prochain"):
                offset = 7
            days.append(today + timedelta(days=offset))
        labels.append(match[0])
    return Period(", ".join(labels), min(days), max(days)) if days else None


def find_period(question: str, today: date) -> Period:
    """Période demandée, de la règle la plus précise à la plus vague ; « à venir » par défaut."""
    text = normalize(question)
    past = bool(PAST_MARKERS.search(text))
    return (
        explicit_date(question, text, today, past)
        or relative_day(text, today)
        or weekend(text, today)
        or week(text, today)
        or relative_month(text, today)
        or month_name(text, today, past)
        or weekdays(text, today, past)
        or (Period(f"les {config.HISTORY_DAYS} derniers jours",
                   today - timedelta(days=config.HISTORY_DAYS), today - timedelta(days=1)) if past else None)
        or Period("à venir", today, None)
    )


def extract_criteria(question: str, today: date, city_index: dict[str, str]) -> Criteria:
    """Critères d'une question : période (toujours), ville et département (s'ils sont cités)."""
    text = normalize(question)
    return Criteria(
        period=find_period(question, today),
        city=find_place(text, CITY_PREPOSITIONS, city_index),
        department=find_place(text, DEPARTMENT_PREPOSITIONS, DEPARTMENTS),
    )
