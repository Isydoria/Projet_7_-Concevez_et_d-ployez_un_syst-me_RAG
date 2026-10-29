"""Paramètres centralisés du projet (chemins, source de données…).

Les valeurs fixes du projet sont regroupées ici : pour changer un paramètre,
on le modifie à un seul endroit.
"""

from pathlib import Path

# --- Chemins ----------------------------------------------------------------------
# Racine du projet = dossier parent de rag/ (fonctionne sous Windows, macOS et Linux)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# --- Source des données : Open Agenda (OpenDataSoft, API Explore v2.1) --------------
OPENAGENDA_DATASET_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda"
)
DEFAULT_REGION = "Nouvelle-Aquitaine"
HISTORY_DAYS = 365  # 1 an d'historique, en plus des événements à venir

# Champs récupérés (26 sur 56), justifiés dans le rapport technique.
# Les champs de contact des contributeurs (données personnelles) sont volontairement exclus.
OPENAGENDA_FIELDS = [
    # Identification et source
    "uid", "title_fr", "canonicalurl", "originagenda_title", "updatedat",
    # Descriptions (texte à vectoriser)
    "description_fr", "longdescription_fr", "keywords_fr",
    # Dates
    "firstdate_begin", "lastdate_end", "timings", "daterange_fr",
    # Lieux
    "location_name", "location_address", "location_city", "location_postalcode",
    "location_department", "location_region", "location_coordinates",
    # Informations pratiques et statut
    "conditions_fr", "age_min", "age_max", "registration", "accessibility_label_fr",
    "status", "attendancemode",
]
