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
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# Photo de référence (versionnée) : base de l'évaluation et du jeu de test annoté
REFERENCE_RAW_FILE = RAW_DATA_DIR / "events_nouvelle-aquitaine_2026-10-29.json.gz"
EVAL_DIR = DATA_DIR / "eval"
TEST_SET_FILE = EVAL_DIR / "test_set.json"   # jeu de test annoté (20 questions)

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

# --- Nettoyage (règles justifiées dans notebooks/01_exploration.ipynb, section 8) --------------
TIMEZONE = "Europe/Paris"
MAX_HORIZON_DAYS = 730   # règle 2 : un début à plus de 2 ans de la date de référence est aberrant
CANCELLED_STATUS_ID = 6  # règle 3 : statut Open Agenda « Annulé »

# Règle 5 : agendas emploi et formation exclus. On compare le DÉBUT du nom, pour couvrir les
# éditions futures (« Semaine de l'industrie 2027 »…). Liste à relire à chaque nouvelle photo.
EXCLUDED_AGENDA_PREFIXES = [
    "Mes événements France Travail",
    "Semaine de l'industrie",
    "Cap Métiers",
    "Semaine des métiers du tourisme",
    "Adie NA",
]

# --- Découpage en chunks (choix justifié par la distribution des longueurs de texte) -------------
CHUNK_SIZE = 1500     # caractères de description au maximum par chunk
CHUNK_OVERLAP = 150   # caractères répétés entre deux chunks consécutifs

# --- Vectorisation et index Faiss ------------------------------------------------------------------
EMBEDDING_MODEL = "mistral-embed-2312"   # nom daté : reproductibilité
EMBED_BATCH_MAX_CHARS = 40_000   # ≈ 15 000 tokens par requête (mesuré : ≈ 2,7 caractères par token en français) ;
                                 # LangChain redécoupe automatiquement un lot qui dépasserait 16 000 tokens
EMBED_MIN_INTERVAL_S = 1.1       # offre gratuite Mistral : 1 requête par seconde au maximum

# --- Recherche (chaîne RAG) -------------------------------------------------------------------------
TOP_K_EVENTS = 5   # événements distincts transmis au LLM (le meilleur chunk de chacun)
MAX_SESSIONS_SHOWN = 5   # séances affichées par événement dans le prompt (un événement en compte 783)

# --- Génération (chaîne RAG) ------------------------------------------------------------------------
# Valeurs explicites : ChatMistralAI prend par défaut une température de 0,7 et une longueur illimitée.
# Ministral 8B, nom daté : reproductibilité. Le POC fonctionne avec la formule gratuite de Mistral, qui
# refuse Mistral Large 4 depuis le 13/11/2026 (« not available in your subscription tier ») et limite
# Small et Medium à 0 requête par minute ; Ministral 8B y est accepté, à 2 requêtes par minute.
# Comparaison des modèles : notebook 04, partie B ; Large 4 (réflexion « low ») y est le meilleur et reste
# la cible pour une mise en production avec une offre payante.
CHAT_MODEL = "ministral-8b-2512"
CHAT_TEMPERATURE = 0.2              # réponses stables et fidèles aux événements fournis
# Niveau de réflexion d'un modèle de raisonnement (Large 4 : low, medium ou high ; le plafond de tokens
# couvre alors la réflexion ET la réponse). None : Ministral 8B ne raisonne pas, le paramètre n'est pas envoyé.
CHAT_REASONING_EFFORT = None
CHAT_MAX_TOKENS = 4000              # plafond de sécurité ; réponse la plus longue observée : 291 tokens

# Règle 4 : départements de Nouvelle-Aquitaine (2 premiers chiffres du code postal -> nom)
NA_DEPARTMENTS = {
    "16": "Charente", "17": "Charente-Maritime", "19": "Corrèze", "23": "Creuse",
    "24": "Dordogne", "33": "Gironde", "40": "Landes", "47": "Lot-et-Garonne",
    "64": "Pyrénées-Atlantiques", "79": "Deux-Sèvres", "86": "Vienne", "87": "Haute-Vienne",
}
