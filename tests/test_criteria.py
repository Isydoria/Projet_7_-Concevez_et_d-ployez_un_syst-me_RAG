"""Tests unitaires du repérage des critères d'une question (ville, département, période).

La date du jour est fixée au jeudi 5 novembre 2026 : les résultats ne dépendent pas du jour
où les tests sont lancés.
"""

from datetime import date

import pytest

from rag.criteria import Criteria, Period, build_city_index, extract_criteria, find_period, normalize, search_text

TODAY = date(2026, 11, 5)  # jeudi

# Petit index de villes, avec les pièges observés dans les données : écritures multiples,
# noms qui sont aussi des mots courants (Cognac, Sens), nom inclus dans un autre (Saint-Jean).
CITY_INDEX = build_city_index([
    "Bordeaux", "BORDEAUX", "Bordeaux", "Poitiers", "Pau", "Limoges", "La Rochelle", "Le Bouscat",
    "Châtellerault", "CHATELLERAULT", "Saint-Jean", "Saint-Jean-de-Luz", "Cognac", "Sens",
])


def criteria(question: str, today: date = TODAY) -> Criteria:
    return extract_criteria(question, today, CITY_INDEX)


# --- Normalisation ------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Châtellerault", "chatellerault"),
    ("CHATELLERAULT", "chatellerault"),
    ("Saint-Médard-en-Jalles", "saint medard en jalles"),
    ("Aujourd'hui, à 20h !", "aujourd hui a 20h"),
    ("Le cœur de la ville", "le coeur de la ville"),
    (None, ""),
])
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_city_index_groups_spellings_and_prefers_the_usual_one():
    index = build_city_index(["BORDEAUX", "Bordeaux", "Bordeaux", "NIORT", "NIORT", "Niort", None, ""])
    # Bordeaux : l'écriture la plus fréquente ; Niort : préférée à « NIORT », même moins fréquente
    assert index == {"bordeaux": "Bordeaux", "niort": "Niort"}


# --- Ville ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("Que faire à Bordeaux ce week-end ?", "Bordeaux"),
    ("Du théâtre sur Poitiers", "Poitiers"),
    ("Un concert près de Pau", "Pau"),
    ("Des expositions autour de Limoges", "Limoges"),
    ("Que voir à La Rochelle ?", "La Rochelle"),
    ("Un spectacle au Bouscat", "Le Bouscat"),               # « au » = « à le »
    ("Un concert à CHATELLERAULT", "Châtellerault"),
    ("Une fête à Saint-Jean-de-Luz", "Saint-Jean-de-Luz"),   # le nom le plus long l'emporte
    ("Qu'y a-t-il à faire à Pau ?", "Pau"),
])
def test_city_is_found_after_a_place_preposition(question, expected):
    assert criteria(question).city == expected


@pytest.mark.parametrize("question", [
    "Une dégustation de cognac",               # Cognac : ville et boisson
    "Une soirée autour des vins de Bordeaux",  # Bordeaux : ville et vin
    "Un spectacle qui a du sens",              # Sens : ville et mot courant
    "Un concert à Paris",                      # hors de la zone : ville inconnue
    "jazz Bordeaux samedi",                    # limite assumée : pas de préposition de lieu
])
def test_no_city_without_a_place_preposition_or_for_an_unknown_city(question):
    assert criteria(question).city is None


# --- Département --------------------------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("Des concerts en Gironde", "Gironde"),
    ("Que faire dans les Landes ?", "Landes"),
    ("Un festival en Charente-Maritime", "Charente-Maritime"),  # et pas « Charente »
    ("Un festival en Charente", "Charente"),
    ("Des balades dans la Haute-Vienne", "Haute-Vienne"),
    ("Des expositions dans les Deux-Sèvres", "Deux-Sèvres"),
    ("Un marché dans le Lot-et-Garonne", "Lot-et-Garonne"),
    ("Des concerts Gironde", None),                              # pas de préposition
])
def test_department(question, expected):
    assert criteria(question).department == expected


# --- Période (aujourd'hui : jeudi 5 novembre 2026) --------------------------------------------

@pytest.mark.parametrize("question, start, end", [
    ("Des concerts de jazz ?", date(2026, 11, 5), None),             # rien : à venir
    ("Que faire aujourd'hui ?", date(2026, 11, 5), date(2026, 11, 5)),
    ("Un concert ce soir", date(2026, 11, 5), date(2026, 11, 5)),
    ("Un spectacle demain", date(2026, 11, 6), date(2026, 11, 6)),
    ("Un spectacle après-demain", date(2026, 11, 7), date(2026, 11, 7)),
    ("Qu'y avait-il hier ?", date(2026, 11, 4), date(2026, 11, 4)),
    ("Et avant-hier ?", date(2026, 11, 3), date(2026, 11, 3)),
    ("Que faire ce week-end ?", date(2026, 11, 7), date(2026, 11, 8)),
    ("Un concert le weekend", date(2026, 11, 7), date(2026, 11, 8)),
    ("Le week-end prochain", date(2026, 11, 7), date(2026, 11, 8)),
    ("Le week-end dernier", date(2026, 10, 31), date(2026, 11, 1)),
    ("Cette semaine", date(2026, 11, 5), date(2026, 11, 8)),
    ("La semaine prochaine", date(2026, 11, 9), date(2026, 11, 15)),
    ("La semaine dernière", date(2026, 10, 26), date(2026, 11, 1)),
    ("Ce mois-ci", date(2026, 11, 5), date(2026, 11, 30)),
    ("Le mois prochain", date(2026, 12, 1), date(2026, 12, 31)),
    ("Le mois dernier", date(2026, 10, 1), date(2026, 10, 31)),
    ("En décembre", date(2026, 12, 1), date(2026, 12, 31)),
    ("En octobre", date(2027, 10, 1), date(2027, 10, 31)),           # à venir : le prochain octobre
    ("En octobre dernier", date(2026, 10, 1), date(2026, 10, 31)),
    ("Les concerts passés en octobre", date(2026, 10, 1), date(2026, 10, 31)),
    ("Fin novembre", date(2026, 11, 1), date(2026, 11, 30)),
    ("Au mois d'août", date(2027, 8, 1), date(2027, 8, 31)),
    ("Mars 2027", date(2027, 3, 1), date(2027, 3, 31)),
    ("Samedi", date(2026, 11, 7), date(2026, 11, 7)),
    ("Dimanche prochain", date(2026, 11, 8), date(2026, 11, 8)),
    ("Jeudi", date(2026, 11, 5), date(2026, 11, 5)),                 # aujourd'hui compris
    ("Samedi dernier", date(2026, 10, 31), date(2026, 10, 31)),
    ("Samedi ou dimanche", date(2026, 11, 7), date(2026, 11, 8)),
    ("Le 14 novembre", date(2026, 11, 14), date(2026, 11, 14)),
    ("Le 1er octobre", date(2027, 10, 1), date(2027, 10, 1)),        # déjà passé cette année
    ("Le 14/11", date(2026, 11, 14), date(2026, 11, 14)),
    ("Le 3/10/2026", date(2026, 10, 3), date(2026, 10, 3)),
    ("Les concerts passés", date(2025, 11, 5), date(2026, 11, 4)),   # les 365 derniers jours
    ("Qu'est-ce qui s'est passé à Pau ?", date(2025, 11, 5), date(2026, 11, 4)),
])
def test_period(question, start, end):
    period = find_period(question, TODAY)
    assert (period.start, period.end) == (start, end)


@pytest.mark.parametrize("question", [
    "Une conférence sur la planète Mars",   # nom de mois sans mot qui l'annonce
    "Qu'est-ce qui se passe à Pau ?",        # présent : pas un mot du passé
    "Un spectacle le 31 février",            # date impossible : ignorée
    "Des activités pour les 3/6 ans",        # tranche d'âge, pas une date
    "Les ateliers des samedis",              # une habitude, pas un jour précis
    "Des événements à venir",
])
def test_default_period_is_upcoming(question):
    assert find_period(question, TODAY) == Period("à venir", TODAY, None)


@pytest.mark.parametrize("today, question, start, end", [
    (date(2026, 11, 7), "Ce week-end", date(2026, 11, 7), date(2026, 11, 8)),            # samedi
    (date(2026, 11, 8), "Ce week-end", date(2026, 11, 8), date(2026, 11, 8)),            # dimanche
    (date(2026, 11, 7), "Le week-end prochain", date(2026, 11, 14), date(2026, 11, 15)),
    (date(2026, 11, 7), "Samedi prochain", date(2026, 11, 14), date(2026, 11, 14)),
    (date(2026, 12, 15), "Le mois prochain", date(2027, 1, 1), date(2027, 1, 31)),       # année suivante
])
def test_period_edge_cases(today, question, start, end):
    period = find_period(question, today)
    assert (period.start, period.end) == (start, end)


# --- Exemple complet ----------------------------------------------------------------------------

def test_extract_criteria_from_a_full_question():
    assert criteria("Un concert de jazz à Bordeaux ce week-end ?") == Criteria(
        period=Period("ce week-end", date(2026, 11, 7), date(2026, 11, 8)),
        city="Bordeaux",
        department=None,
    )


# --- Texte de la recherche par similarité ---------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("Du théâtre à Poitiers", "Du théâtre"),
    ("Un spectacle d'humour à Bordeaux la semaine prochaine", "Un spectacle d'humour"),
    ("Que faire à Pau ce week-end ?", "Que faire"),
    ("Une sortie avec des enfants dans les Landes cette semaine", "Une sortie avec des enfants"),
    ("Un atelier pour enfants à Bordeaux mercredi", "Un atelier pour enfants"),
    ("Une idée de sortie pour dimanche ?", "Une idée de sortie"),
    ("Un concert samedi ou dimanche", "Un concert"),
    ("Y a-t-il eu des concerts en Gironde le mois dernier ?", "Y a-t-il eu des concerts"),
    ("Des concerts de jazz à Limoges en novembre ?", "Des concerts de jazz"),
    ("Un spectacle à La Rochelle le 14 novembre", "Un spectacle"),
    ("Un concert au Bouscat demain", "Un concert"),                                # article contracté
    ("Une exposition à Saint-Jean-de-Luz samedi prochain", "Une exposition"),
    ("Une exposition à Toulouse en novembre", "Une exposition à Toulouse"),         # lieu non repéré : gardé
    ("Une dégustation de cognac", "Une dégustation de cognac"),                     # pas un lieu : rien retiré
    ("À Bordeaux ce week-end", "À Bordeaux ce week-end"),                           # rien d'autre : question entière
])
def test_search_text_leaves_out_the_place_and_period_already_used_as_filters(question, expected):
    # Le lieu et la période filtrent déjà les événements : les garder dans le texte cherché fausse
    # la similarité (« Brive-la-Gaillarde » attirait les « Games Gaillardes » plutôt que le théâtre).
    assert search_text(question, criteria(question)) == expected
