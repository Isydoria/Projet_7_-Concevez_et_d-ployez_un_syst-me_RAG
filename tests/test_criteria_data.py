"""Contrôle des règles de repérage des villes sur les vraies villes de la photo de référence.

tests/test_criteria.py teste le CODE sur quelques villes choisies ; ici, on vérifie que les règles
restent sûres sur toutes les villes réelles : chaque ville est retrouvée, et aucune expression
courante n'est prise pour une ville. Les tests lisent le fichier versionné (aucun appel réseau).
"""

import pytest

from rag import config
from rag.criteria import CITY_PREPOSITIONS, build_city_index, find_place, normalize
from rag.fetch import load_raw_events
from rag.preprocess import clean_text

# Expressions fréquentes dans une question, avec une préposition de lieu mais sans ville
COMMON_EXPRESSIONS = [
    "des événements à venir", "à partir de demain", "que faire à deux", "à faire en famille",
    "à voir absolument", "à ne pas manquer", "à l'affiche", "à la une", "à quelle heure",
    "à combien", "à moins de 10 euros", "à la plage", "à la mer", "à la montagne",
    "à la campagne", "à la ferme", "à la médiathèque", "à l'église", "au programme", "au menu",
    "au théâtre", "au cinéma", "au musée", "au château", "au jardin", "au parc", "au marché",
    "au bord de l'eau", "au festival", "aux enfants", "aux familles", "aux adultes", "sur place",
    "sur réservation", "sur inscription", "sur le thème du jazz", "vers 20h", "vers midi",
    "près de chez moi", "près d'ici", "autour de moi", "à côté de chez moi",
]


@pytest.fixture(scope="module")
def city_index():
    """Villes de la photo de référence, nettoyées comme dans le pipeline."""
    _, events = load_raw_events(config.REFERENCE_RAW_FILE)
    return build_city_index(clean_text(event.get("location_city")) for event in events)


def test_every_city_is_found_after_a_preposition(city_index):
    missed = [name for name in city_index.values()
              if find_place(normalize(f"à {name}"), CITY_PREPOSITIONS, city_index) != name]
    assert missed == []


def test_common_expressions_are_not_taken_for_cities(city_index):
    found = {expression: find_place(normalize(expression), CITY_PREPOSITIONS, city_index)
             for expression in COMMON_EXPRESSIONS}
    assert {expression: city for expression, city in found.items() if city} == {}
