"""Fabriques de données de test, partagées par plusieurs fichiers de tests."""


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
