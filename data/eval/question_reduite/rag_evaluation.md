# Évaluation de l'assistant sur le jeu de test annoté

- Jeu de test : 20 questions ; date du jour fixée : 2026-10-29 ; données : `data/raw/events_nouvelle-aquitaine_2026-10-29.json.gz`
- Réponses : modèle `ministral-8b-2512`, index `nouvelle-aquitaine_2026-10-29` (obtenues à partir du 2026-11-20T17:35:15)
- Juge Ragas : `ministral-14b-2512` (jugement mis à jour le 2026-11-20T17:46:08)

## Synthèse

| Mesure | Valeur | Questions |
|---|---|---|
| Comportement attendu respecté | 19/20 | 20 |
| Recherche : événement attendu trouvé (Exact Match sur les identifiants) | 1,00 | 15 |
| Recherche : MRR (moyenne de 1 / rang du premier événement attendu) | 0,96 | 15 |
| Recherche : précision (part des événements trouvés qui sont pertinents) | 0,71 | 15 |
| Recherche : rappel (part de ce qui pouvait être trouvé) | 0,88 | 15 |
| Citations : part des événements cités qui sont pertinents | 0,98 | 15 |
| Similarité entre la réponse et la référence (mistral-embed) | 0,91 | 20 |
| Ragas : fidélité | — | 0 |
| Ragas : pertinence | — | 0 |
| Ragas : couverture documentaire | — | 0 |
| Ragas : exactitude factuelle | — | 0 |

## Par question

| n° | Question | Catégorie | Comportement | Rang du 1er attendu | Citations pertinentes | Similarité | Fidélité | Pertinence | Couverture | Exactitude |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Des concerts de jazz à Limoges en novembre ? | answer | conforme | 1 | 3/3 | 0,87 | — | — | — | — |
| 2 | Une exposition à voir à La Rochelle | answer | conforme | 1 | 3/3 | 0,94 | — | — | — | — |
| 3 | Un spectacle d'humour à Bordeaux la semaine prochaine | answer | conforme | 1 | 2/2 | 0,91 | — | — | — | — |
| 4 | Du théâtre à Brive-la-Gaillarde | answer | conforme | 1 | 1/1 | 0,93 | — | — | — | — |
| 5 | Un concert de musique classique à Bordeaux en décembre | answer | conforme | 1 | 3/3 | 0,92 | — | — | — | — |
| 6 | Un film documentaire à Bordeaux | answer | conforme | 1 | 2/2 | 0,92 | — | — | — | — |
| 7 | Un atelier pour enfants à Bordeaux mercredi | answer | conforme | 1 | 5/5 | 0,93 | — | — | — | — |
| 8 | Une exposition gratuite à Bordeaux ce week-end | answer | conforme | 1 | 2/2 | 0,91 | — | — | — | — |
| 9 | Un concert à Aubusson | answer | conforme | 3 | 1/1 | 0,94 | — | — | — | — |
| 10 | Une balade nature en Dordogne en novembre | answer | conforme | 1 | 2/2 | 0,95 | — | — | — | — |
| 11 | Que faire à Angoulême en novembre ? | answer | conforme | 1 | 2/2 | 0,90 | — | — | — | — |
| 12 | Que faire à Bayonne cette semaine ? | answer | conforme | 1 | 3/3 | 0,88 | — | — | — | — |
| 13 | Que faire à Niort ce week-end ? | answer | conforme | 1 | 1/1 | 0,94 | — | — | — | — |
| 14 | Un marché de Noël en décembre | answer | conforme | 1 | 2/2 | 0,92 | — | — | — | — |
| 15 | Un concert à Aubusson ce week-end | no_result | conforme | — | — | 0,95 | — | — | — | — |
| 16 | Un opéra à Limoges ce week-end | absent_genre | conforme | — | — | 0,92 | — | — | — | — |
| 17 | Quel temps fera-t-il à Bordeaux ce week-end ? | off_topic | écart | — | — | 0,85 | — | — | — | — |
| 18 | Une exposition à Toulouse en novembre | other_region | conforme | — | — | 0,87 | — | — | — | — |
| 19 | Y a-t-il eu des concerts en Lot-et-Garonne le mois dernier ? | past | conforme | 1 | 2/3 | 0,93 | — | — | — | — |
| 20 | Une idée de sortie pour dimanche ? | vague | conforme | — | — | 0,92 | — | — | — | — |

## Mesures automatiques comparées au classement humain

Le classement humain enregistré porte sur d'autres réponses (réponses régénérées depuis) : comparaison non calculée. Classer à nouveau les réponses actuelles.

## Écarts de comportement

- Question 17 (Quel temps fera-t-il à Bordeaux ce week-end ?) : événement cité malgré le refus ; manque : Nouvelle-Aquitaine

## Affirmations jugées non appuyées par les événements transmis (fidélité)

À lire avant d'interpréter la fidélité : une affirmation juste peut être jugée non appuyée (absence d'un événement, offre d'élargir la recherche).
