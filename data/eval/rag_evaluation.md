# Évaluation de l'assistant sur le jeu de test annoté

- Jeu de test : 20 questions ; date du jour fixée : 2026-10-29 ; données : `data/raw/events_nouvelle-aquitaine_2026-10-29.json.gz`
- Réponses : modèle `ministral-8b-2512`, index `nouvelle-aquitaine_2026-10-29` (obtenues à partir du 2026-11-20T10:33:43)
- Juge Ragas : `ministral-14b-2512` (jugement mis à jour le 2026-11-20T12:13:39)

## Synthèse

| Mesure | Valeur | Questions |
|---|---|---|
| Comportement attendu respecté | 17/20 | 20 |
| Recherche : événement attendu trouvé (Exact Match sur les identifiants) | 0,93 | 15 |
| Recherche : MRR (moyenne de 1 / rang du premier événement attendu) | 0,80 | 15 |
| Recherche : précision (part des événements trouvés qui sont pertinents) | 0,67 | 15 |
| Recherche : rappel (part de ce qui pouvait être trouvé) | 0,78 | 15 |
| Citations : part des événements cités qui sont pertinents | 0,99 | 14 |
| Similarité entre la réponse et la référence (mistral-embed) | 0,91 | 20 |
| Ragas : fidélité | 0,80 | 15 |
| Ragas : pertinence | 0,83 | 15 |
| Ragas : couverture documentaire | 0,57 | 15 |
| Ragas : exactitude factuelle | 0,42 | 15 |

## Par question

| n° | Question | Catégorie | Comportement | Rang du 1er attendu | Citations pertinentes | Similarité | Fidélité | Pertinence | Couverture | Exactitude |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Des concerts de jazz à Limoges en novembre ? | answer | conforme | 2 | 4/5 | 0,87 | 0,91 | 0,91 | 0,40 | 0,10 |
| 2 | Une exposition à voir à La Rochelle | answer | conforme | 1 | 1/1 | 0,90 | 0,71 | 0,81 | 0,80 | 0,42 |
| 3 | Un spectacle d'humour à Bordeaux la semaine prochaine | answer | conforme | 1 | 2/2 | 0,91 | 0,80 | 0,89 | 0,40 | 0,40 |
| 4 | Du théâtre à Brive-la-Gaillarde | answer | écart | non trouvé | 0/0 | 0,86 | 0,00 | 0,81 | 0,00 | 0,00 |
| 5 | Un concert de musique classique à Bordeaux en décembre | answer | conforme | 1 | 3/3 | 0,91 | 0,92 | 0,89 | 0,71 | 0,48 |
| 6 | Un film documentaire à Bordeaux | answer | conforme | 1 | 2/2 | 0,93 | 0,75 | 0,83 | 0,38 | 0,24 |
| 7 | Un atelier pour enfants à Bordeaux mercredi | answer | conforme | 1 | 4/4 | 0,93 | 0,81 | 0,85 | 0,50 | 0,52 |
| 8 | Une exposition gratuite à Bordeaux ce week-end | answer | conforme | 4 | 1/1 | 0,89 | 0,90 | 0,85 | 0,27 | 0,12 |
| 9 | Un concert à Aubusson | answer | conforme | 4 | 1/1 | 0,96 | 1,00 | 0,73 | 1,00 | 0,67 |
| 10 | Une balade nature en Dordogne en novembre | answer | conforme | 1 | 2/2 | 0,95 | 1,00 | 0,84 | 1,00 | 0,64 |
| 11 | Que faire à Angoulême en novembre ? | answer | conforme | 1 | 5/5 | 0,94 | 0,70 | 0,82 | 1,00 | 0,67 |
| 12 | Que faire à Bayonne cette semaine ? | answer | conforme | 1 | 5/5 | 0,88 | 1,00 | 0,86 | 0,38 | 0,32 |
| 13 | Que faire à Niort ce week-end ? | answer | conforme | 1 | 1/1 | 0,94 | 0,75 | 0,76 | 0,67 | 0,67 |
| 14 | Un marché de Noël en décembre | answer | conforme | 1 | 2/2 | 0,92 | 0,85 | 0,78 | 0,60 | 0,41 |
| 15 | Un concert à Aubusson ce week-end | no_result | conforme | — | — | 0,95 | — | — | — | — |
| 16 | Un opéra à Limoges ce week-end | absent_genre | conforme | — | — | 0,92 | — | — | — | — |
| 17 | Quel temps fera-t-il à Bordeaux ce week-end ? | off_topic | écart | — | — | 0,88 | — | — | — | — |
| 18 | Une exposition à Toulouse en novembre | other_region | conforme | — | — | 0,89 | — | — | — | — |
| 19 | Y a-t-il eu des concerts en Lot-et-Garonne le mois dernier ? | past | écart | 1 | 3/3 | 0,94 | 0,88 | 0,84 | 0,50 | 0,63 |
| 20 | Une idée de sortie pour dimanche ? | vague | conforme | — | — | 0,90 | — | — | — | — |

## Écarts de comportement

- Question 4 (Du théâtre à Brive-la-Gaillarde) : aucun événement cité
- Question 17 (Quel temps fera-t-il à Bordeaux ce week-end ?) : événement cité malgré le refus
- Question 19 (Y a-t-il eu des concerts en Lot-et-Garonne le mois dernier ?) : ne présente pas les événements comme passés

## Affirmations jugées non appuyées par les événements transmis (fidélité)

À lire avant d'interpréter la fidélité : une affirmation juste peut être jugée non appuyée (absence d'un événement, offre d'élargir la recherche).

**Question 1. Des concerts de jazz à Limoges en novembre ?**

- « Les concerts de jazz à Limoges en novembre 2026 sont les suivants: » : La phrase est incomplète et ne peut pas être directement vérifiée comme fidèle sans une liste exhaustive des concerts. Le contexte mentionne des événements et concerts, mais ne les énumère pas tous explicitement.

**Question 2. Une exposition à voir à La Rochelle**

- « The exhibition "Le H de Atlantique" will run from November 1, 2026, to November 8, 2026. » : The context states the exhibition runs from September 12, 2026, to November 8, 2026, not starting on November 1, 2026.
- « The user could broaden the search to the following week to find more available options. » : The context does not provide any guidance or suggestion for broadening the search period; this is an external recommendation not supported by the given context.

**Question 3. Un spectacle d'humour à Bordeaux la semaine prochaine**

- « The show **J'aime beaucoup ce que vous faites !** is a cult comedy and is highly entertaining, suitable for families or groups of friends. » : The context describes the show as a 'phénomène théâtral devenu culte' and mentions it has had 2 million spectators, indicating its popularity and entertainment value. However, it does not explicitly state it is suitable for families or groups of friends, though the age range (11-90) suggests broad appeal.

**Question 4. Du théâtre à Brive-la-Gaillarde**

- « No theatre events are currently available in Brive-la-Gaillarde for upcoming dates starting from October 29, 2026, according to available suggestions. » : The context does not mention theatre events at all, nor does it provide any information about events starting from October 29, 2026. The available events are related to board games and apéros for singles.
- « The information is based on the current database of events. » : The context does not explicitly mention a 'database of events,' but it provides specific details about scheduled events. This statement cannot be directly inferred from the given context.
- « A search for other types of cultural events can be conducted if desired. » : The context does not mention any mechanism or suggestion for conducting searches or exploring other types of cultural events beyond what is listed.
- « Cultural events may include music performances, exhibitions, board games, or other related activities. » : The context explicitly mentions board games (jeux de société) and apéros, but does not mention music performances or exhibitions. The statement partially aligns but is not fully supported by the context.
- « An alternative time period can be selected for event searches. » : The context does not provide any information about the possibility of selecting alternative time periods for event searches.
- « Assistance is available to explore additional options for events. » : The context mentions animators are present to help discover and learn about games, but it does not explicitly state that assistance is available for exploring additional event options beyond the listed ones.

**Question 5. Un concert de musique classique à Bordeaux en décembre**

- « Les deux premiers concerts mentionnés correspondent à la période de décembre 2026 demandée. » : Les deux premiers concerts (Concert de Noël Radio Classique et Les Quatre Saisons de Vivaldi) ont lieu respectivement du 17 au 20 décembre et le 27 décembre 2026, mais le second ne correspond pas entièrement à la période demandée (du 17 au 20 décembre).

**Question 6. Un film documentaire à Bordeaux**

- « Two documentary films are proposed in Bordeaux starting from October 29, 2026. » : The context mentions multiple documentary events starting from October 29, 2026, including projections and debates, but does not explicitly state that only two films are proposed. The phrase 'starting from' is ambiguous and could imply more events beyond the listed ones.
- « Other documentary films in Bordeaux have screening dates prior to October 29, 2026. » : The context does not mention any documentary films with screening dates before October 29, 2026. All listed events start on or after that date.

**Question 7. Un atelier pour enfants à Bordeaux mercredi**

- « Les ateliers pour enfants disponibles à Bordeaux le mercredi 4 novembre 2026 sont les suivants: » : La question est une introduction générale et non une affirmation vérifiable. Aucune liste exhaustive n'est fournie dans le contexte.
- « Un autre atelier d'éveil à l'art, également organisé par ABCD'ART, a lieu de 10h30 à 12h00 à la Maison Marandon. » : Le contexte ne mentionne qu'un seul atelier ABCD'ART à cet horaire et lieu. Il n'y a pas d'indication d'un autre atelier similaire.
- « Cet atelier est conçu pour les enfants âgés de 3 à 7 ans. » : Le contexte indique que l'atelier ABCD'ART est destiné aux enfants de 3 à 7 ans, mais cette affirmation est répétitive et non applicable à un autre atelier non mentionné.

**Question 8. Une exposition gratuite à Bordeaux ce week-end**

- « The location of the exhibition is the Jardin botanique Bastide. » : The exhibition 'Mine d'Emile s'expose' is located at Bibliothèque Bordeaux-Lac, not the Jardin botanique Bastide.

**Question 11. Que faire à Angoulême en novembre ?**

- « L'exposition *sténopé/coconstruction* est accessible pendant toute la période demandée. » : Le contexte indique que l'exposition est accessible du lundi 2 novembre au samedi 21 novembre 2026, mais ne précise pas si elle est accessible *toute la journée* pendant cette période. La formulation 'toute la période demandée' est ambiguë et non explicitement confirmée.
- « Une visite guidée gratuite de l'exposition *Croissant Indigerible / علاقات عرنسية* est organisée le dimanche 2 novembre à 16h. » : Le contexte précise qu'une visite accompagnée gratuite a lieu tous les premiers dimanches du mois à 16h, mais la date mentionnée est le dimanche 1 novembre, non le 2 novembre.
- « Les apéros ont lieu un vendredi, deux mardis et un autre jour non précisé dans la liste fournie. » : Le contexte mentionne un apéro le vendredi 6 novembre et deux apéros les mardis 3 et 10 novembre. Aucune autre date n'est précisée, donc la formulation 'un autre jour non précisé' est incorrecte.

**Question 13. Que faire à Niort ce week-end ?**

- « No other cultural events match the specified criteria for this weekend in Niort. » : The context only provides information about this specific event and does not mention or compare it to other events in Niort or elsewhere.
- « The user may choose to expand the search to other dates or types of events. » : The context does not provide any guidance or information about expanding the search to other dates or types of events.

**Question 14. Un marché de Noël en décembre**

- « Ce marché de Noël à Bordeaux aura lieu du mardi 1er décembre 2026 au dimanche 27 décembre 2026. » : Le marché commence le 27 novembre 2026, pas le 1er décembre. La date de début est incorrecte.
- « Les deux marchés de Noël mentionnés sont entièrement gratuits. » : Le marché de Bordeaux est 'entrée libre', mais celui de Fondaudège est 'entrée libre et gratuite'. Cependant, le marché de Bordeaux n'est pas explicitement décrit comme 'entièrement gratuit', seulement libre d'accès.

**Question 19. Y a-t-il eu des concerts en Lot-et-Garonne le mois dernier ?**

- « Les concerts à Saint-Quentin-du-Dropt ont été ponctués de lectures de textes durant toute la journée. » : Le contexte précise qu'il s'agit d'un 'concert ponctué de lectures de textes' de 10h à 18h le samedi 19 septembre, mais ne mentionne pas explicitement la journée du dimanche 20 septembre pour les lectures.
