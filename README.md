# Puls-Events : assistant RAG de recommandation d'événements culturels

Preuve de concept (POC) d'un chatbot qui répond aux questions des utilisateurs sur des événements
culturels. Il repose sur un système **RAG** (*Retrieval-Augmented Generation*) : les événements sont
recherchés par similarité de sens dans une base vectorielle (**Faiss**), puis un modèle de langage
(**Mistral**) rédige la réponse à partir des événements trouvés. L'ensemble est orchestré par
**LangChain** et exposé via une **API REST** (FastAPI).

> Projet en cours de construction : ce README est complété au fil des étapes.

## Objectifs

Démontrer aux équipes produit et marketing de Puls-Events :
- la **faisabilité technique** d'un assistant RAG sur les données d'événements Open Agenda ;
- sa **pertinence métier** : des recommandations à jour, appuyées sur des événements réels ;
- sa **performance**, mesurée sur un jeu de test annoté.

## Stack technique

| Composant | Outil |
|---|---|
| Données | API Open Agenda (jeu public OpenDataSoft) |
| Vectorisation (embeddings) | Mistral `mistral-embed-2312` |
| Base vectorielle | Faiss (`faiss-cpu`), index exact Flat (produit scalaire) |
| Orchestration | LangChain |
| Génération des réponses | Mistral Ministral 8B (`ministral-8b-2512`), accessible avec la formule gratuite ; Mistral Large 4 visé avec une offre payante |
| API | FastAPI + Uvicorn |
| Conteneurisation | Docker |
| Évaluation | Jeu de test annoté + Ragas |

## Données

Les événements proviennent du jeu de données public
[« Événements publics OpenAgenda »](https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda/)
(éditeur : OpenAgenda, diffusion : OpenDataSoft), publié sous licence *Open License v1.0*.

- **Zone** : Nouvelle-Aquitaine.
- **Période** : événements terminés depuis moins d'un an et événements à venir, par rapport à la date de référence.
- **Photo de référence**, versionnée dans `data/raw/` : `events_nouvelle-aquitaine_2026-10-29.json.gz`
  (34 517 événements, récupérés le 29/10/2026), ainsi qu'un échantillon de développement
  (`events_creuse_2026-10-29.json.gz`, 903 événements).

Pour retélécharger les données (aucune clé n'est nécessaire) :

```bash
python -m scripts.fetch_events                              # date du jour
python -m scripts.fetch_events --reference-date 2026-10-29  # même filtre que la photo de référence
python -m scripts.fetch_events --department Creuse          # petit échantillon
```

> Les données Open Agenda évoluent en continu : un nouveau téléchargement donnera des résultats
> légèrement différents de la photo de référence, sur laquelle portent l'évaluation et le jeu de test annoté.

### Nettoyage

Les données brutes sont nettoyées par 13 règles (doublons, événements annulés, agendas emploi et
formation exclus, dates aberrantes, HTML, conversion en heure de Paris…), justifiées dans la section 8
du notebook [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb) :

```bash
python -m scripts.preprocess_events     # photo de référence : 34 517 -> 18 771 événements, dans data/processed/
```

### Index vectoriel

L'index est construit à partir des données brutes : nettoyage, découpage en chunks (un événement par
document ; descriptions de plus de 1 500 caractères découpées ; en-tête de contexte répété sur chaque chunk),
vectorisation par `mistral-embed-2312` (lots d'environ 15 000 tokens, une requête par seconde au maximum),
puis index Faiss exact (Flat, produit scalaire). Les choix sont justifiés dans les notebooks
[`02_embeddings.ipynb`](notebooks/02_embeddings.ipynb) et [`03_faiss_index.ipynb`](notebooks/03_faiss_index.ipynb).

```bash
python -m scripts.build_index      # photo de référence : 20 644 chunks, ≈ 12 min, ≈ 6,6 millions de tokens (≈ 0,67 $)
python -m scripts.build_index --input data/raw/events_creuse_2026-10-29.json.gz   # échantillon : ≈ 15 s
```

L'index est enregistré dans `vectorstore/<zone>_<date>/` (`index.faiss`, `index.pkl`, `manifest.json`) ;
il n'est pas versionné. La clé `MISTRAL_API_KEY` est nécessaire.

## Poser une question à l'assistant

Le script `scripts/ask.py` fait passer une question par toute la chaîne RAG :

1. **critères** : le lieu (ville ou département) et la période (« ce week-end », « demain », « le 14 novembre »…)
   sont repérés dans la question par des règles ; sans période précisée, seuls les événements à venir sont retenus ;
2. **recherche** : les 5 événements les plus proches de la question parmi ceux qui respectent ces critères
   (au moins une séance pendant la période), dans l'index Faiss ;
3. **génération** : Ministral 8B rédige la réponse à partir de ces seuls événements, en citant leurs références `[n]` ;
4. **réponse** : le texte, les sources citées (leurs liens viennent des données, pas du modèle) et les
   métadonnées (modèle, durées, tokens). Si aucun événement ne respecte les critères, un message fixe est
   renvoyé, sans appel au modèle.

Prérequis : l'index de référence construit (`python -m scripts.build_index`) et la clé `MISTRAL_API_KEY`
dans `.env`. Les commandes sont les mêmes sous Windows, macOS et Linux (environnement activé) :

```bash
python -m scripts.ask "Un concert de jazz à Bordeaux ce week-end" --today 2026-10-29
python -m scripts.ask "Que faire ce week-end ?"              # avec la vraie date du jour
python -m scripts.ask "Une exposition à Limoges" --json      # réponse complète en JSON
```

| Option | Rôle |
|---|---|
| `--today AAAA-MM-JJ` | Date du jour, qui fixe le sens de « ce week-end », « demain »… (par défaut : la vraie date). Avec `2026-10-29`, date de la photo de référence, les résultats sont reproductibles. |
| `--index DOSSIER` | Index à interroger (par défaut : `vectorstore/nouvelle-aquitaine_2026-10-29`) |
| `--json` | Réponse complète en JSON : texte, sources, critères repérés, métadonnées |

Exemple de sortie (les données ne contiennent aucun concert de jazz à Bordeaux ce week-end : le modèle le
dit, puis propose l'événement le plus proche) :

```text
Question : Un concert de jazz à Bordeaux ce week-end
Critères : lieu : Bordeaux ; période : ce week-end, du samedi 31 octobre 2026 au dimanche 1 novembre 2026

Je suis désolé, mais aucun concert de jazz n'est proposé ce week-end à Bordeaux dans les événements listés. Si vous êtes ouvert à d'autres styles musicaux, le **concert de musique de chambre** intitulé *"Fais-moi peur"* le dimanche 1 novembre à 11h00 à l'Auditorium de Bordeaux pourrait vous intéresser, car il explore des œuvres inspirées par le cinéma, dont certaines ont une ambiance mystérieuse et immersive [1].

Sources :
[1] Concert du dimanche — Auditorium de Bordeaux, Bordeaux
    https://openagenda.com/ville-de-bordeaux/events/concert-du-dimanche-1929716

ministral-8b-2512 ; 5 événements trouvés ; recherche 1092 ms ; génération 1412 ms ; 2007 tokens en entrée, 102 en sortie
```

> **Formule gratuite de Mistral** : Ministral 8B y est limité à 2 requêtes par minute. Au-delà, le script
> affiche `[ERREUR] Limite de débit Mistral atteinte : réessayez dans une minute.` Le modèle et ses
> réglages sont dans `rag/config.py` ; leur choix est justifié dans le notebook
> [`04_rag_chain.ipynb`](notebooks/04_rag_chain.ipynb) (partie B).
>
> L'avertissement « You are sending unauthenticated requests to the HF Hub » est sans conséquence :
> LangChain récupère sur Hugging Face le tokenizer de Mistral, qui sert à calculer la taille des lots
> envoyés au modèle d'embedding.

## Évaluer l'assistant

Le jeu de test annoté (`data/eval/test_set.json`) compte 20 questions : 14 auxquelles les données répondent
et 6 cas limites (aucun résultat, genre absent, hors sujet, autre région, événements passés, question vague).
Chaque référence est rédigée à partir des données de la photo de référence ; `tests/test_eval_test_set.py`
vérifie qu'elle reste cohérente avec ces données.

Le script `scripts/evaluate_rag.py` évalue l'assistant en deux phases, enregistrées au fil de l'eau :

1. **réponses** : les 20 questions sont posées à l'assistant, avec la date du jour fixée au 29/10/2026 ;
2. **jugement** :
   - mesures sans LLM : événements attendus trouvés par la recherche (Exact Match sur les identifiants,
     rang, précision, rappel), événements cités pertinents, comportement attendu des cas limites,
     similarité entre la réponse et la référence (`mistral-embed`) ;
   - métriques Ragas, avec Ministral 14B comme juge, sur les 15 questions à événements attendus :
     fidélité au contexte, pertinence, couverture documentaire, exactitude factuelle.

```bash
python -m scripts.evaluate_rag                    # les deux phases (≈ 1 h 20 avec la formule gratuite)
python -m scripts.evaluate_rag --phase answers    # phase 1 seulement (≈ 10 min)
python -m scripts.evaluate_rag --phase judge      # phase 2, sur les réponses enregistrées
python -m scripts.evaluate_rag --no-ragas         # mesures sans juge seulement (≈ 1 min)
```

Résultats dans `data/eval/` : `rag_answers.json` (réponses complètes), `rag_evaluation.json` (mesures et
détail du juge), `rag_evaluation.md` (résumé lisible). Une relance reprend là où le calcul s'était arrêté.
Les 20 réponses ont aussi été classées à la main, à l'aveugle (correcte / partielle / incorrecte, motif
vérifié dans les données) : `manual_judgments.json`. Le résumé compare les mesures automatiques à ce
classement, pour savoir lesquelles distinguent vraiment les bonnes réponses des mauvaises.
Avec la formule gratuite (2 requêtes par minute et par modèle), les requêtes sont espacées de 31 s ; le
calcul s'arrête au premier refus de Mistral qu'une attente ne résoudrait pas (accès refusé, serveurs
saturés). Les statistiques d'usage de Ragas sont désactivées (`RAGAS_DO_NOT_TRACK`).

## Structure du dépôt

```
.
├── data/
│   ├── raw/                Données brutes Open Agenda (photo de référence + échantillon)
│   ├── processed/          Données nettoyées (générées par le script, non versionnées)
│   └── eval/               Jeu de test annoté (test_set.json), résultats de l'évaluation, réponses du notebook 04
├── notebooks/
│   ├── 01_exploration.ipynb    Exploration des données et règles de nettoyage
│   ├── 02_embeddings.ipynb     Démonstration des embeddings Mistral (dimension, similarité, limites)
│   ├── 03_faiss_index.ipynb    Tests de recherche et choix de l'index Faiss
│   └── 04_rag_chain.ipynb      Recherche filtrée (partie A), choix du modèle et réglage du prompt (partie B)
├── rag/                    Logique métier (package Python)
│   ├── config.py           Paramètres centralisés (chemins, source, nettoyage, découpage, index, modèle de chat)
│   ├── fetch.py            Récupération des événements Open Agenda
│   ├── preprocess.py       Nettoyage et structuration des événements
│   ├── chunking.py         Découpage en chunks (en-tête de contexte, métadonnées)
│   ├── indexing.py         Vectorisation Mistral par lots et index Faiss
│   ├── criteria.py         Repérage du lieu et de la période d'une question (règles)
│   ├── retrieval.py        Recherche filtrée dans l'index (critères, un résultat par événement)
│   ├── prompts.py          Textes envoyés au modèle (règles, événements numérotés)
│   ├── assistant.py        Chaîne RAG complète (LangChain LCEL) : de la question à la réponse et ses sources
│   ├── evaluation.py       Mesures d'évaluation sans LLM (recherche, citations, comportement) et synthèse
│   └── judge.py            Juge LLM de l'évaluation : métriques Ragas avec un modèle Mistral
├── scripts/
│   ├── check_env.py        Vérification de l'environnement (imports, clé API, accès Mistral)
│   ├── fetch_events.py     Téléchargement des événements dans data/raw/
│   ├── preprocess_events.py    Nettoyage des événements dans data/processed/
│   ├── build_index.py      Construction de l'index vectoriel dans vectorstore/
│   ├── ask.py              Question à l'assistant en ligne de commande
│   └── evaluate_rag.py     Évaluation sur le jeu de test annoté (réponses, puis jugement)
├── tests/
│   ├── factories.py            Événements de test partagés
│   ├── fake_judge.py           Faux juge et faux embeddings pour tester Ragas sans appel
│   ├── test_fetch.py           Tests unitaires de la récupération
│   ├── test_raw_data.py        Contrôle qualité des données brutes
│   ├── test_preprocess.py      Tests unitaires du nettoyage
│   ├── test_processed_data.py  Contrôle qualité des données nettoyées
│   ├── test_chunking.py        Tests unitaires du découpage
│   ├── test_indexing.py        Tests unitaires de la vectorisation et de l'index
│   ├── test_index_data.py      Contrôle de l'index (ignoré tant qu'il n'est pas construit)
│   ├── test_criteria.py        Tests unitaires du repérage des critères
│   ├── test_criteria_data.py   Contrôle des règles de repérage sur les villes de la photo de référence
│   ├── test_retrieval.py       Tests unitaires de la recherche filtrée
│   ├── test_prompts.py         Tests unitaires des textes envoyés au modèle
│   ├── test_assistant.py       Tests unitaires de la chaîne complète (faux modèle, sans appel à Mistral)
│   ├── test_ask.py             Tests unitaires du script de question
│   ├── test_dependencies.py    Contrôle des dépendances d'évaluation (import de Ragas)
│   ├── test_eval_test_set.py   Contrôle du jeu de test annoté sur la photo de référence
│   ├── test_evaluation.py      Tests unitaires des mesures d'évaluation sans LLM
│   ├── test_judge.py           Tests unitaires du juge Ragas (faux juge)
│   └── test_evaluate_rag.py    Tests unitaires du script d'évaluation (faux assistant, faux juge)
├── vectorstore/            Index Faiss (généré par le script, non versionné)
├── pytest.ini              Configuration des tests
├── requirements.in         Dépendances directes d'exécution (écrites à la main)
├── requirements.txt        Toutes les dépendances d'exécution, versions figées (calculé)
├── requirements-dev.in     Dépendances de développement : tests, évaluation, notebooks
├── requirements-dev.txt    Toutes les dépendances de développement, versions figées (calculé)
├── .env.example            Modèle du fichier .env (variables d'environnement attendues)
├── .gitattributes          Normalisation des fins de ligne (compatibilité Windows / Linux)
└── .gitignore
```

## Prérequis

- **Git** ;
- **Python 3.12** ([python.org](https://www.python.org/downloads/)), inutile si vous utilisez
  [**uv**](https://docs.astral.sh/uv/), qui le télécharge automatiquement ;
- une **clé API Mistral** (offre gratuite disponible) : https://console.mistral.ai, menu *Clés API*.

Toutes les commandes se lancent **depuis la racine du dépôt**.

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Isydoria/Projet_7_-Concevez_et_d-ployez_un_syst-me_RAG.git
cd Projet_7_-Concevez_et_d-ployez_un_syst-me_RAG
```

### 2. Créer et activer l'environnement virtuel

| | macOS / Linux | Windows (PowerShell) |
|---|---|---|
| Création avec uv | `uv venv --python 3.12 .venv` | `uv venv --python 3.12 .venv` |
| Création sans uv | `python3.12 -m venv .venv` | `py -3.12 -m venv .venv` |
| Activation | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |

> Windows : si PowerShell refuse l'activation (exécution de scripts désactivée), lancer une fois
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, ou utiliser l'invite de commandes (`cmd`)
> avec `.venv\Scripts\activate.bat`.

### 3. Installer les dépendances

| Usage | Avec uv | Sans uv |
|---|---|---|
| Développement complet (tests, évaluation, notebooks) | `uv pip install -r requirements-dev.txt` | `pip install -r requirements-dev.txt` |
| Exécution seule (API) | `uv pip install -r requirements.txt` | `pip install -r requirements.txt` |

Les versions sont figées et valables sous Windows, macOS et Linux.

### 4. Configurer la clé API Mistral

Copier le modèle `.env.example` en `.env` :
- macOS / Linux : `cp .env.example .env`
- Windows (PowerShell) : `Copy-Item .env.example .env`

Puis ouvrir `.env` et renseigner la clé : `MISTRAL_API_KEY=votre_clé`.
Le fichier `.env` est ignoré par Git : la clé n'est jamais versionnée.

### 5. Vérifier l'installation

```bash
python -m scripts.check_env
```

Le script vérifie la version de Python, les imports des bibliothèques clés, la présence de la clé
et l'accès à l'API Mistral. Il doit se terminer par `Environnement prêt.`

## Lancer les tests

```bash
python -m pytest        # résumé
python -m pytest -v     # détail, test par test
```

Les tests n'appellent aucune API : les tests unitaires simulent les appels, et les tests de contrôle
qualité lisent les données versionnées. Ils ne nécessitent ni connexion Internet ni clé.
Les tests de `tests/test_index_data.py` sont ignorés tant que l'index de référence n'a pas été construit.

## Mettre à jour les dépendances

Les dépendances directes sont écrites à la main dans `requirements.in` (exécution) et
`requirements-dev.in` (développement). Les fichiers `.txt` sont **calculés** avec uv, dans cet ordre :
le développement d'abord, puis l'exécution avec les mêmes versions.

```bash
uv pip compile --universal --python-version 3.12 requirements-dev.in -o requirements-dev.txt
uv pip compile --universal --python-version 3.12 requirements.in -c requirements-dev.txt -o requirements.txt
```

Ajouter `--upgrade` aux deux commandes pour passer aux dernières versions compatibles, puis
vérifier les versions obtenues.

## Auteur

Nathanaëlle Debaque
