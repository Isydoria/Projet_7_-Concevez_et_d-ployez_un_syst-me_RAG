# Image de l'API Puls-Events (FastAPI + système RAG), construite et lancée en local.
# Construction et lancement : docker compose up --build (voir le README, « Lancer avec Docker »).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dépendances d'exécution seulement (sans Ragas, pytest ni Jupyter), installées avant le code :
# cette couche reste en cache tant que requirements.txt ne change pas.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Tokenizer de Mistral, utilisé par LangChain pour découper les lots d'embeddings : téléchargé une fois ici
# plutôt qu'à chaque démarrage (sans lui, LangChain se rabat sur un découpage approximatif).
RUN python -c "from tokenizers import Tokenizer; Tokenizer.from_pretrained('mistralai/Mixtral-8x7B-v0.1')" \
    || echo "Tokenizer non téléchargé : LangChain le récupérera au démarrage"

# Code et données de référence (pour reconstruire l'index si besoin) ; aucun secret (voir .dockerignore)
COPY rag/ rag/
COPY api/ api/
COPY scripts/ scripts/
COPY data/raw/ data/raw/

# Utilisateur sans droits d'administrateur ; l'index est rangé dans vectorstore/, monté comme volume
RUN useradd --create-home appuser && mkdir -p vectorstore && chown -R appuser /app
USER appuser

EXPOSE 8000
# Prêt quand /health répond 200 (le premier démarrage télécharge d'abord l'index)
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Prépare l'index s'il manque (téléchargement vérifié, sinon construction), puis lance l'API
CMD ["python", "-m", "scripts.serve", "--host", "0.0.0.0", "--port", "8000"]
