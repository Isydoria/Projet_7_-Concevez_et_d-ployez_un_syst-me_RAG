"""Vérifie que l'environnement du projet est prêt.

Contrôles effectués :
1. la version de Python ;
2. les imports des bibliothèques clés (équivalents actuels des imports cités dans l'énoncé) ;
3. la présence de la clé MISTRAL_API_KEY, lue depuis le fichier .env (jamais affichée) ;
4. la connexion à l'API Mistral, en listant les modèles disponibles (aucun crédit consommé).

Lancement, depuis la racine du projet :
    python -m scripts.check_env
"""

import importlib
import os
import sys

from dotenv import load_dotenv

MIN_PYTHON = (3, 12)

# (module, objets attendus) : équivalents actuels des imports proposés par l'énoncé
REQUIRED_IMPORTS = [
    ("faiss", []),
    ("langchain_community.vectorstores", ["FAISS"]),
    ("langchain_mistralai", ["MistralAIEmbeddings", "ChatMistralAI"]),
    ("langchain_text_splitters", ["RecursiveCharacterTextSplitter"]),
    ("mistralai.client", ["Mistral"]),
    ("pandas", []),
    ("fastapi", ["FastAPI"]),
]


def check_python() -> bool:
    """Vérifie que la version de Python est suffisante."""
    version = ".".join(str(n) for n in sys.version_info[:3])
    if sys.version_info >= MIN_PYTHON:
        print(f"[OK]     Python {version}")
        return True
    print(f"[ERREUR] Python {version} : version 3.12 minimum requise")
    return False


def check_imports() -> bool:
    """Importe chaque bibliothèque clé et vérifie la présence des objets attendus."""
    all_ok = True
    for module_name, names in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            for name in names:
                getattr(module, name)
            print(f"[OK]     import {module_name} {', '.join(names)}")
        except (ImportError, AttributeError) as error:
            print(f"[ERREUR] import {module_name} : {error}")
            all_ok = False
    return all_ok


def load_api_key() -> str | None:
    """Charge le fichier .env et renvoie la clé Mistral, sans jamais l'afficher."""
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if api_key:
        print(f"[OK]     MISTRAL_API_KEY trouvée ({len(api_key)} caractères)")
        return api_key
    print("[ERREUR] MISTRAL_API_KEY absente : copier .env.example en .env et y renseigner la clé")
    return None


def check_mistral_api(api_key: str) -> bool:
    """Liste les modèles disponibles via l'API Mistral (aucun crédit consommé)."""
    from mistralai.client import Mistral

    try:
        with Mistral(api_key=api_key) as client:
            response = client.models.list()
    except Exception as error:  # clé refusée, pas de réseau...
        print(f"[ERREUR] API Mistral : {type(error).__name__} – {error}")
        return False

    model_ids = sorted(model.id for model in (response.data or []))
    embedding_ids = [model_id for model_id in model_ids if "embed" in model_id]
    print(f"[OK]     API Mistral : {len(model_ids)} modèles disponibles")
    print(f"         Embedding : {', '.join(embedding_ids)}")
    print(f"         Tous      : {', '.join(model_ids)}")
    return True


def main() -> int:
    """Lance toutes les vérifications et renvoie 0 si tout est OK, 1 sinon."""
    print("Vérification de l'environnement du projet\n")
    results = [check_python(), check_imports()]
    api_key = load_api_key()
    results.append(api_key is not None and check_mistral_api(api_key))

    if all(results):
        print("\nEnvironnement prêt.")
        return 0
    print("\nCertaines vérifications ont échoué (voir les lignes [ERREUR]).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
