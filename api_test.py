"""Test fonctionnel de l'API : appelle chaque route comme le ferait un client, contre l'API déjà lancée.

Depuis la racine du projet, après avoir lancé l'API (uvicorn api.main:app) dans un autre terminal :
    python api_test.py                              # API locale : http://127.0.0.1:8000
    python api_test.py --url http://127.0.0.1:8000  # autre adresse

Affiche [OK] ou [ECHEC] pour chaque vérification ; code de sortie 0 si tout passe, 1 sinon.
La question de démonstration appelle Mistral (une requête de génération et une d'embeddings).
"""

import argparse
import sys

import httpx

DEMO = {"question": "Un concert de jazz à Bordeaux ce week-end", "today": "2026-10-29"}


def check(name: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    return name, ok, detail


def run_checks(client) -> list[tuple[str, bool, str]]:
    """Vérifications de l'API ; `client` est un client httpx (ou le TestClient de FastAPI, pour les tests)."""
    results = []

    response = client.get("/health")
    results.append(check("GET /health : assistant prêt", response.status_code == 200
                         and response.json().get("status") == "ok", f"{response.status_code} {response.text[:150]}"))

    response = client.get("/metadata")
    index = response.json().get("index") if response.status_code == 200 else None
    results.append(check("GET /metadata : index décrit", bool(index and index.get("events")),
                         f"{response.status_code} index : {index and index.get('name')}"))

    response = client.post("/ask", json=DEMO)
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    answered = response.status_code == 200 and bool(body.get("answer")) and isinstance(body.get("sources"), list)
    results.append(check(f"POST /ask : « {DEMO['question']} »", answered,
                         (body.get("answer") or body.get("detail") or response.text)[:200].replace("\n", " ")))

    response = client.post("/ask", json={"question": ""})
    results.append(check("POST /ask : question vide refusée (422)", response.status_code == 422,
                         str(response.status_code)))

    response = client.post("/rebuild")
    results.append(check("POST /rebuild sans jeton : refusé (401 ou 403)", response.status_code in (401, 403),
                         f"{response.status_code} {response.text[:150]}"))

    response = client.get("/docs")
    results.append(check("GET /docs : documentation Swagger", response.status_code == 200, str(response.status_code)))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test fonctionnel de l'API Puls-Events (API déjà lancée).")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Adresse de l'API.")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):   # accents lisibles même dans une sortie redirigée (Windows)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        with httpx.Client(base_url=args.url, timeout=120) as client:
            results = run_checks(client)
    except httpx.ConnectError:
        print(f"[ECHEC] API injoignable à {args.url} : la lancer d'abord (uvicorn api.main:app)", file=sys.stderr)
        return 1
    for name, ok, detail in results:
        print(f"[{'OK' if ok else 'ECHEC'}] {name}\n        {detail}")
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} vérifications réussies")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
