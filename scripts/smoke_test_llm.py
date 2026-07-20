"""
Smoke-test de connectivité LiteLLM → Ollama.

Usage :
    uv run scripts/smoke_test_llm.py

Ce script vérifie que :
    1. Les variables d'environnement sont correctement chargées.
    2. Le serveur Ollama est joignable à OLLAMA_BASE_URL.
    3. Le modèle DEFAULT_FAST_MODEL répond à une invite simple.
"""

import os
import sys

from dotenv import load_dotenv
from litellm import completion

# ── Chargement des variables d'environnement ─────────────────────────────────
load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_FAST_MODEL: str = os.getenv("DEFAULT_FAST_MODEL", "ollama/qwen2.5:3b")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Prompt de test ────────────────────────────────────────────────────────────
TEST_PROMPT = "Bonjour, es-tu en ligne ? Réponds par une seule phrase."


def run_smoke_test() -> None:
    """Envoie un prompt minimal à Ollama via LiteLLM et affiche la réponse."""
    print("=" * 60)
    print("RAG-REASON — Smoke-Test LiteLLM → Ollama")
    print("=" * 60)
    print(f"  Modèle     : {DEFAULT_FAST_MODEL}")
    print(f"  API base   : {OLLAMA_BASE_URL}")
    print(f"  Prompt     : {TEST_PROMPT}")
    print("-" * 60)

    try:
        response = completion(
            model=DEFAULT_FAST_MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            api_base=OLLAMA_BASE_URL,
        )

        answer: str = response.choices[0].message.content or ""
        print(f"  Réponse    : {answer.strip()}")
        print("-" * 60)
        print("✅ Connexion LiteLLM → Ollama : OK")
        print("=" * 60)

    except Exception as exc:
        print("-" * 60)
        print(f"❌ ÉCHEC de la connexion : {exc}")
        print("   Vérifiez qu'Ollama est lancé et que le modèle est disponible :")
        print("     ollama serve")
        print(f"     ollama pull {DEFAULT_FAST_MODEL.replace('ollama/', '')}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
