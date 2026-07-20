# RAG-REASON — Module de Raisonnement Avancé

> Module **REASONING** d'un agent RAG multi-sauts, orchestré via LangGraph.
> Composants : Query Analyzer · Planner · Critic · Verifier

## Architecture

```
User Query → [Query Analyzer] → [Planner] ⇄ MODULE ACTION (JSON) ⇄ [Critic] → [Verifier] → Answer
```

## Prérequis

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de dépendances)
- [Ollama](https://ollama.com/) (exécution locale des LLMs)

## Installation rapide

```bash
# 1. Cloner le dépôt
git clone https://github.com/MohammedSol/RAG-REASON.git
cd RAG-REASON

# 2. Installer les dépendances
uv sync

# 3. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env selon votre configuration

# 4. Puller le modèle Ollama
ollama pull qwen2.5:7b
```

## Utilisation

> *Documentation à compléter au fil des sprints.*

## Évaluation RAGAS

> *Pipeline d'évaluation à compléter en Sprint 7.*

## Structure du projet

```
src/reasoning/      ← Code source du module REASONING
tests/              ← Tests unitaires, intégration et évaluation
configs/            ← Fichiers de configuration
docs/               ← Documentation technique
scripts/            ← Utilitaires (setup, smoke-tests)
```

## Feuille de Route

Voir [plan.md](plan.md) pour le détail des sprints.
