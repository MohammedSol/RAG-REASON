"""
Tests d'intégration — Sprint 3.3 : Planner (Ollama réel)

Ces tests nécessitent que le serveur Ollama soit actif en local avec le
modèle qwen2.5:7b chargé. Ils sont protégés par le marqueur `integration`
pour être skippés en CI ou sans GPU.

Run (avec Ollama actif) :
    uv run pytest tests/integration/test_planner_integration.py -v -m integration
"""

from __future__ import annotations

import pytest
from reasoning.contracts.internal_models import (
    AnalysisResult,
    ExecutionPlan,
    QueryType,
)
from reasoning.planner import Planner

# ─────────────────────────────────────────────────────────────────────────────
# Tableau paramétré — 20 requêtes types réalistes
# Format : (query, query_type, expected_budget)
# ─────────────────────────────────────────────────────────────────────────────

QUERY_CASES: list[tuple[str, QueryType, int]] = [
    # ── Cas SIMPLE (budget=1) : une seule réponse attendue, pas de LLM call ──
    (
        "Qu'est-ce que la rétropropagation ?",
        QueryType.SIMPLE,
        1,
    ),
    (
        "Quel est le rôle d'un Transformer dans le NLP ?",
        QueryType.SIMPLE,
        1,
    ),
    (
        "Quand a été fondée la société OpenAI ?",
        QueryType.SIMPLE,
        1,
    ),
    (
        "Qu'est-ce que le RAG (Retrieval-Augmented Generation) ?",
        QueryType.SIMPLE,
        1,
    ),
    (
        "Quelle est la différence entre supervised et unsupervised learning ?",
        QueryType.SIMPLE,
        1,
    ),
    # ── Cas MULTI_HOP (budget=3) : plusieurs sauts de raisonnement ───────────
    (
        "Qui a créé LangChain, et quel est son rôle actuel chez LangSmith ?",
        QueryType.MULTI_HOP,
        3,
    ),
    (
        "Quel est le PDG de l'entreprise qui a publié GPT-4 ?",
        QueryType.MULTI_HOP,
        3,
    ),
    (
        "Depuis que l'attention a été introduite, comment la taille des LLMs a-t-elle évolué ?",
        QueryType.MULTI_HOP,
        3,
    ),
    (
        "Qui a inventé le BERT, et dans quelle entreprise travaillait-il à ce moment ?",
        QueryType.MULTI_HOP,
        3,
    ),
    (
        "Quel framework de vectorisation utilise Chroma, et qui maintient ce framework ?",
        QueryType.MULTI_HOP,
        3,
    ),
    (
        "Quel est le lien entre l'université de Stanford, le projet GPT-3 et Sam Altman ?",
        QueryType.MULTI_HOP,
        3,
    ),
    (
        "Comment fonctionne le mécanisme d'attention, et quel modèle l'a popularisé ?",
        QueryType.MULTI_HOP,
        3,
    ),
    # ── Cas COMPARATIVE (budget=2) : comparaison entre deux entités ──────────
    (
        "Compare le RAG et le fine-tuning pour une application médicale.",
        QueryType.COMPARATIVE,
        2,
    ),
    (
        "Quelle est la différence entre BM25 et un retriever dense type FAISS ?",
        QueryType.COMPARATIVE,
        2,
    ),
    (
        "Compare les architectures de BERT et GPT en termes d'objectifs d'entraînement.",
        QueryType.COMPARATIVE,
        2,
    ),
    (
        "En quoi Chroma et Pinecone diffèrent-ils pour une application RAG en production ?",
        QueryType.COMPARATIVE,
        2,
    ),
    (
        "Compare les stratégies de chunking fixe et sémantique pour le RAG.",
        QueryType.COMPARATIVE,
        2,
    ),
    (
        "Quelles sont les différences entre LiteLLM et l'API OpenAI directe ?",
        QueryType.COMPARATIVE,
        2,
    ),
    (
        "Compare l'approche ReAct et l'approche Plan-and-Solve pour le raisonnement LLM.",
        QueryType.COMPARATIVE,
        2,
    ),
    (
        "Quelles sont les différences entre Ollama et LM Studio pour le déploiement local ?",
        QueryType.COMPARATIVE,
        2,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Fixture — Planner réel (nécessite Ollama actif)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_planner() -> Planner:
    """Instancie un Planner réel qui tape sur Ollama/qwen2.5:7b en local."""
    return Planner(model="ollama/qwen2.5:7b", temperature=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests d'intégration paramétrés
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize(
    "query, query_type, budget",
    QUERY_CASES,
    ids=[f"{qt.value}_{i}" for i, (_, qt, _) in enumerate(QUERY_CASES, start=1)],
)
def test_planner_live(
    live_planner: Planner,
    query: str,
    query_type: QueryType,
    budget: int,
) -> None:
    """Test d'intégration : le Planner réel produit un ExecutionPlan valide.

    Assertions basiques (ne pas over-asserter sur le contenu métier du plan,
    car le LLM reste non-déterministe) :
    1. Le retour est bien un ExecutionPlan (contrat Pydantic respecté)
    2. Le plan contient au moins 1 étape
    3. Le nombre d'étapes ne dépasse jamais le budget
    4. Le DAG est acyclique (validé par Kahn)
    5. Chaque PlanStep a un step_id et une sub_query non vide
    """
    analysis = AnalysisResult(
        query_type=query_type,
        confidence=0.90,
        detected_entities=[],
        reasoning_budget=budget,
    )

    result = live_planner.decompose(query=query, analysis=analysis)

    # ── Assertion 1 : type de retour correct ──────────────────────────────
    assert isinstance(
        result, ExecutionPlan
    ), f"decompose() doit retourner un ExecutionPlan, reçu {type(result)}"

    # ── Assertion 2 : au moins 1 étape dans le plan ───────────────────────
    assert len(result.steps) >= 1, f"Le plan ne doit jamais être vide. Query: {query!r}"

    # ── Assertion 3 : budget respecté ─────────────────────────────────────
    assert len(result.steps) <= budget, (
        f"Budget dépassé : {len(result.steps)} étapes pour un budget de {budget}. "
        f"Query: {query!r}"
    )

    # ── Assertion 4 : DAG acyclique ────────────────────────────────────────
    assert live_planner._validate_dag(result) is True, (
        f"Le plan généré contient un cycle ou une référence invalide. "
        f"Query: {query!r}"
    )

    # ── Assertion 5 : champs obligatoires non vides dans chaque PlanStep ──
    for step in result.steps:
        assert step.step_id, f"step_id vide détecté dans le plan. Query: {query!r}"
        assert step.sub_query, f"sub_query vide pour {step.step_id}. Query: {query!r}"

    # ── Assertion bonus : la requête originale est bien préservée ──────────
    assert result.original_query == query
