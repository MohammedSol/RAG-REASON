"""
Tests unitaires — Sprint 3.3 : Planner (mocks LiteLLM)

Ces tests sont rapides et isolés. Ils ne nécessitent PAS Ollama en arrière-plan.
Toutes les réponses LLM sont simulées via unittest.mock.patch.

Run:
    uv run pytest tests/unit/test_planner.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from reasoning.contracts.internal_models import (
    AnalysisResult,
    ExecutionPlan,
    QueryType,
    StepStatus,
)
from reasoning.planner import Planner

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — Fixtures & factories
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def planner() -> Planner:
    """Instance Planner with neutral params for unit testing."""
    return Planner(model="ollama/qwen2.5:7b", temperature=0.0)


def _make_analysis(
    query_type: QueryType,
    budget: int,
    entities: list[str] | None = None,
    confidence: float = 0.92,
) -> AnalysisResult:
    """Factory pour créer un AnalysisResult de test rapidement."""
    return AnalysisResult(
        query_type=query_type,
        confidence=confidence,
        detected_entities=entities or [],
        reasoning_budget=budget,
    )


def _make_llm_response(toon_content: str) -> MagicMock:
    """Construit un objet MagicMock imitant la structure de retour de litellm.completion.

    Args:
        toon_content: La chaîne TOON brute à retourner dans choices[0].message.content.
    """
    mock_response = MagicMock()
    mock_response.choices[0].message.content = toon_content
    return mock_response


# ─────────────────────────────────────────────────────────────────────────────
# Helper — TOON builders pour les mocks
# ─────────────────────────────────────────────────────────────────────────────


def _toon_header(rationale: str, total_steps: int) -> str:
    """Génère un bloc TOON d'en-tête (plan_rationale + total_steps)."""
    return f"<<<\nplan_rationale :: {rationale}\ntotal_steps :: {total_steps}\n>>>"


def _toon_step(
    step_id: str, sub_query: str, depends_on: list[str] | None = None
) -> str:
    """Génère un bloc TOON pour une seule étape du plan."""
    deps = " | ".join(depends_on) if depends_on else ""
    return (
        f"<<<\n"
        f"step_id :: {step_id}\n"
        f"sub_query :: {sub_query}\n"
        f"depends_on :: {deps}\n"
        f">>>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Cas dégénéré SIMPLE : 0 appel LLM, 1 seule étape
# ─────────────────────────────────────────────────────────────────────────────


@patch("reasoning.planner.planner.completion")
def test_simple_query(mock_completion: MagicMock, planner: Planner) -> None:
    """Requête SIMPLE → plan mono-étape instancié en Python pur, sans appel LLM.

    Règle métier critique (planner_spec.md §2) :
    - 0 appel à litellm.completion
    - 1 seule PlanStep avec depends_on=[]
    - La sub_query de l'étape == la requête originale
    """
    query = "Qu'est-ce que la rétropropagation ?"
    analysis = _make_analysis(QueryType.SIMPLE, budget=1)

    result = planner.decompose(query=query, analysis=analysis)

    # LLM ne doit JAMAIS être appelé pour une requête SIMPLE
    mock_completion.assert_not_called()

    # Vérification de la structure du plan
    assert isinstance(result, ExecutionPlan)
    assert len(result.steps) == 1
    assert result.steps[0].step_id == "step_1"
    assert result.steps[0].sub_query == query
    assert result.steps[0].depends_on == []
    assert result.steps[0].status == StepStatus.PENDING
    assert result.original_query == query
    # Le graphe de dépendances doit aussi être cohérent
    assert result.dependencies_graph == {"step_1": []}


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Requête MULTI_HOP : plan séquentiel à 3 étapes avec dépendances
# ─────────────────────────────────────────────────────────────────────────────


@patch("reasoning.planner.planner.completion")
def test_multi_hop_query(mock_completion: MagicMock, planner: Planner) -> None:
    """Requête MULTI_HOP → LLM appelé, TOON parsé, dépendances séquentielles vérifiées.

    Scénario : Chaque étape dépend de la précédente (chaîne A → B → C).
    Vérification que le DAG est bien construit depuis la réponse TOON mockée.
    """
    query = "Quel est le rôle du fondateur de LangChain chez LangSmith ?"
    analysis = _make_analysis(
        QueryType.MULTI_HOP,
        budget=3,
        entities=["LangChain", "LangSmith"],
    )

    # Construction du payload TOON que le LLM "retourne"
    toon_response = "\n\n".join(
        [
            _toon_header("Décomposition séquentielle en 3 sauts.", 3),
            _toon_step("step_1", "Qui a fondé LangChain ?"),
            _toon_step("step_2", "Qu'est-ce que LangSmith ?", depends_on=["step_1"]),
            _toon_step(
                "step_3",
                "Quel est le rôle du fondateur de LangChain dans LangSmith ?",
                depends_on=["step_2"],
            ),
        ]
    )
    mock_completion.return_value = _make_llm_response(toon_response)

    result = planner.decompose(query=query, analysis=analysis)

    # LLM appelé exactement une fois
    mock_completion.assert_called_once()

    # Structure générale du plan
    assert isinstance(result, ExecutionPlan)
    assert len(result.steps) == 3

    # Vérification des dépendances séquentielles
    assert result.steps[0].step_id == "step_1"
    assert result.steps[0].depends_on == []  # step_1 : aucun prérequis

    assert result.steps[1].step_id == "step_2"
    assert result.steps[1].depends_on == ["step_1"]  # step_2 attend step_1

    assert result.steps[2].step_id == "step_3"
    assert result.steps[2].depends_on == ["step_2"]  # step_3 attend step_2

    # Cohérence du graphe
    assert result.dependencies_graph["step_1"] == []
    assert result.dependencies_graph["step_2"] == ["step_1"]
    assert result.dependencies_graph["step_3"] == ["step_2"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Détection de cycle : le fallback séquentiel doit être activé
# ─────────────────────────────────────────────────────────────────────────────


@patch("reasoning.planner.planner.completion")
def test_cycle_detection(mock_completion: MagicMock, planner: Planner) -> None:
    """LLM retourne un plan avec un cycle (A → B → A) → fallback séquentiel activé.

    Règle de sécurité (planner_spec.md §9) :
    - _validate_dag() détecte le cycle via Kahn → retourne False
    - decompose() bascule sur _build_sequential_fallback()
    - Le plan final doit être valide malgré la réponse LLM corrompue
    """
    query = "Quel est le lien entre BERT et GPT ?"
    analysis = _make_analysis(
        QueryType.MULTI_HOP,
        budget=2,
        entities=["BERT", "GPT"],
    )

    # Cycle délibéré : step_1 dépend de step_2 ET step_2 dépend de step_1
    toon_cyclic = "\n\n".join(
        [
            _toon_header("Plan invalide avec cycle.", 2),
            _toon_step("step_1", "Architecture de BERT ?", depends_on=["step_2"]),
            _toon_step("step_2", "Architecture de GPT ?", depends_on=["step_1"]),
        ]
    )
    mock_completion.return_value = _make_llm_response(toon_cyclic)

    result = planner.decompose(query=query, analysis=analysis)

    # Le résultat doit quand même être un ExecutionPlan valide
    assert isinstance(result, ExecutionPlan)
    assert len(result.steps) >= 1

    # Vérification que le plan de fallback est DAG-valide (pas de cycle)
    for step in result.steps:
        for dep in step.depends_on:
            # Chaque dépendance doit référencer un step_id existant
            known_ids = {s.step_id for s in result.steps}
            assert dep in known_ids, f"Dépendance inconnue : {dep}"

    # Le plan de fallback doit lui-même être acyclique (Kahn doit valider)
    assert planner._validate_dag(result) is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Troncature : LLM retourne plus d'étapes que le budget autorisé
# ─────────────────────────────────────────────────────────────────────────────


@patch("reasoning.planner.planner.completion")
def test_max_steps_truncation(mock_completion: MagicMock, planner: Planner) -> None:
    """LLM retourne 4 étapes mais le budget est de 2 → plan tronqué à 2 étapes.

    Règle de budget (planner_spec.md §3) :
    len(plan.steps) <= reasoning_budget, toujours, sans exception.
    """
    query = "Compare RAG, fine-tuning et in-context learning."
    # Budget volontairement restreint à 2
    analysis = _make_analysis(QueryType.COMPARATIVE, budget=2)

    # Le LLM "désobéit" et retourne 4 étapes au lieu de 2
    toon_response = "\n\n".join(
        [
            _toon_header("Décomposition comparative en 4 étapes.", 4),
            _toon_step("step_1", "Qu'est-ce que le RAG ?"),
            _toon_step("step_2", "Qu'est-ce que le fine-tuning ?"),
            _toon_step("step_3", "Qu'est-ce que le in-context learning ?"),
            _toon_step(
                "step_4",
                "Comparaison finale des trois approches.",
                depends_on=["step_1", "step_2", "step_3"],
            ),
        ]
    )
    mock_completion.return_value = _make_llm_response(toon_response)

    result = planner.decompose(query=query, analysis=analysis)

    assert isinstance(result, ExecutionPlan)

    # La règle du budget doit être strictement respectée
    assert (
        len(result.steps) <= 2
    ), f"Budget dépassé : {len(result.steps)} étapes retournées pour un budget de 2"

    # Les étapes conservées doivent être les premières (step_1, step_2)
    step_ids = [s.step_id for s in result.steps]
    assert "step_1" in step_ids
    assert "step_2" in step_ids
    # step_3 et step_4 doivent avoir été éliminées
    assert "step_3" not in step_ids
    assert "step_4" not in step_ids

    # Le DAG résultant (sur seulement 2 étapes) doit rester valide
    assert planner._validate_dag(result) is True
