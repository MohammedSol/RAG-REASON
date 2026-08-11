"""
Tests unitaires — Sprint 6.1/6.3 : ReasoningPolicy (logique pure, sans LLM).

`ReasoningPolicy` ne dépend ni de LangGraph, ni d'Ollama, ni d'aucun
composant du module REASONING — ces tests s'exécutent donc en isolation
complète et instantanément.

Exécution :
    uv run pytest tests/unit/test_policy.py -v --cov=reasoning.graph.policy
"""

from __future__ import annotations

from reasoning.graph.policy import (
    ROUTE_CLARIFY,
    ROUTE_END,
    ROUTE_GENERATE_ANSWER,
    ROUTE_PLAN,
    ROUTE_RETRIEVE,
    ReasoningPolicy,
)


class TestRouteAfterAnalysis:
    """Routage post-`analyze_query` : AMBIGUOUS (budget=0) → clarify."""

    def test_zero_budget_routes_to_clarify(self) -> None:
        policy = ReasoningPolicy()
        assert policy.route_after_analysis(reasoning_budget=0) == ROUTE_CLARIFY

    def test_simple_budget_routes_to_plan(self) -> None:
        policy = ReasoningPolicy()
        assert policy.route_after_analysis(reasoning_budget=1) == ROUTE_PLAN

    def test_multi_hop_budget_routes_to_plan(self) -> None:
        policy = ReasoningPolicy()
        assert policy.route_after_analysis(reasoning_budget=3) == ROUTE_PLAN

    def test_comparative_budget_routes_to_plan(self) -> None:
        policy = ReasoningPolicy()
        assert policy.route_after_analysis(reasoning_budget=2) == ROUTE_PLAN


class TestRouteAfterCritique:
    """Routage post-`critique` : garde globale > verdict > garde locale."""

    def test_sufficient_context_with_next_step_retrieves_next(self) -> None:
        policy = ReasoningPolicy()
        decision = policy.route_after_critique(
            reasoning_budget=3,
            feedback_loop_count=1,
            is_sufficient=True,
            retry_count=0,
            max_retries=2,
            has_next_step=True,
        )
        assert decision.route == ROUTE_RETRIEVE
        assert decision.advance_step is True

    def test_sufficient_context_no_next_step_generates_answer(self) -> None:
        policy = ReasoningPolicy()
        decision = policy.route_after_critique(
            reasoning_budget=3,
            feedback_loop_count=1,
            is_sufficient=True,
            retry_count=0,
            max_retries=2,
            has_next_step=False,
        )
        assert decision.route == ROUTE_GENERATE_ANSWER
        assert decision.advance_step is True

    def test_insufficient_context_under_retry_limit_retries_same_step(self) -> None:
        policy = ReasoningPolicy()
        decision = policy.route_after_critique(
            reasoning_budget=3,
            feedback_loop_count=1,
            is_sufficient=False,
            retry_count=0,
            max_retries=2,
            has_next_step=False,
        )
        assert decision.route == ROUTE_RETRIEVE
        assert decision.advance_step is False

    def test_insufficient_context_retries_exhausted_advances(self) -> None:
        policy = ReasoningPolicy()
        decision = policy.route_after_critique(
            reasoning_budget=5,
            feedback_loop_count=2,
            is_sufficient=False,
            retry_count=2,
            max_retries=2,
            has_next_step=True,
        )
        assert decision.route == ROUTE_RETRIEVE
        assert decision.advance_step is True

    def test_insufficient_context_retries_exhausted_no_next_step_generates(
        self,
    ) -> None:
        policy = ReasoningPolicy()
        decision = policy.route_after_critique(
            reasoning_budget=5,
            feedback_loop_count=2,
            is_sufficient=False,
            retry_count=2,
            max_retries=2,
            has_next_step=False,
        )
        assert decision.route == ROUTE_GENERATE_ANSWER
        assert decision.advance_step is True

    def test_global_budget_exhausted_forces_generate_answer_even_if_sufficient(
        self,
    ) -> None:
        """La garde globale est prioritaire — même si le contexte est suffisant."""
        policy = ReasoningPolicy()
        decision = policy.route_after_critique(
            reasoning_budget=2,
            feedback_loop_count=2,
            is_sufficient=True,
            retry_count=0,
            max_retries=2,
            has_next_step=True,
        )
        assert decision.route == ROUTE_GENERATE_ANSWER
        assert decision.advance_step is True

    def test_global_budget_exhausted_forces_generate_answer_with_retries_left(
        self,
    ) -> None:
        """La garde globale est prioritaire sur la garde locale (retries restants)."""
        policy = ReasoningPolicy()
        decision = policy.route_after_critique(
            reasoning_budget=1,
            feedback_loop_count=1,
            is_sufficient=False,
            retry_count=0,
            max_retries=2,
            has_next_step=True,
        )
        assert decision.route == ROUTE_GENERATE_ANSWER
        assert decision.advance_step is True

    def test_budget_not_yet_exhausted_at_exact_boundary_minus_one(self) -> None:
        """feedback_loop_count juste sous le budget → pas de garde globale déclenchée."""
        policy = ReasoningPolicy()
        decision = policy.route_after_critique(
            reasoning_budget=3,
            feedback_loop_count=2,
            is_sufficient=False,
            retry_count=0,
            max_retries=2,
            has_next_step=False,
        )
        assert decision.route == ROUTE_RETRIEVE
        assert decision.advance_step is False


class TestRouteAfterVerification:
    """Routage post-`verify` : is_grounded > garde globale > re-planification."""

    def test_grounded_answer_routes_to_end(self) -> None:
        policy = ReasoningPolicy()
        route = policy.route_after_verification(
            reasoning_budget=3, feedback_loop_count=2, is_grounded=True
        )
        assert route == ROUTE_END

    def test_ungrounded_answer_with_budget_left_replans(self) -> None:
        policy = ReasoningPolicy()
        route = policy.route_after_verification(
            reasoning_budget=4, feedback_loop_count=2, is_grounded=False
        )
        assert route == ROUTE_PLAN

    def test_ungrounded_answer_with_budget_exhausted_routes_to_end(self) -> None:
        """Garde globale : budget épuisé → sortie forcée même si non fondé."""
        policy = ReasoningPolicy()
        route = policy.route_after_verification(
            reasoning_budget=2, feedback_loop_count=2, is_grounded=False
        )
        assert route == ROUTE_END

    def test_grounded_answer_routes_to_end_even_if_budget_exhausted(self) -> None:
        """is_grounded=True est prioritaire — pas besoin d'être sous le budget."""
        policy = ReasoningPolicy()
        route = policy.route_after_verification(
            reasoning_budget=1, feedback_loop_count=5, is_grounded=True
        )
        assert route == ROUTE_END
