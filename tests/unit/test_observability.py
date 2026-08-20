"""
Tests unitaires — Sprint I5-A : traçabilité et plafond d'appels LLM.

Couvre `reasoning.observability` (activation conditionnelle de Langfuse,
propagation de l'identifiant de trace, comptage des appels) et le plafond
`MAX_LLM_CALLS_PER_QUERY` appliqué par le nœud `critique`.

Aucun appel réseau, aucun appel LLM, aucune dépendance à Langfuse : les
fonctions instrumentées sont des doubles.

Exécution :
    uv run pytest tests/unit/test_observability.py -v
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reasoning import observability
from reasoning.contracts.action_interface import (
    RetrievalResponse,
    RetrievedChunk,
)
from reasoning.contracts.internal_models import (
    AgentState,
    AnalysisResult,
    CriticEvaluation,
    ExecutionPlan,
    PlanStep,
    QueryType,
)
from reasoning.graph.nodes import make_critique_node
from reasoning.graph.policy import (
    ROUTE_GENERATE_ANSWER,
    ROUTE_RETRIEVE,
    ReasoningPolicy,
)
from reasoning.graph.state import GraphState, build_initial_state


@pytest.fixture(autouse=True)
def clean_instrumentation() -> Any:
    """Garantit qu'aucun test ne laisse un module instrumenté derrière lui."""
    yield
    observability.uninstrument()


# ─────────────────────────────────────────────────────────────────────────────
# Comptage des appels
# ─────────────────────────────────────────────────────────────────────────────


class TestCallCounting:
    """Le compteur suit les appels d'UNE requête, à travers les tâches."""

    @staticmethod
    def _wrapped() -> Any:
        """Un `completion` factice, enveloppé par l'instrumentation."""
        return observability._wrap(lambda **_: "réponse")

    def test_no_counting_outside_a_request(self) -> None:
        """Hors `trace_query`, rien n'est compté : pas de fuite entre requêtes."""
        self._wrapped()()

        assert observability.llm_call_count() == 0

    def test_calls_are_counted_within_a_request(self) -> None:
        """Chaque appel incrémente le compteur de la requête courante."""
        call = self._wrapped()

        with observability.trace_query("q-1"):
            call()
            call()
            call()
            assert observability.llm_call_count() == 3

    def test_counter_is_reset_between_requests(self) -> None:
        """Chaque requête repart d'un budget neuf."""
        call = self._wrapped()

        with observability.trace_query("q-1"):
            call()
        with observability.trace_query("q-2"):
            assert observability.llm_call_count() == 0

    def test_calls_from_child_tasks_are_counted(self) -> None:
        """Les appels émis dans des tâches asyncio filles remontent au parent.

        C'est le cas réel : LangGraph exécute chaque nœud dans sa propre
        tâche. Un compteur entier stocké directement dans un `ContextVar`
        échouerait ici — la tâche fille travaillerait sur une COPIE du
        contexte et le parent lirait toujours 0.
        """
        call = self._wrapped()

        async def child() -> None:
            call()

        async def main() -> int:
            await asyncio.gather(*(asyncio.create_task(child()) for _ in range(5)))
            total: int = observability.llm_call_count()
            return total

        with observability.trace_query("q-async"):
            counted = asyncio.run(main())

        assert counted == 5

    def test_reset_keeps_the_trace_context(self) -> None:
        """Remettre le compteur à zéro ne change pas l'identifiant de trace."""
        call = self._wrapped()

        with observability.trace_query("q-1"):
            call()
            observability.reset_llm_call_count()

            assert observability.llm_call_count() == 0
            assert observability.current_trace_id() == "q-1"


# ─────────────────────────────────────────────────────────────────────────────
# Propagation de l'identifiant de trace
# ─────────────────────────────────────────────────────────────────────────────


class TestTracePropagation:
    """L'identifiant de requête est attaché à chaque appel LLM."""

    def test_metadata_carries_the_session_id(self) -> None:
        """`session_id` — la clé qui regroupe les appels dans Langfuse."""
        captured: list[dict[str, Any]] = []

        def fake(**kwargs: Any) -> str:
            captured.append(kwargs)
            return "ok"

        with observability.trace_query("plan-abc123"):
            observability._wrap(fake)(model="m", messages=[])

        metadata = captured[0]["metadata"]
        assert metadata["session_id"] == "plan-abc123"
        assert metadata["trace_id"] == "plan-abc123"

    def test_caller_metadata_is_preserved(self) -> None:
        """Un `metadata` déjà fourni par l'appelant n'est pas écrasé."""
        captured: list[dict[str, Any]] = []

        def fake(**kwargs: Any) -> str:
            captured.append(kwargs)
            return "ok"

        with observability.trace_query("plan-abc123"):
            observability._wrap(fake)(metadata={"session_id": "explicite", "x": 1})

        metadata = captured[0]["metadata"]
        assert metadata["session_id"] == "explicite"
        assert metadata["x"] == 1

    def test_no_metadata_added_outside_a_request(self) -> None:
        """Hors contexte, l'appel passe inchangé — aucune pollution."""
        captured: list[dict[str, Any]] = []

        def fake(**kwargs: Any) -> str:
            captured.append(kwargs)
            return "ok"

        observability._wrap(fake)(model="m")

        assert "metadata" not in captured[0]

    def test_trace_id_is_none_outside_a_request(self) -> None:
        """Aucun identifiant ne fuit hors d'un bloc `trace_query`."""
        assert observability.current_trace_id() is None


# ─────────────────────────────────────────────────────────────────────────────
# Activation conditionnelle — aucune dépendance dure
# ─────────────────────────────────────────────────────────────────────────────


class TestConditionalActivation:
    """Sans clés ni SDK, le pipeline fonctionne — Langfuse est facultatif."""

    def test_missing_keys_disable_langfuse(self, monkeypatch: Any) -> None:
        """Clés absentes → désactivation silencieuse, sans erreur."""
        monkeypatch.setattr(observability, "_langfuse_active", False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        assert observability.configure_langfuse() is False

    def test_partial_keys_disable_langfuse(self, monkeypatch: Any) -> None:
        """Une seule clé ne suffit pas."""
        monkeypatch.setattr(observability, "_langfuse_active", False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-quelque-chose")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        assert observability.configure_langfuse() is False

    def test_flush_is_a_no_op_when_inactive(self, monkeypatch: Any) -> None:
        """`flush()` ne lève jamais, même sans Langfuse."""
        monkeypatch.setattr(observability, "_langfuse_active", False)

        observability.flush()  # ne doit pas lever

    def test_counting_works_without_langfuse(self, monkeypatch: Any) -> None:
        """Le comptage ne dépend pas de Langfuse : c'est le plafond qui en vit."""
        monkeypatch.setattr(observability, "_langfuse_active", False)
        call = observability._wrap(lambda **_: "ok")

        with observability.trace_query("q-1"):
            call()

            assert observability.llm_call_count() == 1


class TestInstrumentation:
    """Remplacement et restauration du symbole `completion`."""

    def test_instrument_covers_every_llm_caller(self) -> None:
        """Les cinq modules qui appellent le LLM sont instrumentés."""
        instrumented = observability.instrument()

        assert set(instrumented) == set(observability._INSTRUMENTED_MODULES)

    @staticmethod
    def _completion_of(module: Any) -> Any:
        """Lit le symbole `completion` d'un module.

        Accès par `getattr` : `nodes` fait `from litellm import completion`,
        que mypy --strict refuse de considérer comme un ré-export explicite.
        """
        return getattr(module, observability._COMPLETION_ATTR)

    def test_instrument_is_idempotent(self) -> None:
        """Deux appels n'empilent pas deux couches d'enveloppes."""
        from reasoning.graph import nodes

        observability.instrument()
        first = self._completion_of(nodes)
        observability.instrument()

        assert self._completion_of(nodes) is first

    def test_uninstrument_restores_the_original(self) -> None:
        """La restauration rend exactement la fonction d'origine."""
        from reasoning.graph import nodes

        original = self._completion_of(nodes)
        observability.instrument()
        assert self._completion_of(nodes) is not original

        observability.uninstrument()

        assert self._completion_of(nodes) is original


# ─────────────────────────────────────────────────────────────────────────────
# Plafond d'appels LLM — nœud `critique`
# ─────────────────────────────────────────────────────────────────────────────


class FakeCritic:
    """Critic in-memory retournant un verdict figé."""

    max_retries = 2

    def __init__(self, is_sufficient: bool = False) -> None:
        self._is_sufficient = is_sufficient

    def evaluate(self, step: Any, response: Any) -> CriticEvaluation:
        return CriticEvaluation(
            step_id=step.step_id,
            is_sufficient=self._is_sufficient,
            relevance_score=0.9 if self._is_sufficient else 0.4,
            missing_aspects=[] if self._is_sufficient else ["un aspect"],
            feedback="" if self._is_sufficient else "Contexte insuffisant.",
        )


def _critique_state() -> GraphState:
    """État prêt pour le nœud `critique`, plan à deux étapes."""
    plan = ExecutionPlan(
        plan_id="plan-cap",
        original_query="Une question multi-hop",
        steps=[
            PlanStep(step_id="step_1", sub_query="Première sous-question ?"),
            PlanStep(step_id="step_2", sub_query="Seconde ?", depends_on=["step_1"]),
        ],
        dependencies_graph={"step_1": [], "step_2": ["step_1"]},
    )
    return GraphState(
        agent_state=AgentState(
            original_query=plan.original_query,
            analysis=AnalysisResult(
                query_type=QueryType.MULTI_HOP,
                confidence=0.9,
                detected_entities=[],
                reasoning_budget=3,
            ),
            plan=plan,
        ),
        retrieved_chunks=[],
        retry_counts={},
        pending_step_ids=["step_1", "step_2"],
        current_step_id="step_1",
        last_retrieval_response=RetrievalResponse(
            query_id=plan.plan_id,
            chunks=[
                RetrievedChunk(
                    chunk_id="1",
                    content="Un contenu quelconque.",
                    source="Article.txt",
                    relevance_score=0.8,
                )
            ],
            retrieval_score=0.8,
        ),
        answer=None,
        next_route="",
        step_answers={},
        llm_calls=0,
    )


class TestLlmCallCap:
    """Le plafond force une sortie propre, jamais une exception."""

    async def test_under_the_cap_routing_is_untouched(self) -> None:
        """Sous le plafond, la politique décide seule."""
        node = make_critique_node(FakeCritic(), ReasoningPolicy())

        with (
            patch("reasoning.graph.nodes._MAX_LLM_CALLS_PER_QUERY", 100),
            patch("reasoning.graph.nodes._llm_call_count", return_value=3),
        ):
            result = await node(_critique_state())

        assert result["next_route"] == ROUTE_RETRIEVE
        assert result["llm_calls"] == 3

    async def test_reaching_the_cap_exits_to_generate_answer(self) -> None:
        """Plafond atteint → sortie vers `generate_answer`, sans exception."""
        node = make_critique_node(FakeCritic(), ReasoningPolicy())

        with (
            patch("reasoning.graph.nodes._MAX_LLM_CALLS_PER_QUERY", 5),
            patch("reasoning.graph.nodes._llm_call_count", return_value=5),
        ):
            result = await node(_critique_state())

        assert result["next_route"] == ROUTE_GENERATE_ANSWER
        assert result["llm_calls"] == 5

    async def test_cap_removes_the_current_step_from_the_queue(self) -> None:
        """L'étape courante est retirée : `generate_answer` ne boucle pas."""
        node = make_critique_node(FakeCritic(), ReasoningPolicy())

        with (
            patch("reasoning.graph.nodes._MAX_LLM_CALLS_PER_QUERY", 5),
            patch("reasoning.graph.nodes._llm_call_count", return_value=9),
        ):
            result = await node(_critique_state())

        assert "step_1" not in result["pending_step_ids"]

    async def test_cap_logs_a_warning(self, caplog: Any) -> None:
        """Le dépassement est journalisé en WARNING, pas en silence."""
        node = make_critique_node(FakeCritic(), ReasoningPolicy())

        with (
            patch("reasoning.graph.nodes._MAX_LLM_CALLS_PER_QUERY", 5),
            patch("reasoning.graph.nodes._llm_call_count", return_value=5),
            caplog.at_level("WARNING", logger="reasoning.graph.nodes"),
        ):
            await node(_critique_state())

        assert any(
            "plafond d'appels LLM atteint" in r.message for r in caplog.records
        ), caplog.text

    async def test_zero_disables_the_cap(self) -> None:
        """`MAX_LLM_CALLS_PER_QUERY=0` désactive le plafond."""
        node = make_critique_node(FakeCritic(), ReasoningPolicy())

        with (
            patch("reasoning.graph.nodes._MAX_LLM_CALLS_PER_QUERY", 0),
            patch("reasoning.graph.nodes._llm_call_count", return_value=9999),
        ):
            result = await node(_critique_state())

        assert result["next_route"] == ROUTE_RETRIEVE

    async def test_no_instrumentation_never_throttles(self) -> None:
        """Sans instrumentation le compteur vaut 0 : le pipeline n'est pas bridé."""
        node = make_critique_node(FakeCritic(), ReasoningPolicy())

        with patch("reasoning.graph.nodes._MAX_LLM_CALLS_PER_QUERY", 1):
            result = await node(_critique_state())

        assert result["next_route"] == ROUTE_RETRIEVE
        assert result["llm_calls"] == 0


class TestInitialState:
    """L'état initial expose le compteur, à zéro."""

    def test_llm_calls_starts_at_zero(self) -> None:
        assert build_initial_state("Une question")["llm_calls"] == 0


class TestNodesReadTheSharedCounter:
    """`nodes._llm_call_count` délègue bien au module d'observabilité."""

    def test_counter_is_read_from_observability(self) -> None:
        from reasoning.graph import nodes

        with observability.trace_query("q-partage"):
            observability._wrap(MagicMock(return_value="ok"))()

            assert nodes._llm_call_count() == 1
