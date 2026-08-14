"""
Tests d'intégration — Sprint 6.3 : graphe LangGraph complet (module REASONING).

Ces tests exercent l'INTÉGRATION de tous les composants (Analyzer, Planner,
Critic, Verifier, ActionClient) assemblés par `build_graph()`, en injectant
des doubles de test (`Fake*`) pour chaque composant — conformément à
docs/graph_spec.md §7 : le module ACTION n'étant pas encore branché, aucun
test ne suppose une API distante joignable, et aucun appel Ollama réel n'est
nécessaire (seul le nœud `generate_answer` appelle directement LiteLLM en
interne — il est mocké via `unittest.mock.patch`).

Décision de classification (documentée dans le rapport final) : ces tests ne
portent PAS le marqueur `@pytest.mark.integration` du projet, celui-ci étant
réservé aux tests nécessitant Ollama en local (cf. pyproject.toml). Ici,
« intégration » signifie assemblage multi-composants du graphe, pas
dépendance à un LLM réel — ces tests s'exécutent donc dans toute suite
`pytest tests/`, y compris `-m "not integration"`.

Exécution :
    uv run pytest tests/integration/test_graph.py -v
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import reasoning.graph.policy as policy_module
from reasoning.contracts.action_interface import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from reasoning.contracts.internal_models import (
    AnalysisResult,
    CriticEvaluation,
    ExecutionPlan,
    PlanStep,
    QueryType,
    VerificationResult,
)
from reasoning.graph.graph import build_graph
from reasoning.graph.state import build_initial_state

# ─────────────────────────────────────────────────────────────────────────────
# Doubles de test (Fakes) — un par composant injectable de build_graph()
# ─────────────────────────────────────────────────────────────────────────────


class FakeAnalyzer:
    """Double de QueryAnalyzer : retourne un AnalysisResult fixe."""

    def __init__(self, result: AnalysisResult) -> None:
        self._result = result
        self.calls = 0

    def analyze(self, query: str) -> AnalysisResult:
        self.calls += 1
        return self._result


class FakePlanner:
    """Double de Planner : retourne un ExecutionPlan fixe, journalise les requêtes reçues."""

    def __init__(self, plan: ExecutionPlan) -> None:
        self._plan = plan
        self.calls = 0
        self.received_queries: list[str] = []

    def decompose(self, query: str, analysis: AnalysisResult) -> ExecutionPlan:
        self.calls += 1
        self.received_queries.append(query)
        return self._plan


class FakeCritic:
    """Double de Critic : consomme une séquence d'évaluations, une par appel."""

    def __init__(
        self, evaluations: list[CriticEvaluation], max_retries: int = 2
    ) -> None:
        self._evaluations = evaluations
        self.max_retries = max_retries
        self.calls = 0

    def evaluate(self, step: PlanStep, response: RetrievalResponse) -> CriticEvaluation:
        idx = min(self.calls, len(self._evaluations) - 1)
        self.calls += 1
        return self._evaluations[idx]


class FakeVerifier:
    """Double de Verifier : consomme une séquence de verdicts, un par appel."""

    def __init__(self, results: list[VerificationResult]) -> None:
        self._results = results
        self.calls = 0

    def verify(self, answer: str, sources: list[RetrievedChunk]) -> VerificationResult:
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


class FakeRetrievalClient:
    """Double de ActionClient — in-memory, aucune dépendance réseau."""

    def __init__(self, response: RetrievalResponse) -> None:
        self._response = response
        self.calls = 0
        self.received_requests: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        self.calls += 1
        self.received_requests.append(request)
        return self._response


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — données réalistes
# ─────────────────────────────────────────────────────────────────────────────


def _retrieval_response(query_id: str) -> RetrievalResponse:
    return RetrievalResponse(
        query_id=query_id,
        chunks=[
            RetrievedChunk(
                chunk_id="openai-history-001",
                content="OpenAI was founded in December 2015 by Sam Altman and others.",
                source="openai_history.pdf",
                relevance_score=0.91,
            )
        ],
        retrieval_score=0.91,
    )


def _mock_generation_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = text
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Parcours SIMPLE complet
# ─────────────────────────────────────────────────────────────────────────────


class TestSimpleQueryFullPath:
    """analyze → plan → retrieve → critique → generate_answer → verify → END."""

    @pytest.mark.asyncio
    @patch("reasoning.graph.nodes.completion")
    async def test_simple_query_reaches_grounded_end(
        self, mock_completion: MagicMock
    ) -> None:
        analysis = AnalysisResult(
            query_type=QueryType.SIMPLE,
            confidence=0.95,
            detected_entities=["OpenAI"],
            reasoning_budget=1,
        )
        plan = ExecutionPlan(
            plan_id="plan-simple-001",
            original_query="When was OpenAI founded?",
            steps=[PlanStep(step_id="step_1", sub_query="When was OpenAI founded?")],
            dependencies_graph={"step_1": []},
        )

        analyzer = FakeAnalyzer(analysis)
        planner = FakePlanner(plan)
        critic = FakeCritic(
            [
                CriticEvaluation(
                    step_id="step_1",
                    is_sufficient=True,
                    relevance_score=0.9,
                    missing_aspects=[],
                    feedback="",
                )
            ]
        )
        verifier = FakeVerifier(
            [
                VerificationResult(
                    is_grounded=True,
                    faithfulness_score=1.0,
                    unsupported_claims=[],
                    final_answer="OpenAI was founded in December 2015.",
                )
            ]
        )
        retrieval_client = FakeRetrievalClient(_retrieval_response("plan-simple-001"))
        mock_completion.return_value = _mock_generation_response(
            "OpenAI was founded in December 2015."
        )

        graph = build_graph(
            analyzer=analyzer,
            planner=planner,
            critic=critic,
            verifier=verifier,
            retrieval_client=retrieval_client,
        )
        final_state: dict[str, Any] = await graph.ainvoke(
            build_initial_state("When was OpenAI founded?")
        )

        result = final_state["agent_state"].verification
        assert result.is_grounded is True
        assert analyzer.calls == 1
        assert planner.calls == 1
        assert retrieval_client.calls == 1
        assert critic.calls == 1
        assert verifier.calls == 1
        # 1 passage critique + 1 passage verify = 2
        assert final_state["agent_state"].feedback_loop_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Parcours MULTI_HOP : critique échoue au 1er tour, réussit au 2e
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiHopTwoSteps:
    """2 étapes ; critique insuffisant sur step_1 puis suffisant, puis step_2."""

    @pytest.mark.asyncio
    @patch("reasoning.graph.nodes.completion")
    async def test_retry_then_advance_through_two_steps(
        self, mock_completion: MagicMock
    ) -> None:
        analysis = AnalysisResult(
            query_type=QueryType.MULTI_HOP,
            confidence=0.88,
            detected_entities=["OpenAI", "GPT-4"],
            reasoning_budget=5,
        )
        plan = ExecutionPlan(
            plan_id="plan-multihop-001",
            original_query="Who founded the company that created GPT-4?",
            steps=[
                PlanStep(step_id="step_1", sub_query="Which company created GPT-4?"),
                PlanStep(
                    step_id="step_2",
                    sub_query="Who founded that company?",
                    depends_on=["step_1"],
                ),
            ],
            dependencies_graph={"step_1": [], "step_2": ["step_1"]},
        )

        analyzer = FakeAnalyzer(analysis)
        planner = FakePlanner(plan)
        critic = FakeCritic(
            [
                CriticEvaluation(
                    step_id="step_1",
                    is_sufficient=False,
                    relevance_score=0.4,
                    missing_aspects=["company name"],
                    feedback="Context does not clearly name the company.",
                ),
                CriticEvaluation(
                    step_id="step_1",
                    is_sufficient=True,
                    relevance_score=0.9,
                    missing_aspects=[],
                    feedback="",
                ),
                CriticEvaluation(
                    step_id="step_2",
                    is_sufficient=True,
                    relevance_score=0.92,
                    missing_aspects=[],
                    feedback="",
                ),
            ]
        )
        verifier = FakeVerifier(
            [
                VerificationResult(
                    is_grounded=True,
                    faithfulness_score=1.0,
                    unsupported_claims=[],
                    final_answer="Sam Altman co-founded OpenAI, creator of GPT-4.",
                )
            ]
        )
        retrieval_client = FakeRetrievalClient(_retrieval_response("plan-multihop-001"))
        mock_completion.return_value = _mock_generation_response(
            "Sam Altman co-founded OpenAI, creator of GPT-4."
        )

        graph = build_graph(
            analyzer=analyzer,
            planner=planner,
            critic=critic,
            verifier=verifier,
            retrieval_client=retrieval_client,
        )
        final_state: dict[str, Any] = await graph.ainvoke(
            build_initial_state("Who founded the company that created GPT-4?")
        )

        assert critic.calls == 3
        assert retrieval_client.calls == 3
        assert verifier.calls == 1
        assert planner.calls == 1
        result = final_state["agent_state"].verification
        assert result.is_grounded is True
        # 3 passages critique + 1 passage verify = 4
        assert final_state["agent_state"].feedback_loop_count == 4


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Boucle de re-planification après échec de verify
# ─────────────────────────────────────────────────────────────────────────────


class TestReplanningLoop:
    """verify échoue une première fois → retour à plan → succès au 2e tour."""

    @pytest.mark.asyncio
    @patch("reasoning.graph.nodes.completion")
    async def test_verify_failure_triggers_replanning_with_feedback(
        self, mock_completion: MagicMock
    ) -> None:
        analysis = AnalysisResult(
            query_type=QueryType.MULTI_HOP,
            confidence=0.85,
            detected_entities=["GPT-4"],
            reasoning_budget=4,
        )
        plan = ExecutionPlan(
            plan_id="plan-replan-001",
            original_query="What is GPT-4's parameter count?",
            steps=[
                PlanStep(step_id="step_1", sub_query="What is GPT-4's parameter count?")
            ],
            dependencies_graph={"step_1": []},
        )

        analyzer = FakeAnalyzer(analysis)
        planner = FakePlanner(plan)
        critic = FakeCritic(
            [
                CriticEvaluation(
                    step_id="step_1",
                    is_sufficient=True,
                    relevance_score=0.85,
                    missing_aspects=[],
                    feedback="",
                )
            ]
        )
        verifier = FakeVerifier(
            [
                VerificationResult(
                    is_grounded=False,
                    faithfulness_score=0.3,
                    unsupported_claims=["GPT-4 has 1 trillion parameters"],
                    final_answer="GPT-4 has 1 trillion parameters.",
                ),
                VerificationResult(
                    is_grounded=True,
                    faithfulness_score=1.0,
                    unsupported_claims=[],
                    final_answer="GPT-4's exact parameter count is undisclosed by OpenAI.",
                ),
            ]
        )
        retrieval_client = FakeRetrievalClient(_retrieval_response("plan-replan-001"))
        mock_completion.return_value = _mock_generation_response(
            "GPT-4's exact parameter count is undisclosed by OpenAI."
        )

        graph = build_graph(
            analyzer=analyzer,
            planner=planner,
            critic=critic,
            verifier=verifier,
            retrieval_client=retrieval_client,
        )
        final_state: dict[str, Any] = await graph.ainvoke(
            build_initial_state("What is GPT-4's parameter count?")
        )

        assert planner.calls == 2
        assert verifier.calls == 2
        assert critic.calls == 2
        # La re-planification doit injecter le feedback du Verifier dans la requête
        assert "1 trillion parameters" in planner.received_queries[1]
        result = final_state["agent_state"].verification
        assert result.is_grounded is True
        # 2x(critique + verify) = 4, exactement au plafond du budget
        assert final_state["agent_state"].feedback_loop_count == 4


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Garde anti-boucle globale (point d'attention critique de la mission)
# ─────────────────────────────────────────────────────────────────────────────


class TestBudgetExhaustionGuard:
    """Critic systématiquement insatisfait — le graphe DOIT terminer, pas boucler."""

    @pytest.mark.asyncio
    @patch("reasoning.graph.nodes.completion")
    async def test_budget_exhaustion_forces_exit_without_infinite_loop(
        self, mock_completion: MagicMock
    ) -> None:
        analysis = AnalysisResult(
            query_type=QueryType.SIMPLE,
            confidence=0.9,
            detected_entities=[],
            reasoning_budget=1,
        )
        plan = ExecutionPlan(
            plan_id="plan-budget-001",
            original_query="Impossible question with no good source.",
            steps=[
                PlanStep(
                    step_id="step_1",
                    sub_query="Impossible question with no good source.",
                )
            ],
            dependencies_graph={"step_1": []},
        )

        analyzer = FakeAnalyzer(analysis)
        planner = FakePlanner(plan)
        # Toujours insuffisant, quel que soit le nombre d'appels.
        always_insufficient = CriticEvaluation(
            step_id="step_1",
            is_sufficient=False,
            relevance_score=0.1,
            missing_aspects=["everything"],
            feedback="No relevant context found.",
        )
        critic = FakeCritic([always_insufficient], max_retries=2)
        # Toujours non-fondé, quel que soit le nombre d'appels.
        always_ungrounded = VerificationResult(
            is_grounded=False,
            faithfulness_score=0.0,
            unsupported_claims=["everything"],
            final_answer="I don't know.",
        )
        verifier = FakeVerifier([always_ungrounded])
        retrieval_client = FakeRetrievalClient(_retrieval_response("plan-budget-001"))
        mock_completion.return_value = _mock_generation_response("I don't know.")

        graph = build_graph(
            analyzer=analyzer,
            planner=planner,
            critic=critic,
            verifier=verifier,
            retrieval_client=retrieval_client,
        )
        # Si la garde anti-boucle est défaillante, ce await ne retournera jamais
        # et le test dépassera le timeout par défaut de pytest-asyncio.
        final_state: dict[str, Any] = await graph.ainvoke(
            build_initial_state("Impossible question with no good source.")
        )

        # reasoning_budget=1 : critique(1) atteint déjà le budget → generate_answer
        # direct (garde globale prioritaire, malgré is_sufficient=False). Puis
        # verify(1) porte le compteur à 2. Budget déjà dépassé → END forcé (pas
        # de re-planification), même si le résultat n'est pas fondé.
        assert critic.calls == 1
        assert retrieval_client.calls == 1
        assert verifier.calls == 1
        assert planner.calls == 1
        assert final_state["agent_state"].feedback_loop_count == 2
        result = final_state["agent_state"].verification
        assert result.is_grounded is False

    @pytest.mark.asyncio
    @patch("reasoning.graph.nodes.completion")
    async def test_local_retry_guard_exhausted_before_global_budget(
        self, mock_completion: MagicMock
    ) -> None:
        """max_retries épuisé alors qu'il reste du budget global → avance quand même."""
        analysis = AnalysisResult(
            query_type=QueryType.SIMPLE,
            confidence=0.9,
            detected_entities=[],
            reasoning_budget=10,
        )
        plan = ExecutionPlan(
            plan_id="plan-retry-001",
            original_query="A question the retriever cannot answer.",
            steps=[
                PlanStep(
                    step_id="step_1",
                    sub_query="A question the retriever cannot answer.",
                )
            ],
            dependencies_graph={"step_1": []},
        )

        analyzer = FakeAnalyzer(analysis)
        planner = FakePlanner(plan)
        always_insufficient = CriticEvaluation(
            step_id="step_1",
            is_sufficient=False,
            relevance_score=0.2,
            missing_aspects=["everything"],
            feedback="No relevant context found.",
        )
        # max_retries=1 : 1er appel → retry (retry_count 0 < 1), 2e appel →
        # retries épuisés (retry_count 1 >= 1) → avance vers generate_answer.
        critic = FakeCritic([always_insufficient], max_retries=1)
        grounded = VerificationResult(
            is_grounded=True,
            faithfulness_score=1.0,
            unsupported_claims=[],
            final_answer="Best-effort answer with partial context.",
        )
        verifier = FakeVerifier([grounded])
        retrieval_client = FakeRetrievalClient(_retrieval_response("plan-retry-001"))
        mock_completion.return_value = _mock_generation_response(
            "Best-effort answer with partial context."
        )

        graph = build_graph(
            analyzer=analyzer,
            planner=planner,
            critic=critic,
            verifier=verifier,
            retrieval_client=retrieval_client,
        )
        final_state: dict[str, Any] = await graph.ainvoke(
            build_initial_state("A question the retriever cannot answer.")
        )

        # Garde locale (max_retries=1) déclenchée bien avant le budget global (10) :
        # exactement 2 appels critique (1 retry + 1 abandon), pas 10.
        assert critic.calls == 2
        assert retrieval_client.calls == 2
        # 2 passages critique + 1 passage verify = 3 (bien en-deçà du budget=10,
        # preuve que c'est la garde LOCALE qui a stoppé la boucle, pas la globale).
        assert final_state["agent_state"].feedback_loop_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Requête AMBIGUOUS : court-circuit sans retrieval
# ─────────────────────────────────────────────────────────────────────────────


class TestAmbiguousQueryClarification:
    """reasoning_budget=0 → clarify → END, aucun appel Planner/Critic/Verifier."""

    @pytest.mark.asyncio
    async def test_ambiguous_query_skips_retrieval_entirely(self) -> None:
        analysis = AnalysisResult(
            query_type=QueryType.AMBIGUOUS,
            confidence=0.8,
            detected_entities=["python"],
            reasoning_budget=0,
        )
        # Plan/Critic/Verifier ne devraient jamais être appelés — plans/évaluations
        # volontairement vides pour détecter tout appel inattendu (IndexError).
        analyzer = FakeAnalyzer(analysis)
        planner = FakePlanner(
            ExecutionPlan(plan_id="unused", original_query="unused", steps=[])
        )
        critic = FakeCritic([])
        verifier = FakeVerifier([])
        retrieval_client = FakeRetrievalClient(_retrieval_response("unused"))

        graph = build_graph(
            analyzer=analyzer,
            planner=planner,
            critic=critic,
            verifier=verifier,
            retrieval_client=retrieval_client,
        )
        final_state: dict[str, Any] = await graph.ainvoke(
            build_initial_state("How does Python work?")
        )

        assert planner.calls == 0
        assert critic.calls == 0
        assert verifier.calls == 0
        assert retrieval_client.calls == 0
        result = final_state["agent_state"].verification
        assert result.is_grounded is False
        assert result.unsupported_claims == ["ambiguous_query"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Client de retrieval injectable (double en mémoire)
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrievalClientInjectable:
    """Le nœud retrieve fonctionne avec un double, sans dépendance réseau réelle."""

    @pytest.mark.asyncio
    @patch("reasoning.graph.nodes.completion")
    async def test_retrieve_node_builds_well_formed_request(
        self, mock_completion: MagicMock
    ) -> None:
        analysis = AnalysisResult(
            query_type=QueryType.SIMPLE,
            confidence=0.95,
            detected_entities=[],
            reasoning_budget=1,
        )
        plan = ExecutionPlan(
            plan_id="plan-req-001",
            original_query="What is BERT?",
            steps=[PlanStep(step_id="step_1", sub_query="What is BERT?")],
            dependencies_graph={"step_1": []},
        )
        analyzer = FakeAnalyzer(analysis)
        planner = FakePlanner(plan)
        critic = FakeCritic(
            [
                CriticEvaluation(
                    step_id="step_1",
                    is_sufficient=True,
                    relevance_score=0.9,
                    missing_aspects=[],
                    feedback="",
                )
            ]
        )
        verifier = FakeVerifier(
            [
                VerificationResult(
                    is_grounded=True,
                    faithfulness_score=1.0,
                    unsupported_claims=[],
                    final_answer="BERT is a transformer-based language model.",
                )
            ]
        )
        retrieval_client = FakeRetrievalClient(_retrieval_response("plan-req-001"))
        mock_completion.return_value = _mock_generation_response(
            "BERT is a transformer-based language model."
        )

        graph = build_graph(
            analyzer=analyzer,
            planner=planner,
            critic=critic,
            verifier=verifier,
            retrieval_client=retrieval_client,
        )
        await graph.ainvoke(build_initial_state("What is BERT?"))

        assert len(retrieval_client.received_requests) == 1
        sent_request = retrieval_client.received_requests[0]
        assert sent_request.query_id == "plan-req-001"
        assert sent_request.sub_query == "What is BERT?"
        assert sent_request.hop_index == 0
        assert sent_request.top_k > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test — Contrainte d'architecture : policy.py sans import LangGraph
# ─────────────────────────────────────────────────────────────────────────────


def test_policy_module_has_no_orchestration_framework_import() -> None:
    """ReasoningPolicy reste framework-free (contrainte non négociable de la mission)."""
    policy_path = Path(policy_module.__file__ or "")
    source = policy_path.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+langgraph", source, re.MULTILINE)
