"""
Tests unitaires — Lot B : transmission de l'entité résolue entre étapes (§3).

Le Planner produit toutes les sous-requêtes en une passe, avant exécution :
une sous-requête dépendante désigne donc sa cible par une périphrase (« the
identified woman ») que le moteur lexical ne peut pas résoudre. Le graphe ne
produisant aucune réponse intermédiaire, l'entité doit être extraite des
chunks de l'étape dont on dépend.

Correction validée : le nœud `critique` synthétise une réponse d'une phrase
quand une étape est SUFFISANTE et qu'une autre en DÉPEND ; le nœud `retrieve`
la concatène à la sous-requête des étapes dépendantes.

Aucun appel réseau, aucun appel LLM réel : `completion` est mocké et le client
de retrieval est un espion in-memory.

Exécution :
    uv run pytest tests/unit/test_step_answers.py -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from reasoning.contracts.action_interface import (
    RetrievalRequest,
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
from reasoning.graph.nodes import (
    build_dependent_sub_query,
    has_dependents,
    make_critique_node,
    make_retrieve_node,
)
from reasoning.graph.policy import ReasoningPolicy
from reasoning.graph.state import GraphState, build_initial_state

# ─────────────────────────────────────────────────────────────────────────────
# Données de référence — trace réelle de la question Corliss Archer
# ─────────────────────────────────────────────────────────────────────────────

STEP_1_QUERY = "Who portrayed Corliss Archer in the film Kiss and Tell?"
STEP_2_QUERY = "What government position did the identified woman hold?"
RESOLVED = "Shirley Temple portrayed Corliss Archer in Kiss and Tell."

# Chunk réellement retourné par le moteur, verbatim : il contient l'entité pont.
STEP_1_CHUNKS = [
    RetrievedChunk(
        chunk_id="1675",
        content=(
            "Kiss and Tell (1945 film) Kiss and Tell is a 1945 American comedy "
            "film starring then 17-year-old Shirley Temple as Corliss Archer."
        ),
        source="Kiss_and_Tell_(1945_film).txt",
        relevance_score=0.90,
    )
]


def _plan() -> ExecutionPlan:
    """Plan bridge à deux étapes, `step_2` dépendant de `step_1`."""
    return ExecutionPlan(
        plan_id="plan-lot-b",
        original_query=(
            "What government position was held by the woman who portrayed "
            "Corliss Archer in the film Kiss and Tell?"
        ),
        steps=[
            PlanStep(step_id="step_1", sub_query=STEP_1_QUERY),
            PlanStep(step_id="step_2", sub_query=STEP_2_QUERY, depends_on=["step_1"]),
        ],
        dependencies_graph={"step_1": [], "step_2": ["step_1"]},
    )


def _state(
    *,
    pending: list[str],
    step_answers: dict[str, str] | None = None,
    retry_counts: dict[str, int] | None = None,
    evaluations: list[CriticEvaluation] | None = None,
    last_response: RetrievalResponse | None = None,
    current_step_id: str | None = None,
) -> GraphState:
    """GraphState complet pour l'exécution d'un nœud."""
    plan = _plan()
    state = GraphState(
        agent_state=AgentState(
            original_query=plan.original_query,
            analysis=AnalysisResult(
                query_type=QueryType.MULTI_HOP,
                confidence=0.86,
                detected_entities=[],
                reasoning_budget=3,
            ),
            plan=plan,
            evaluations=[] if evaluations is None else evaluations,
        ),
        retrieved_chunks=[],
        retry_counts={} if retry_counts is None else retry_counts,
        pending_step_ids=pending,
        current_step_id=current_step_id,
        last_retrieval_response=last_response,
        answer=None,
        next_route="",
        step_answers={} if step_answers is None else step_answers,
    )
    return state


class SpyClient:
    """Client de retrieval in-memory : capture les requêtes émises."""

    def __init__(self) -> None:
        self.requests: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        self.requests.append(request)
        return RetrievalResponse(
            query_id=request.query_id, chunks=list(STEP_1_CHUNKS), retrieval_score=0.9
        )


def _llm_response(text: str) -> MagicMock:
    """Fausse réponse LiteLLM portant `text`."""
    mock = MagicMock()
    mock.choices[0].message.content = text
    return mock


class FakeCritic:
    """Critic in-memory retournant un verdict figé."""

    max_retries = 2

    def __init__(self, is_sufficient: bool) -> None:
        self._is_sufficient = is_sufficient
        self.calls = 0

    def evaluate(self, step: Any, response: Any) -> CriticEvaluation:
        self.calls += 1
        return CriticEvaluation(
            step_id=step.step_id,
            is_sufficient=self._is_sufficient,
            relevance_score=0.9 if self._is_sufficient else 0.4,
            missing_aspects=[] if self._is_sufficient else ["l'entité recherchée"],
            feedback="" if self._is_sufficient else "Contexte insuffisant.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers purs
# ─────────────────────────────────────────────────────────────────────────────


class TestHasDependents:
    """`has_dependents` — évite de payer un appel LLM inutile."""

    def test_step_with_a_dependent_is_detected(self) -> None:
        """`step_1` a un dépendant : la synthèse vaut la peine."""
        assert has_dependents(_plan(), "step_1") is True

    def test_last_step_has_no_dependent(self) -> None:
        """`step_2` n'a aucun dépendant : personne ne lirait sa réponse."""
        assert has_dependents(_plan(), "step_2") is False

    def test_unknown_step_has_no_dependent(self) -> None:
        """Un step_id absent du plan ne déclenche rien."""
        assert has_dependents(_plan(), "step_inexistant") is False


class TestBuildDependentSubQuery:
    """`build_dependent_sub_query` — concaténation, jamais substitution."""

    def test_answer_is_appended_to_the_original_query(self) -> None:
        """La sous-requête d'origine est CONSERVÉE, la réponse ajoutée."""
        result = build_dependent_sub_query(STEP_2_QUERY, [RESOLVED])

        assert result.startswith(STEP_2_QUERY)
        assert RESOLVED in result

    def test_several_dependencies_are_all_appended(self) -> None:
        """Un plan en losange concatène toutes les réponses amont."""
        result = build_dependent_sub_query(STEP_2_QUERY, ["Alpha.", "Bravo."])

        assert "Alpha." in result
        assert "Bravo." in result

    def test_no_answer_leaves_the_query_untouched(self) -> None:
        """Aucune réponse exploitable → sous-requête d'origine (B3)."""
        assert build_dependent_sub_query(STEP_2_QUERY, []) == STEP_2_QUERY

    def test_blank_answers_are_ignored(self) -> None:
        """Des réponses vides ou blanches ne polluent pas la requête."""
        assert build_dependent_sub_query(STEP_2_QUERY, ["", "   "]) == STEP_2_QUERY


# ─────────────────────────────────────────────────────────────────────────────
# Production de la réponse intermédiaire — nœud `critique`
# ─────────────────────────────────────────────────────────────────────────────


class TestCritiqueProducesIntermediateAnswer:
    """Le nœud `critique` ne synthétise que quand c'est utile ET fondé."""

    @staticmethod
    def _critique_state(
        is_sufficient: bool = True, step_id: str = "step_1"
    ) -> GraphState:
        plan = _plan()
        return _state(
            pending=[s.step_id for s in plan.steps],
            current_step_id=step_id,
            last_response=RetrievalResponse(
                query_id=plan.plan_id, chunks=list(STEP_1_CHUNKS), retrieval_score=0.9
            ),
        )

    @patch("reasoning.graph.nodes.completion")
    async def test_sufficient_step_with_dependents_is_synthesized(
        self, mock_completion: MagicMock
    ) -> None:
        """Contexte accepté + dépendant → réponse intermédiaire stockée."""
        mock_completion.return_value = _llm_response(RESOLVED)
        node = make_critique_node(FakeCritic(is_sufficient=True), ReasoningPolicy())

        result = await node(self._critique_state())

        assert result["step_answers"]["step_1"] == RESOLVED
        mock_completion.assert_called_once()

    @patch("reasoning.graph.nodes.completion")
    async def test_step_left_without_being_accepted_is_synthesized(
        self, mock_completion: MagicMock
    ) -> None:
        """DÉCLENCHEUR RELÂCHÉ : une étape quittée SANS acceptation est synthétisée.

        Configuration exacte du blocage mesuré au Lot B : le Critic rejette
        `step_1`, ses relances locales sont épuisées (`retry_count == 2 ==
        max_retries`) et la garde du Lot A fait avancer vers `step_2`. Sous
        l'ancien déclencheur (`is_sufficient=True`), aucune synthèse n'avait
        jamais lieu et l'entité résolue n'était jamais transmise.

        C'est la dernière occasion de capturer ce que les chunks contiennent.
        Le garde-fou contre une entité douteuse est la sentinelle `UNKNOWN`
        du prompt, pas le verdict du Critic.
        """
        mock_completion.return_value = _llm_response(RESOLVED)
        node = make_critique_node(FakeCritic(is_sufficient=False), ReasoningPolicy())

        state = self._critique_state()
        state["retry_counts"] = {"step_1": 2}  # relances locales épuisées

        result = await node(state)

        # L'étape a bien été quittée sans avoir été acceptée…
        assert result["pending_step_ids"] == ["step_2"]
        # …et sa réponse intermédiaire a tout de même été produite.
        assert result["step_answers"]["step_1"] == RESOLVED
        mock_completion.assert_called_once()

    @patch("reasoning.graph.nodes.completion")
    async def test_retried_step_is_not_synthesized(
        self, mock_completion: MagicMock
    ) -> None:
        """Étape RE-TENTÉE (non quittée) → aucune synthèse.

        `advance_step` est faux : l'étape reste en tête de file et sera
        réévaluée au passage suivant. Synthétiser ici paierait un appel LLM
        pour un résultat aussitôt périmé.
        """
        node = make_critique_node(FakeCritic(is_sufficient=False), ReasoningPolicy())

        result = await node(self._critique_state(is_sufficient=False))

        assert result["pending_step_ids"][0] == "step_1", "l'étape a été quittée"
        assert result["step_answers"] == {}
        mock_completion.assert_not_called()

    @patch("reasoning.graph.nodes.completion")
    async def test_step_without_dependents_is_not_synthesized(
        self, mock_completion: MagicMock
    ) -> None:
        """Dernière étape du plan → aucun appel LLM, personne ne lirait."""
        node = make_critique_node(FakeCritic(is_sufficient=True), ReasoningPolicy())

        result = await node(self._critique_state(step_id="step_2"))

        assert result["step_answers"] == {}
        mock_completion.assert_not_called()

    @patch("reasoning.graph.nodes.completion")
    async def test_prompt_carries_the_sub_query_and_the_chunks(
        self, mock_completion: MagicMock
    ) -> None:
        """La synthèse porte sur la sous-requête de l'étape et ses chunks."""
        mock_completion.return_value = _llm_response(RESOLVED)
        node = make_critique_node(FakeCritic(is_sufficient=True), ReasoningPolicy())

        await node(self._critique_state())

        prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
        assert STEP_1_QUERY in prompt
        assert "Shirley Temple" in prompt

    @patch("reasoning.graph.nodes.completion")
    async def test_fast_model_is_used_by_default(
        self, mock_completion: MagicMock
    ) -> None:
        """Le modèle par défaut est le RAPIDE — c'est de l'extraction."""
        mock_completion.return_value = _llm_response(RESOLVED)
        node = make_critique_node(FakeCritic(is_sufficient=True), ReasoningPolicy())

        await node(self._critique_state())

        assert "3b" in mock_completion.call_args.kwargs["model"]


class TestIntermediateAnswerDegradesCleanly:
    """B3 — toute défaillance retombe sur la sous-requête d'origine."""

    @staticmethod
    def _critique_state() -> GraphState:
        plan = _plan()
        return _state(
            pending=[s.step_id for s in plan.steps],
            current_step_id="step_1",
            last_response=RetrievalResponse(
                query_id=plan.plan_id, chunks=list(STEP_1_CHUNKS), retrieval_score=0.9
            ),
        )

    @pytest.mark.parametrize(
        "raw",
        ["UNKNOWN", "  unknown  ", "", "   ", "x" * 400],
        ids=["sentinelle", "sentinelle-casse", "vide", "blanc", "trop-long"],
    )
    @patch("reasoning.graph.nodes.completion")
    async def test_unusable_output_stores_nothing(
        self, mock_completion: MagicMock, raw: str
    ) -> None:
        """Sortie inexploitable → aucune entrée, aucune exception."""
        mock_completion.return_value = _llm_response(raw)
        node = make_critique_node(FakeCritic(is_sufficient=True), ReasoningPolicy())

        result = await node(self._critique_state())

        assert result["step_answers"] == {}

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("connexion refusée"),
            TimeoutError("délai dépassé"),
            RuntimeError("panne inattendue"),
        ],
        ids=["reseau", "timeout", "inattendue"],
    )
    @patch("reasoning.graph.nodes.completion")
    async def test_llm_failure_is_swallowed(
        self, mock_completion: MagicMock, exc: Exception
    ) -> None:
        """Appel LLM en échec → aucune exception propagée, routage intact."""
        mock_completion.side_effect = exc
        node = make_critique_node(FakeCritic(is_sufficient=True), ReasoningPolicy())

        result = await node(self._critique_state())

        assert result["step_answers"] == {}
        assert result["next_route"] != ""

    @patch("reasoning.graph.nodes.completion")
    async def test_empty_chunks_skip_the_call_entirely(
        self, mock_completion: MagicMock
    ) -> None:
        """Aucun chunk → rien à synthétiser, aucun appel LLM."""
        plan = _plan()
        state = _state(
            pending=[s.step_id for s in plan.steps],
            current_step_id="step_1",
            last_response=RetrievalResponse(
                query_id=plan.plan_id, chunks=[], retrieval_score=None
            ),
        )
        node = make_critique_node(FakeCritic(is_sufficient=True), ReasoningPolicy())

        result = await node(state)

        assert result["step_answers"] == {}
        mock_completion.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Consommation de la réponse intermédiaire — nœud `retrieve`
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrieveConsumesIntermediateAnswer:
    """B1 — la sous-requête dépendante porte l'entité résolue."""

    async def test_dependent_step_carries_the_resolved_entity(self) -> None:
        """`step_2` émet une requête contenant « Shirley Temple »."""
        client = SpyClient()
        node = make_retrieve_node(client)

        await node(_state(pending=["step_2"], step_answers={"step_1": RESOLVED}))

        sent = client.requests[0].sub_query
        assert sent.startswith(STEP_2_QUERY)
        assert "Shirley Temple" in sent

    async def test_independent_step_is_untouched(self) -> None:
        """Une étape sans `depends_on` part avec sa sous-requête d'origine."""
        client = SpyClient()
        node = make_retrieve_node(client)

        await node(_state(pending=["step_1", "step_2"], step_answers={}))

        assert client.requests[0].sub_query == STEP_1_QUERY

    async def test_missing_intermediate_answer_falls_back(self) -> None:
        """B3 — dépendance sans réponse intermédiaire → requête d'origine."""
        client = SpyClient()
        node = make_retrieve_node(client)

        await node(_state(pending=["step_2"], step_answers={}))

        assert client.requests[0].sub_query == STEP_2_QUERY

    async def test_state_without_the_key_at_all_falls_back(self) -> None:
        """`step_answers` absent de l'état (clé NotRequired) → aucun crash.

        Un `GraphState` construit sans cette clé reste valide : le champ est
        déclaré `NotRequired` pour n'imposer aucune migration au code
        existant.
        """
        state = _state(pending=["step_2"])
        del state["step_answers"]
        client = SpyClient()

        await make_retrieve_node(client)(state)

        assert client.requests[0].sub_query == STEP_2_QUERY

    async def test_contract_type_is_preserved(self) -> None:
        """`RetrievalRequest.sub_query` reste une chaîne — contrat inchangé."""
        client = SpyClient()

        await make_retrieve_node(client)(
            _state(pending=["step_2"], step_answers={"step_1": RESOLVED})
        )

        request = client.requests[0]
        assert isinstance(request.sub_query, str)
        assert RetrievalRequest.model_validate(request.model_dump()) == request


class TestBothEnrichmentsCompose:
    """Articulation Lot A × Lot B : les deux apports coexistent."""

    async def test_dependent_and_retried_step_keeps_both(self) -> None:
        """Une étape à la fois dépendante ET relancée cumule les deux apports.

        Ordre appliqué : dépendances (Lot B) d'abord, relance (Lot A) par
        dessus. Les deux sont additifs — l'un ajoute l'entité résolue, l'autre
        les aspects que le Critic déclare manquants.
        """
        client = SpyClient()
        node = make_retrieve_node(client)

        await node(
            _state(
                pending=["step_2"],
                step_answers={"step_1": RESOLVED},
                retry_counts={"step_2": 1},
                evaluations=[
                    CriticEvaluation(
                        step_id="step_2",
                        is_sufficient=False,
                        relevance_score=0.4,
                        missing_aspects=["ambassadorship"],
                        feedback="Le rôle exact manque.",
                    )
                ],
            )
        )

        sent = client.requests[0].sub_query
        assert sent.startswith(STEP_2_QUERY)
        assert "Shirley Temple" in sent, "l'apport du Lot B a été perdu"
        assert "ambassadorship" in sent, "l'apport du Lot A a été perdu"

    async def test_retry_alone_still_works_without_dependencies(self) -> None:
        """Non-régression Lot A : une relance sans dépendance reste enrichie."""
        client = SpyClient()
        node = make_retrieve_node(client)

        await node(
            _state(
                pending=["step_1", "step_2"],
                retry_counts={"step_1": 1},
                evaluations=[
                    CriticEvaluation(
                        step_id="step_1",
                        is_sufficient=False,
                        relevance_score=0.4,
                        missing_aspects=["actress name"],
                        feedback="",
                    )
                ],
            )
        )

        assert "actress name" in client.requests[0].sub_query


# ─────────────────────────────────────────────────────────────────────────────
# État initial
# ─────────────────────────────────────────────────────────────────────────────


class TestInitialState:
    """`build_initial_state` expose le nouveau champ, vide."""

    def test_step_answers_starts_empty(self) -> None:
        """Aucune réponse intermédiaire au démarrage."""
        assert build_initial_state("Une question quelconque")["step_answers"] == {}
