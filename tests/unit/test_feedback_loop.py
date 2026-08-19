"""
Tests unitaires — Lot A : boucle de rétroaction fonctionnelle.

Couvre les deux corrections issues des constats du Sprint I4 :

    §2  le retour du Critic n'était jamais propagé — la relance renvoyait la
        sous-requête inchangée à un moteur déterministe, donc les mêmes chunks ;
    §4  une étape épuisait à elle seule le budget global, si bien que les
        étapes suivantes du plan n'étaient jamais tentées.

Aucun appel réseau, aucun appel LLM : le client de retrieval est un espion
in-memory et la politique est testée en valeurs pures.

Exécution :
    uv run pytest tests/unit/test_feedback_loop.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from reasoning.graph.nodes import enrich_sub_query, make_retrieve_node
from reasoning.graph.policy import (
    ROUTE_GENERATE_ANSWER,
    ROUTE_RETRIEVE,
    ReasoningPolicy,
)
from reasoning.graph.state import GraphState

# ─────────────────────────────────────────────────────────────────────────────
# Données de référence — reprises de la trace réelle du Sprint I4
# ─────────────────────────────────────────────────────────────────────────────

BASE_SUB_QUERY = "Who portrayed Corliss Archer in the film Kiss and Tell?"


def _evaluation(
    step_id: str = "step_1",
    *,
    is_sufficient: bool = False,
    missing_aspects: list[str] | None = None,
    feedback: str = "",
) -> CriticEvaluation:
    """Construit une CriticEvaluation de test."""
    return CriticEvaluation(
        step_id=step_id,
        is_sufficient=is_sufficient,
        relevance_score=0.6 if not is_sufficient else 0.9,
        missing_aspects=[] if missing_aspects is None else missing_aspects,
        feedback=feedback,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §2 — Enrichissement de la sous-requête
# ─────────────────────────────────────────────────────────────────────────────


class TestEnrichSubQuery:
    """`enrich_sub_query` — expansion de requête par rétroaction de pertinence."""

    def test_no_evaluation_leaves_query_untouched(self) -> None:
        """Sans évaluation antérieure, la requête d'origine est renvoyée."""
        assert enrich_sub_query(BASE_SUB_QUERY, None) == BASE_SUB_QUERY

    def test_missing_aspects_are_appended_verbatim(self) -> None:
        """Les aspects manquants sont ajoutés tels quels, après la requête."""
        evaluation = _evaluation(missing_aspects=["birth year", "nationality"])

        enriched = enrich_sub_query(BASE_SUB_QUERY, evaluation)

        assert enriched.startswith(BASE_SUB_QUERY)
        assert "birth year" in enriched
        assert "nationality" in enriched

    def test_aspect_already_in_query_is_not_duplicated(self) -> None:
        """Un aspect déjà présent dans la requête n'est pas répété."""
        evaluation = _evaluation(missing_aspects=["Corliss Archer"])

        enriched = enrich_sub_query(BASE_SUB_QUERY, evaluation)

        assert enriched.lower().count("corliss archer") == 1

    def test_feedback_content_words_are_added(self) -> None:
        """Les mots de contenu du feedback, absents par ailleurs, sont retenus."""
        evaluation = _evaluation(
            missing_aspects=[],
            feedback="The context does not mention her senatorial appointment.",
        )

        enriched = enrich_sub_query(BASE_SUB_QUERY, evaluation)

        assert "senatorial" in enriched
        assert "appointment" in enriched

    def test_feedback_stopwords_are_filtered(self) -> None:
        """Les tournures récurrentes du Critic n'entrent pas dans la requête."""
        evaluation = _evaluation(
            missing_aspects=[],
            feedback=(
                "The retrieved context does not provide any information about "
                "this. More information is needed."
            ),
        )

        enriched = enrich_sub_query(BASE_SUB_QUERY, evaluation)

        assert enriched == BASE_SUB_QUERY, (
            f"des mots vides ont été injectés : {enriched!r}"
        )

    def test_empty_feedback_leaves_query_untouched(self) -> None:
        """Un Critic sans rien à dire ne modifie pas la requête.

        Cas légitime et observable : le nœud `retrieve` le journalise
        explicitement comme « sous-requête INCHANGÉE ».
        """
        assert enrich_sub_query(BASE_SUB_QUERY, _evaluation()) == BASE_SUB_QUERY

    def test_result_stays_a_usable_search_query(self) -> None:
        """La requête reste bornée — un moteur lexical dilue les requêtes longues."""
        evaluation = _evaluation(
            missing_aspects=[f"aspect numéro {i}" for i in range(40)],
            feedback="Beaucoup de mots inutiles " * 40,
        )

        enriched = enrich_sub_query(BASE_SUB_QUERY, evaluation)

        assert len(enriched) <= 300
        assert enriched.startswith(BASE_SUB_QUERY)

    def test_feedback_terms_are_capped(self) -> None:
        """Au plus quelques termes issus du feedback, jamais la phrase entière."""
        evaluation = _evaluation(
            missing_aspects=[],
            feedback="alpha bravo charlie delta echo foxtrot golf hotel india",
        )

        added = enrich_sub_query(BASE_SUB_QUERY, evaluation)[len(BASE_SUB_QUERY) :]

        assert len(added.split()) <= 4


# ─────────────────────────────────────────────────────────────────────────────
# §2 — Câblage dans le nœud `retrieve`
# ─────────────────────────────────────────────────────────────────────────────


class SpyClient:
    """Client de retrieval in-memory : capture les requêtes émises."""

    def __init__(self) -> None:
        self.requests: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        self.requests.append(request)
        return RetrievalResponse(
            query_id=request.query_id,
            chunks=[
                RetrievedChunk(
                    chunk_id="202",
                    content="A Kiss for Corliss stars Shirley Temple.",
                    source="A_Kiss_for_Corliss.txt",
                    relevance_score=0.92,
                )
            ],
            retrieval_score=0.92,
        )


def _plan() -> ExecutionPlan:
    """Plan à deux étapes, de la forme produite par le Planner."""
    return ExecutionPlan(
        plan_id="plan-lot-a",
        original_query="What government position was held by …?",
        steps=[
            PlanStep(step_id="step_1", sub_query=BASE_SUB_QUERY),
            PlanStep(
                step_id="step_2",
                sub_query="What government position did the identified woman hold?",
                depends_on=["step_1"],
            ),
        ],
        dependencies_graph={"step_1": [], "step_2": ["step_1"]},
    )


def _state(
    *,
    retry_counts: dict[str, int],
    evaluations: list[CriticEvaluation],
) -> GraphState:
    """GraphState complet, prêt pour l'exécution du nœud `retrieve`."""
    plan = _plan()
    return GraphState(
        agent_state=AgentState(
            original_query=plan.original_query,
            analysis=AnalysisResult(
                query_type=QueryType.MULTI_HOP,
                confidence=0.86,
                detected_entities=[],
                reasoning_budget=3,
            ),
            plan=plan,
            evaluations=evaluations,
        ),
        retrieved_chunks=[],
        retry_counts=retry_counts,
        pending_step_ids=["step_1", "step_2"],
        current_step_id=None,
        last_retrieval_response=None,
        answer=None,
        next_route="",
    )


class TestRetrieveNodeUsesFeedback:
    """Le nœud `retrieve` n'enrichit QUE les relances."""

    async def test_first_attempt_sends_the_untouched_sub_query(self) -> None:
        """Première tentative : `PlanStep.sub_query` est envoyée telle quelle.

        Même si une évaluation existe déjà (cas d'une re-planification), un
        premier passage sur l'étape ne doit pas être enrichi.
        """
        client = SpyClient()
        node = make_retrieve_node(client)

        await node(
            _state(
                retry_counts={},
                evaluations=[_evaluation(missing_aspects=["government position"])],
            )
        )

        assert client.requests[0].sub_query == BASE_SUB_QUERY

    async def test_retry_sends_an_enriched_sub_query(self) -> None:
        """Relance : la sous-requête porte le retour du Critic."""
        client = SpyClient()
        node = make_retrieve_node(client)

        await node(
            _state(
                retry_counts={"step_1": 1},
                evaluations=[_evaluation(missing_aspects=["Shirley Temple biography"])],
            )
        )

        sent = client.requests[0].sub_query
        assert sent != BASE_SUB_QUERY
        assert sent.startswith(BASE_SUB_QUERY)
        assert "Shirley Temple biography" in sent

    async def test_retry_uses_the_latest_evaluation_of_that_step(self) -> None:
        """C'est le DERNIER verdict de l'étape courante qui est propagé."""
        client = SpyClient()
        node = make_retrieve_node(client)

        await node(
            _state(
                retry_counts={"step_1": 2},
                evaluations=[
                    _evaluation("step_1", missing_aspects=["premier aspect"]),
                    _evaluation("step_2", missing_aspects=["aspect d'une autre étape"]),
                    _evaluation("step_1", missing_aspects=["dernier aspect"]),
                ],
            )
        )

        sent = client.requests[0].sub_query
        assert "dernier aspect" in sent
        assert "premier aspect" not in sent
        assert "aspect d'une autre étape" not in sent

    async def test_contract_type_is_preserved(self) -> None:
        """`RetrievalRequest.sub_query` reste une chaîne — le contrat ne change pas."""
        client = SpyClient()
        node = make_retrieve_node(client)

        await node(
            _state(
                retry_counts={"step_1": 1},
                evaluations=[_evaluation(missing_aspects=["un aspect"])],
            )
        )

        request = client.requests[0]
        assert isinstance(request.sub_query, str)
        assert RetrievalRequest.model_validate(request.model_dump()) == request


# ─────────────────────────────────────────────────────────────────────────────
# §4 — Garde locale : une étape n'affame plus les suivantes
# ─────────────────────────────────────────────────────────────────────────────


class TestLocalGuardAdvancesThePlan:
    """Articulation entre garde locale `max_retries` et budget global."""

    def test_exhausted_retries_advance_even_when_budget_is_spent(self) -> None:
        """CORRECTION §4 : l'étape suivante est tentée malgré le budget épuisé.

        Configuration exacte du blocage mesuré au Sprint I4 : budget global 3
        entièrement consommé par les relances de `step_1`, alors que `step_2`
        n'avait jamais été tenté. La garde locale prime désormais, car avancer
        dans le plan est un progrès, jamais une boucle.
        """
        decision = ReasoningPolicy().route_after_critique(
            reasoning_budget=3,
            feedback_loop_count=3,
            is_sufficient=False,
            retry_count=2,
            max_retries=2,
            has_next_step=True,
        )

        assert decision.route == ROUTE_RETRIEVE
        assert decision.advance_step is True

    def test_exhausted_retries_without_next_step_still_generate(self) -> None:
        """Sans étape suivante, la garde globale reprend la main : pas d'échappatoire."""
        decision = ReasoningPolicy().route_after_critique(
            reasoning_budget=3,
            feedback_loop_count=3,
            is_sufficient=False,
            retry_count=2,
            max_retries=2,
            has_next_step=False,
        )

        assert decision.route == ROUTE_GENERATE_ANSWER
        assert decision.advance_step is True

    def test_budget_still_wins_when_retries_remain(self) -> None:
        """Garde-fou de non-régression : le budget reste prioritaire par défaut."""
        decision = ReasoningPolicy().route_after_critique(
            reasoning_budget=2,
            feedback_loop_count=2,
            is_sufficient=False,
            retry_count=0,
            max_retries=2,
            has_next_step=True,
        )

        assert decision.route == ROUTE_GENERATE_ANSWER
        assert decision.advance_step is True

    def test_budget_still_wins_when_context_is_sufficient(self) -> None:
        """Garde-fou : un contexte suffisant ne contourne pas la garde globale."""
        decision = ReasoningPolicy().route_after_critique(
            reasoning_budget=2,
            feedback_loop_count=2,
            is_sufficient=True,
            retry_count=2,
            max_retries=2,
            has_next_step=True,
        )

        assert decision.route == ROUTE_GENERATE_ANSWER
        assert decision.advance_step is True


class TestTerminationIsGuaranteed:
    """A3 — aucune combinaison ne produit de boucle infinie."""

    @staticmethod
    def _simulate(
        *, n_steps: int, budget: int, max_retries: int, always_insufficient: bool
    ) -> tuple[int, int]:
        """Rejoue la boucle critique→retrieve en valeurs pures.

        Returns:
            (nombre de critiques exécutées, nombre d'étapes restées en file).
        """
        policy = ReasoningPolicy()
        pending = [f"step_{i}" for i in range(1, n_steps + 1)]
        retry_counts: dict[str, int] = {}
        feedback_loop_count = 0
        critiques = 0
        hard_stop = 10_000

        while pending and critiques < hard_stop:
            step_id = pending[0]
            feedback_loop_count += 1
            critiques += 1
            decision = policy.route_after_critique(
                reasoning_budget=budget,
                feedback_loop_count=feedback_loop_count,
                is_sufficient=not always_insufficient,
                retry_count=retry_counts.get(step_id, 0),
                max_retries=max_retries,
                has_next_step=len(pending) > 1,
            )
            if decision.advance_step:
                pending.pop(0)
                retry_counts.pop(step_id, None)
            else:
                retry_counts[step_id] = retry_counts.get(step_id, 0) + 1
            if decision.route == ROUTE_GENERATE_ANSWER:
                break

        assert critiques < hard_stop, "la boucle ne termine pas"
        return critiques, len(pending)

    @pytest.mark.parametrize("n_steps", [1, 2, 3, 5, 10])
    @pytest.mark.parametrize("budget", [0, 1, 2, 3, 10])
    @pytest.mark.parametrize("max_retries", [0, 1, 2, 5])
    def test_loop_always_terminates(
        self, n_steps: int, budget: int, max_retries: int
    ) -> None:
        """Quelle que soit la combinaison, la boucle se termine.

        Le pire cas est `always_insufficient=True` : le Critic ne valide jamais
        rien, donc seules les gardes peuvent arrêter l'exécution.
        """
        critiques, _ = self._simulate(
            n_steps=n_steps,
            budget=budget,
            max_retries=max_retries,
            always_insufficient=True,
        )

        # Borne théorique : chaque étape consomme au plus 1 + max_retries
        # critiques, et la garde globale peut interrompre plus tôt.
        assert critiques <= n_steps * (1 + max_retries)

    def test_sprint_i4_configuration_now_reaches_the_second_step(self) -> None:
        """La configuration exacte du Sprint I4 atteint désormais `step_2`.

        Au baseline : 3 critiques, toutes sur `step_1`, `step_2` jamais tenté.
        """
        critiques, remaining = self._simulate(
            n_steps=2, budget=3, max_retries=2, always_insufficient=True
        )

        # 3 critiques sur step_1 (1 initiale + 2 relances) puis la garde locale
        # fait avancer, et step_2 est bien tenté.
        assert critiques >= 4, (
            "step_2 n'a pas été atteint — la correction du §4 est inopérante"
        )
        assert remaining == 0


# ─────────────────────────────────────────────────────────────────────────────
# A4 — Séparation architecturale
# ─────────────────────────────────────────────────────────────────────────────


class TestPolicyStaysFrameworkFree:
    """`policy.py` ne doit dépendre d'aucun framework d'orchestration."""

    def test_policy_source_has_no_orchestration_import(self) -> None:
        """Aucune occurrence du paquet d'orchestration dans le source.

        Contrainte d'architecture posée en tête de `policy.py` : la logique de
        décision reste du Python pur, testable sans LangGraph ni LLM.
        """
        import reasoning.graph.policy as policy_module

        source = Path(policy_module.__file__).read_text(encoding="utf-8")
        offending = [
            line
            for line in source.splitlines()
            if line.lstrip().startswith(("import ", "from "))
            and "langgraph" in line.lower()
        ]
        assert offending == [], f"import d'orchestration détecté : {offending}"

    def test_policy_module_does_not_pull_orchestration_at_runtime(self) -> None:
        """Le module ne référence pas non plus le framework dans son espace de noms."""
        import reasoning.graph.policy as policy_module

        pulled = [
            name
            for name, value in vars(policy_module).items()
            if getattr(value, "__module__", "").startswith("langgraph")
        ]
        assert pulled == [], f"symboles d'orchestration exposés : {pulled}"

    def test_policy_is_usable_without_any_graph_state(self) -> None:
        """La politique se pilote en valeurs primitives, sans objet de framework."""
        policy = ReasoningPolicy()
        decision: Any = policy.route_after_critique(
            reasoning_budget=3,
            feedback_loop_count=1,
            is_sufficient=True,
            retry_count=0,
            max_retries=2,
            has_next_step=True,
        )
        assert isinstance(decision.route, str)
        assert isinstance(decision.advance_step, bool)
