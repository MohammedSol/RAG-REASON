"""
Tests unitaires — Sprint 1.3 : Contrats d'Interface JSON (Pydantic).

Objectifs :
    - Couverture 100% sur src/reasoning/contracts/
    - Sérialisation / désérialisation JSON (model_dump / model_validate)
    - Validation des contraintes strictes (Field validators)
    - Validation des Enums (QueryType, StepStatus)
    - Conformité mypy (typage strict, -> None sur toutes les fonctions)

Exécution :
    uv run pytest tests/unit/test_contracts.py -v --cov=src/reasoning/contracts
"""

import pytest
from pydantic import ValidationError
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
    StepStatus,
    VerificationResult,
)

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures partagées
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def valid_chunk() -> RetrievedChunk:
    """Chunk minimal valide."""
    return RetrievedChunk(
        chunk_id="chunk-001",
        content="Paris est la capitale de la France.",
        source="wikipedia/france",
        relevance_score=0.95,
    )


@pytest.fixture
def valid_request() -> RetrievalRequest:
    """Requête de retrieval minimale valide."""
    return RetrievalRequest(
        query_id="qid-001",
        sub_query="Quelle est la capitale de la France ?",
        hop_index=0,
        top_k=5,
    )


@pytest.fixture
def valid_response(valid_chunk: RetrievedChunk) -> RetrievalResponse:
    """Réponse de retrieval minimale valide."""
    return RetrievalResponse(
        query_id="qid-001",
        chunks=[valid_chunk],
    )


@pytest.fixture
def valid_plan_step() -> PlanStep:
    """Étape de plan minimale valide."""
    return PlanStep(step_id="step_1", sub_query="Qui est le président ?")


@pytest.fixture
def valid_execution_plan(valid_plan_step: PlanStep) -> ExecutionPlan:
    """Plan d'exécution minimal valide."""
    return ExecutionPlan(
        plan_id="plan-abc",
        original_query="Qui est le président de la France et quel est son âge ?",
        steps=[valid_plan_step],
        dependencies_graph={"step_1": []},
    )


@pytest.fixture
def valid_analysis() -> AnalysisResult:
    """Résultat d'analyse minimal valide."""
    return AnalysisResult(
        query_type=QueryType.MULTI_HOP,
        confidence=0.92,
        detected_entities=["France", "président"],
        reasoning_budget=3,
    )


@pytest.fixture
def valid_critic_eval() -> CriticEvaluation:
    """Évaluation du Critic minimale valide."""
    return CriticEvaluation(
        step_id="step_1",
        is_sufficient=True,
        relevance_score=0.85,
    )


@pytest.fixture
def valid_verification() -> VerificationResult:
    """Résultat de vérification minimal valide."""
    return VerificationResult(
        is_grounded=True,
        faithfulness_score=1.0,
        final_answer="Emmanuel Macron est le président de la France.",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tests : action_interface.py — RetrievedChunk
# ═════════════════════════════════════════════════════════════════════════════


class TestRetrievedChunk:
    """Couverture de RetrievedChunk."""

    def test_instantiation_valide(self, valid_chunk: RetrievedChunk) -> None:
        """Vérifie que les champs sont correctement assignés."""
        assert valid_chunk.chunk_id == "chunk-001"
        assert valid_chunk.content == "Paris est la capitale de la France."
        assert valid_chunk.source == "wikipedia/france"
        assert valid_chunk.relevance_score == 0.95

    def test_serialisation_json(self, valid_chunk: RetrievedChunk) -> None:
        """model_dump() doit retourner un dict JSON-compatible."""
        data = valid_chunk.model_dump()
        assert data["chunk_id"] == "chunk-001"
        assert data["relevance_score"] == 0.95

    def test_deserialisation_depuis_dict(self) -> None:
        """model_validate() reconstruit le modèle depuis un dict brut."""
        raw = {
            "chunk_id": "c-002",
            "content": "Texte du chunk.",
            "source": "doc/A",
            "relevance_score": 0.7,
        }
        chunk = RetrievedChunk.model_validate(raw)
        assert chunk.chunk_id == "c-002"
        assert chunk.relevance_score == 0.7

    def test_champ_manquant_leve_erreur(self) -> None:
        """Un champ requis absent doit lever ValidationError."""
        with pytest.raises(ValidationError):
            RetrievedChunk.model_validate(
                {"chunk_id": "c-003", "content": "Texte."}
                # source et relevance_score manquants
            )


# ═════════════════════════════════════════════════════════════════════════════
# Tests : action_interface.py — RetrievalRequest
# ═════════════════════════════════════════════════════════════════════════════


class TestRetrievalRequest:
    """Couverture de RetrievalRequest, notamment la contrainte top_k > 0."""

    def test_instantiation_minimale(self, valid_request: RetrievalRequest) -> None:
        """Vérifie les champs requis et les optionnels à None."""
        assert valid_request.query_id == "qid-001"
        assert valid_request.hop_index == 0
        assert valid_request.top_k == 5
        assert valid_request.filters is None
        assert valid_request.metadata is None

    def test_top_k_zero_leve_erreur(self) -> None:
        """top_k = 0 viole la contrainte Field(gt=0)."""
        with pytest.raises(ValidationError):
            RetrievalRequest(
                query_id="q1",
                sub_query="Test",
                hop_index=0,
                top_k=0,
            )

    def test_top_k_negatif_leve_erreur(self) -> None:
        """top_k négatif viole la contrainte Field(gt=0)."""
        with pytest.raises(ValidationError):
            RetrievalRequest(
                query_id="q1",
                sub_query="Test",
                hop_index=0,
                top_k=-1,
            )

    def test_top_k_positif_accepte(self) -> None:
        """top_k = 1 est la valeur minimale valide."""
        req = RetrievalRequest(
            query_id="q1",
            sub_query="Test",
            hop_index=0,
            top_k=1,
        )
        assert req.top_k == 1

    def test_champs_optionnels_renseignes(self) -> None:
        """filters et metadata sont acceptés quand fournis."""
        req = RetrievalRequest(
            query_id="q2",
            sub_query="Test",
            hop_index=1,
            top_k=10,
            filters={"lang": "fr"},
            metadata={"source_type": "article"},
        )
        assert req.filters == {"lang": "fr"}
        assert req.metadata == {"source_type": "article"}

    def test_serialisation_roundtrip(self, valid_request: RetrievalRequest) -> None:
        """Sérialisation puis désérialisation produit un objet identique."""
        data = valid_request.model_dump()
        reconstructed = RetrievalRequest.model_validate(data)
        assert reconstructed == valid_request


# ═════════════════════════════════════════════════════════════════════════════
# Tests : action_interface.py — RetrievalResponse
# ═════════════════════════════════════════════════════════════════════════════


class TestRetrievalResponse:
    """Couverture de RetrievalResponse."""

    def test_instantiation_avec_chunks(self, valid_response: RetrievalResponse) -> None:
        """Vérifie la structure de la réponse et l'imbrication des chunks."""
        assert valid_response.query_id == "qid-001"
        assert len(valid_response.chunks) == 1
        assert valid_response.chunks[0].chunk_id == "chunk-001"
        assert valid_response.retrieval_score is None
        assert valid_response.metadata is None

    def test_liste_chunks_vide_acceptee(self) -> None:
        """Une réponse sans chunks est valide (retrieval vide)."""
        resp = RetrievalResponse(query_id="q-empty", chunks=[])
        assert resp.chunks == []

    def test_retrieval_score_optionnel(self) -> None:
        """retrieval_score peut être fourni ou absent."""
        resp = RetrievalResponse(
            query_id="q-score",
            chunks=[],
            retrieval_score=0.75,
        )
        assert resp.retrieval_score == 0.75

    def test_serialisation_roundtrip(self, valid_response: RetrievalResponse) -> None:
        """Sérialisation puis désérialisation conserve les chunks imbriqués."""
        data = valid_response.model_dump()
        reconstructed = RetrievalResponse.model_validate(data)
        assert reconstructed.chunks[0].content == valid_response.chunks[0].content


# ═════════════════════════════════════════════════════════════════════════════
# Tests : internal_models.py — QueryType
# ═════════════════════════════════════════════════════════════════════════════


class TestQueryType:
    """Couverture de l'Enum QueryType."""

    def test_toutes_les_valeurs_existent(self) -> None:
        """Les 4 membres doivent être accessibles."""
        assert QueryType.SIMPLE == "SIMPLE"
        assert QueryType.MULTI_HOP == "MULTI_HOP"
        assert QueryType.AMBIGUOUS == "AMBIGUOUS"
        assert QueryType.COMPARATIVE == "COMPARATIVE"

    def test_est_instance_str(self) -> None:
        """QueryType hérite de str : compatible JSON nativement."""
        assert isinstance(QueryType.SIMPLE, str)

    def test_valeur_inconnue_leve_erreur_dans_modele(self) -> None:
        """Un QueryType inconnu dans AnalysisResult doit lever ValidationError."""
        with pytest.raises(ValidationError):
            AnalysisResult.model_validate(
                {
                    "query_type": "INEXISTANT",
                    "confidence": 0.9,
                    "reasoning_budget": 2,
                }
            )


# ═════════════════════════════════════════════════════════════════════════════
# Tests : internal_models.py — StepStatus
# ═════════════════════════════════════════════════════════════════════════════


class TestStepStatus:
    """Couverture de l'Enum StepStatus et son intégration dans PlanStep."""

    def test_toutes_les_valeurs_existent(self) -> None:
        """Les 3 statuts du cycle de vie doivent être accessibles."""
        assert StepStatus.PENDING == "PENDING"
        assert StepStatus.IN_PROGRESS == "IN_PROGRESS"
        assert StepStatus.COMPLETED == "COMPLETED"

    def test_est_instance_str(self) -> None:
        """StepStatus hérite de str."""
        assert isinstance(StepStatus.IN_PROGRESS, str)

    def test_statut_par_defaut_pending(self, valid_plan_step: PlanStep) -> None:
        """Le statut par défaut d'un PlanStep est PENDING."""
        assert valid_plan_step.status == StepStatus.PENDING

    def test_statut_inconnu_leve_validation_error(self) -> None:
        """Un statut non défini dans StepStatus doit lever ValidationError."""
        with pytest.raises(ValidationError):
            PlanStep(
                step_id="s1",
                sub_query="Question",
                status="UN_STATUT_INVENTE",  # type: ignore[arg-type]
            )

    def test_transition_vers_in_progress(self) -> None:
        """Transition explicite vers IN_PROGRESS."""
        step = PlanStep(
            step_id="s2",
            sub_query="Question",
            status=StepStatus.IN_PROGRESS,
        )
        assert step.status == StepStatus.IN_PROGRESS

    def test_transition_vers_completed(self) -> None:
        """Transition explicite vers COMPLETED."""
        step = PlanStep(
            step_id="s3",
            sub_query="Question",
            status=StepStatus.COMPLETED,
        )
        assert step.status == StepStatus.COMPLETED


# ═════════════════════════════════════════════════════════════════════════════
# Tests : internal_models.py — AnalysisResult
# ═════════════════════════════════════════════════════════════════════════════


class TestAnalysisResult:
    """Couverture de AnalysisResult."""

    def test_instantiation_valide(self, valid_analysis: AnalysisResult) -> None:
        assert valid_analysis.query_type == QueryType.MULTI_HOP
        assert valid_analysis.confidence == 0.92
        assert valid_analysis.reasoning_budget == 3

    def test_confidence_hors_plage_leve_erreur(self) -> None:
        """confidence > 1.0 viole Field(le=1.0)."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                query_type=QueryType.SIMPLE,
                confidence=1.1,
                reasoning_budget=1,
            )

    def test_confidence_negative_leve_erreur(self) -> None:
        """confidence < 0.0 viole Field(ge=0.0)."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                query_type=QueryType.SIMPLE,
                confidence=-0.1,
                reasoning_budget=1,
            )

    def test_reasoning_budget_zero_autorise_pour_ambiguous(self) -> None:
        """reasoning_budget = 0 est valide : c'est la valeur AMBIGUOUS (ge=0)."""
        analysis = AnalysisResult(
            query_type=QueryType.AMBIGUOUS,
            confidence=0.8,
            reasoning_budget=0,
        )
        assert analysis.reasoning_budget == 0

    def test_reasoning_budget_negatif_leve_erreur(self) -> None:
        """reasoning_budget < 0 viole Field(ge=0)."""
        with pytest.raises(ValidationError):
            AnalysisResult(
                query_type=QueryType.SIMPLE,
                confidence=0.8,
                reasoning_budget=-1,
            )

    def test_detected_entities_vide_par_defaut(self) -> None:
        """detected_entities est une liste vide par défaut."""
        analysis = AnalysisResult(
            query_type=QueryType.SIMPLE,
            confidence=0.5,
            reasoning_budget=1,
        )
        assert analysis.detected_entities == []

    def test_detected_entities_independantes_entre_instances(self) -> None:
        """Deux instances ne partagent pas la même liste (default_factory)."""
        a1 = AnalysisResult(
            query_type=QueryType.SIMPLE, confidence=0.5, reasoning_budget=1
        )
        a2 = AnalysisResult(
            query_type=QueryType.SIMPLE, confidence=0.5, reasoning_budget=1
        )
        a1.detected_entities.append("test")
        assert a2.detected_entities == []


# ═════════════════════════════════════════════════════════════════════════════
# Tests : internal_models.py — PlanStep & ExecutionPlan
# ═════════════════════════════════════════════════════════════════════════════


class TestPlanStep:
    """Couverture de PlanStep."""

    def test_depends_on_vide_par_defaut(self, valid_plan_step: PlanStep) -> None:
        assert valid_plan_step.depends_on == []

    def test_depends_on_renseignes(self) -> None:
        """depends_on accepte une liste de step_id."""
        step = PlanStep(
            step_id="step_2",
            sub_query="Quel est son âge ?",
            depends_on=["step_1"],
        )
        assert "step_1" in step.depends_on

    def test_serialisation_roundtrip(self, valid_plan_step: PlanStep) -> None:
        data = valid_plan_step.model_dump()
        reconstructed = PlanStep.model_validate(data)
        assert reconstructed == valid_plan_step


class TestExecutionPlan:
    """Couverture de ExecutionPlan."""

    def test_instantiation_valide(self, valid_execution_plan: ExecutionPlan) -> None:
        assert valid_execution_plan.plan_id == "plan-abc"
        assert len(valid_execution_plan.steps) == 1

    def test_plan_vide_accepte(self) -> None:
        """Un plan sans étapes est valide (ex: requête SIMPLE non décomposée)."""
        plan = ExecutionPlan(plan_id="empty", original_query="Bonjour")
        assert plan.steps == []
        assert plan.dependencies_graph == {}

    def test_serialisation_roundtrip(self, valid_execution_plan: ExecutionPlan) -> None:
        """Sérialisation du graphe imbriqué (steps contenant des PlanStep)."""
        data = valid_execution_plan.model_dump()
        reconstructed = ExecutionPlan.model_validate(data)
        assert reconstructed.steps[0].step_id == "step_1"
        assert reconstructed.steps[0].status == StepStatus.PENDING


# ═════════════════════════════════════════════════════════════════════════════
# Tests : internal_models.py — CriticEvaluation
# ═════════════════════════════════════════════════════════════════════════════


class TestCriticEvaluation:
    """Couverture de CriticEvaluation."""

    def test_instantiation_valide(self, valid_critic_eval: CriticEvaluation) -> None:
        assert valid_critic_eval.is_sufficient is True
        assert valid_critic_eval.relevance_score == 0.85
        assert valid_critic_eval.missing_aspects == []
        assert valid_critic_eval.feedback == ""

    def test_relevance_score_hors_plage(self) -> None:
        """relevance_score > 1.0 viole Field(le=1.0)."""
        with pytest.raises(ValidationError):
            CriticEvaluation(
                step_id="s1",
                is_sufficient=False,
                relevance_score=1.5,
            )

    def test_is_sufficient_false_avec_feedback(self) -> None:
        """Un Critic négatif doit pouvoir transmettre des aspects manquants."""
        eval_ = CriticEvaluation(
            step_id="s1",
            is_sufficient=False,
            relevance_score=0.3,
            missing_aspects=["date", "lieu"],
            feedback="Rechercher des informations temporelles et géographiques.",
        )
        assert len(eval_.missing_aspects) == 2
        assert "date" in eval_.missing_aspects


# ═════════════════════════════════════════════════════════════════════════════
# Tests : internal_models.py — VerificationResult
# ═════════════════════════════════════════════════════════════════════════════


class TestVerificationResult:
    """Couverture de VerificationResult."""

    def test_reponse_totalement_fondee(
        self, valid_verification: VerificationResult
    ) -> None:
        assert valid_verification.is_grounded is True
        assert valid_verification.faithfulness_score == 1.0
        assert valid_verification.unsupported_claims == []

    def test_reponse_avec_hallucination(self) -> None:
        """is_grounded=False avec des claims non supportés."""
        result = VerificationResult(
            is_grounded=False,
            faithfulness_score=0.5,
            unsupported_claims=["Il a 45 ans."],
            final_answer="Emmanuel Macron est président. Il a 45 ans.",
        )
        assert result.faithfulness_score == 0.5
        assert len(result.unsupported_claims) == 1

    def test_faithfulness_score_negatif_leve_erreur(self) -> None:
        """faithfulness_score < 0.0 viole Field(ge=0.0)."""
        with pytest.raises(ValidationError):
            VerificationResult(
                is_grounded=False,
                faithfulness_score=-0.1,
                final_answer="Réponse.",
            )

    def test_serialisation_roundtrip(
        self, valid_verification: VerificationResult
    ) -> None:
        data = valid_verification.model_dump()
        reconstructed = VerificationResult.model_validate(data)
        assert reconstructed == valid_verification


# ═════════════════════════════════════════════════════════════════════════════
# Tests : internal_models.py — AgentState
# ═════════════════════════════════════════════════════════════════════════════


class TestAgentState:
    """Couverture de AgentState — état global du graphe LangGraph."""

    def test_etat_initial_vide(self) -> None:
        """À sa création, un AgentState ne contient que la requête originale."""
        state = AgentState(original_query="Qui est le président ?")
        assert state.analysis is None
        assert state.plan is None
        assert state.evaluations == []
        assert state.verification is None
        assert state.feedback_loop_count == 0

    def test_enrichissement_progressif(
        self,
        valid_analysis: AnalysisResult,
        valid_execution_plan: ExecutionPlan,
        valid_critic_eval: CriticEvaluation,
        valid_verification: VerificationResult,
    ) -> None:
        """Simule l'enrichissement de l'état nœud par nœud."""
        state = AgentState(original_query="Question multi-hop")
        state.analysis = valid_analysis
        state.plan = valid_execution_plan
        state.evaluations.append(valid_critic_eval)
        state.feedback_loop_count += 1
        state.verification = valid_verification

        assert state.analysis.query_type == QueryType.MULTI_HOP
        assert state.plan.plan_id == "plan-abc"
        assert len(state.evaluations) == 1
        assert state.feedback_loop_count == 1
        assert state.verification.is_grounded is True

    def test_feedback_loop_count_negatif_leve_erreur(self) -> None:
        """feedback_loop_count < 0 viole Field(ge=0)."""
        with pytest.raises(ValidationError):
            AgentState(
                original_query="Test",
                feedback_loop_count=-1,
            )

    def test_evaluations_independantes_entre_instances(self) -> None:
        """Deux AgentState ne partagent pas la même liste d'évaluations."""
        s1 = AgentState(original_query="Q1")
        s2 = AgentState(original_query="Q2")
        s1.evaluations.append(
            CriticEvaluation(step_id="s", is_sufficient=True, relevance_score=0.9)
        )
        assert s2.evaluations == []

    def test_serialisation_roundtrip_etat_complet(
        self,
        valid_analysis: AnalysisResult,
        valid_execution_plan: ExecutionPlan,
        valid_verification: VerificationResult,
    ) -> None:
        """Sérialisation / désérialisation d'un AgentState entièrement rempli."""
        state = AgentState(
            original_query="Question complète",
            analysis=valid_analysis,
            plan=valid_execution_plan,
            verification=valid_verification,
            feedback_loop_count=2,
        )
        data = state.model_dump()
        reconstructed = AgentState.model_validate(data)

        assert reconstructed.original_query == "Question complète"
        assert reconstructed.analysis is not None
        assert reconstructed.analysis.query_type == QueryType.MULTI_HOP
        assert reconstructed.plan is not None
        assert reconstructed.plan.steps[0].status == StepStatus.PENDING
        assert reconstructed.verification is not None
        assert reconstructed.verification.faithfulness_score == 1.0
        assert reconstructed.feedback_loop_count == 2
