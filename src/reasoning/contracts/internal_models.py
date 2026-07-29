"""
Modèles Pydantic internes du module REASONING.

Ces contrats définissent les structures de données circulant
à l'intérieur du graphe LangGraph : entre le Query Analyzer,
le Planner, le Critic, le Verifier et l'état global de l'agent.

Ces modèles sont INTERNES au module REASONING et ne sont pas
partagés directement avec le module ACTION (voir action_interface.py).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Énumération : types de requêtes
# ─────────────────────────────────────────────────────────────────────────────


class QueryType(StrEnum):
    """Classification sémantique de la requête utilisateur.

    Hérite de `StrEnum` (Python 3.11+) : sérialisable en JSON nativement.

    Attributes:
        SIMPLE: Requête directe, réponse attendue en un seul retrieval.
        MULTI_HOP: Requête nécessitant plusieurs sauts de raisonnement.
        AMBIGUOUS: Requête dont l'intention est incertaine, nécessite clarification.
        COMPARATIVE: Requête demandant une comparaison entre plusieurs entités.
    """

    SIMPLE = "SIMPLE"
    MULTI_HOP = "MULTI_HOP"
    AMBIGUOUS = "AMBIGUOUS"
    COMPARATIVE = "COMPARATIVE"


class StepStatus(StrEnum):
    """Cycle de vie d'une étape dans le plan d'exécution.

    Hérite de `StrEnum` (Python 3.11+) : sérialisable en JSON nativement.

    Attributes:
        PENDING: L'étape est planifiée mais pas encore exécutée.
        IN_PROGRESS: Le retrieval est en cours pour cette étape.
        COMPLETED: L'étape a été exécutée et validée par le Critic.
    """

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


# ─────────────────────────────────────────────────────────────────────────────
# Sortie du Query Analyzer
# ─────────────────────────────────────────────────────────────────────────────


class AnalysisResult(BaseModel):
    """Résultat de l'analyse de la requête par le Query Analyzer.

    Attributes:
        query_type: Type de requête détecté.
        confidence: Niveau de confiance de la classification (0.0 – 1.0).
        detected_entities: Entités nommées ou concepts clés extraits de la requête.
        reasoning_budget: Nombre maximum d'étapes de raisonnement allouées.
    """

    query_type: QueryType
    confidence: float = Field(ge=0.0, le=1.0)
    detected_entities: list[str] = Field(default_factory=list)
    reasoning_budget: int = Field(ge=0, description="0 = AMBIGUOUS (aucun retrieval)")


# ─────────────────────────────────────────────────────────────────────────────
# Sortie du Planner
# ─────────────────────────────────────────────────────────────────────────────


class PlanStep(BaseModel):
    """Une étape atomique dans le plan d'exécution du Planner.

    Attributes:
        step_id: Identifiant unique de l'étape (ex : "step_1", "step_2").
        sub_query: Sous-requête à envoyer au module ACTION pour cette étape.
        depends_on: Liste des step_id dont cette étape dépend (exécution séquentielle).
        status: Statut courant de l'étape ("PENDING", "IN_PROGRESS", "COMPLETED").
    """

    step_id: str
    sub_query: str
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING


class ExecutionPlan(BaseModel):
    """Graphe d'exécution généré par le Planner pour une requête complexe.

    Attributes:
        plan_id: Identifiant unique du plan (ex : UUID ou hash de la requête).
        original_query: Requête utilisateur originale, non décomposée.
        steps: Liste ordonnée des étapes du plan.
        dependencies_graph: Représentation en liste d'adjacence des dépendances
            entre étapes (clé = step_id, valeur = liste des step_id prérequis).
    """

    plan_id: str
    original_query: str
    steps: list[PlanStep] = Field(default_factory=list)
    dependencies_graph: dict[str, list[str]] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Sortie du Critic
# ─────────────────────────────────────────────────────────────────────────────


class CriticEvaluation(BaseModel):
    """Évaluation du contexte récupéré par le Critic pour une étape donnée.

    Attributes:
        step_id: Identifiant de l'étape évaluée (référence à PlanStep.step_id).
        is_sufficient: True si le contexte est suffisant pour répondre à l'étape.
        relevance_score: Score de pertinence du contexte récupéré (0.0 – 1.0).
        missing_aspects: Aspects de la question non couverts par le contexte.
        feedback: Message structuré destiné au Planner pour guider une re-requête.
    """

    step_id: str
    is_sufficient: bool
    relevance_score: float = Field(ge=0.0, le=1.0)
    missing_aspects: list[str] = Field(default_factory=list)
    feedback: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Sortie du Verifier
# ─────────────────────────────────────────────────────────────────────────────


class VerificationResult(BaseModel):
    """Résultat de la vérification de fidélité (Groundedness) par le Verifier.

    Attributes:
        is_grounded: True si la réponse finale est entièrement fondée sur les sources.
        faithfulness_score: Proportion de claims supportés par les sources (0.0 – 1.0).
        unsupported_claims: Liste des affirmations non traçables dans les sources.
        final_answer: Réponse finale validée (ou tronquée si non-fondée).
    """

    is_grounded: bool
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    final_answer: str


# ─────────────────────────────────────────────────────────────────────────────
# État global du graphe LangGraph
# ─────────────────────────────────────────────────────────────────────────────


class AgentState(BaseModel):
    """État partagé et mutable circulant à travers tous les nœuds du graphe LangGraph.

    Chaque nœud lit et enrichit cet état. Il constitue la "mémoire de travail"
    de l'agent pour une requête donnée.

    Attributes:
        original_query: Requête utilisateur initiale, immuable durant l'exécution.
        analysis: Résultat du Query Analyzer (None avant son exécution).
        plan: Plan d'exécution du Planner (None avant son exécution).
        evaluations: Historique de toutes les évaluations du Critic.
        verification: Résultat du Verifier (None avant son exécution).
        feedback_loop_count: Compteur global de boucles de rétroaction
            (garde anti-infini).
    """

    original_query: str
    analysis: AnalysisResult | None = None
    plan: ExecutionPlan | None = None
    evaluations: list[CriticEvaluation] = Field(default_factory=list)
    verification: VerificationResult | None = None
    feedback_loop_count: int = Field(default=0, ge=0)
