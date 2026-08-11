"""
Interfaces structurelles (`Protocol`) des composants injectables du graphe.

`build_graph()` accepte n'importe quelle implémentation satisfaisant ces
protocoles — pas seulement les classes concrètes `QueryAnalyzer`, `Planner`,
`Critic`, `Verifier`. Cela permet l'injection de doubles de test
(`tests/integration/test_graph.py`) sans lien de sous-classement avec les
composants figés, conformément au même principe déjà appliqué à
`RetrievalClient` (`action_client.py`).
"""

from __future__ import annotations

from typing import Protocol

from reasoning.contracts.action_interface import RetrievalResponse, RetrievedChunk
from reasoning.contracts.internal_models import (
    AnalysisResult,
    CriticEvaluation,
    ExecutionPlan,
    PlanStep,
    VerificationResult,
)


class AnalyzerProtocol(Protocol):
    """Interface structurelle de `QueryAnalyzer`."""

    def analyze(self, query: str) -> AnalysisResult:
        """Classifie une requête et retourne un AnalysisResult."""
        ...


class PlannerProtocol(Protocol):
    """Interface structurelle de `Planner`."""

    def decompose(self, query: str, analysis: AnalysisResult) -> ExecutionPlan:
        """Décompose une requête en ExecutionPlan."""
        ...


class CriticProtocol(Protocol):
    """Interface structurelle de `Critic`."""

    max_retries: int

    def evaluate(self, step: PlanStep, response: RetrievalResponse) -> CriticEvaluation:
        """Évalue la qualité du contexte récupéré pour une étape."""
        ...


class VerifierProtocol(Protocol):
    """Interface structurelle de `Verifier`."""

    def verify(self, answer: str, sources: list[RetrievedChunk]) -> VerificationResult:
        """Vérifie la fidélité d'une réponse par rapport aux sources."""
        ...


__all__ = [
    "AnalyzerProtocol",
    "CriticProtocol",
    "PlannerProtocol",
    "VerifierProtocol",
]
