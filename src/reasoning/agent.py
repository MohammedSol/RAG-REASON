"""
Point d'entrée public du module REASONING — Sprint 6.

Expose `run_agent(query: str) -> VerificationResult`, conformément à
`plan.md §6.2`. Assemble le graphe LangGraph complet (`build_graph`) et
l'exécute de bout en bout pour une requête utilisateur donnée.
"""

from __future__ import annotations

from typing import cast

from reasoning.action_client import RetrievalClient
from reasoning.contracts.internal_models import VerificationResult
from reasoning.graph.graph import build_graph
from reasoning.graph.protocols import (
    AnalyzerProtocol,
    CriticProtocol,
    PlannerProtocol,
    VerifierProtocol,
)
from reasoning.graph.state import GraphState, build_initial_state


async def run_agent(
    query: str,
    *,
    analyzer: AnalyzerProtocol | None = None,
    planner: PlannerProtocol | None = None,
    critic: CriticProtocol | None = None,
    verifier: VerifierProtocol | None = None,
    retrieval_client: RetrievalClient | None = None,
) -> VerificationResult:
    """Exécute le graphe REASONING complet pour une requête utilisateur.

    Args:
        query: La requête utilisateur brute.
        analyzer: Composant injectable (défaut : instance standard).
        planner: Composant injectable (défaut : instance standard).
        critic: Composant injectable (défaut : instance standard).
        verifier: Composant injectable (défaut : instance standard).
        retrieval_client: Client de retrieval injectable — utile pour les
            tests (double en mémoire) tant que le module ACTION n'est pas
            branché (défaut : `ActionClient()`, HTTP réel).

    Returns:
        Le `VerificationResult` final — `is_grounded`, `faithfulness_score`,
        `unsupported_claims`, `final_answer`.

    Raises:
        RuntimeError: Si le graphe se termine sans qu'aucune vérification
            n'ait été produite (état incohérent — ne devrait jamais se
            produire dans un flux valide).
    """
    graph = build_graph(
        analyzer=analyzer,
        planner=planner,
        critic=critic,
        verifier=verifier,
        retrieval_client=retrieval_client,
    )

    initial_state = build_initial_state(query)
    raw_final_state = await graph.ainvoke(initial_state)
    final_state = cast(GraphState, raw_final_state)

    verification = final_state["agent_state"].verification
    if verification is None:
        raise RuntimeError(
            "Le graphe REASONING s'est terminé sans produire de "
            "VerificationResult — état incohérent."
        )
    return verification


__all__ = ["run_agent"]
