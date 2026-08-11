"""
Fonctions de routage conditionnel du graphe LangGraph.

Note d'implémentation (docs/graph_spec.md §5.1, §6) : LangGraph n'autorise
pas les fonctions passées à `add_conditional_edges` à muter l'état — elles
ne peuvent que le LIRE et retourner un nom de nœud. La décision de routage
elle-même est calculée en amont par le nœud précédent via `ReasoningPolicy`
(framework-free) et stockée dans `state["next_route"]`. Les fonctions
ci-dessous se contentent donc de relire ce champ, tout en gardant la
responsabilité du nommage des arêtes LangGraph.
"""

from __future__ import annotations

from reasoning.graph.state import GraphState


def route_after_analysis(state: GraphState) -> str:
    """Relit la décision de routage calculée par `analyze_query`."""
    return state["next_route"]


def route_after_critique(state: GraphState) -> str:
    """Relit la décision de routage calculée par `critique`."""
    return state["next_route"]


def route_after_verification(state: GraphState) -> str:
    """Relit la décision de routage calculée par `verify`."""
    return state["next_route"]


__all__ = ["route_after_analysis", "route_after_critique", "route_after_verification"]
