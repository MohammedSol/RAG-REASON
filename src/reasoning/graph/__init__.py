"""Package Graph du module REASONING — orchestration LangGraph (Sprint 6).

Expose `build_graph`, qui assemble et compile le graphe d'état complet
(Query Analyzer → Planner → retrieve → Critic → generate_answer → Verifier),
conformément à docs/graph_spec.md.

Usage :
    from reasoning.graph import build_graph
    from reasoning.graph.state import build_initial_state

    graph = build_graph()
    result_state = await graph.ainvoke(build_initial_state("ma requête"))
"""

from reasoning.graph.graph import build_graph

__all__ = ["build_graph"]
