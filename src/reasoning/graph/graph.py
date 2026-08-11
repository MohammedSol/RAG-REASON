"""
Assemblage du graphe LangGraph — module REASONING (Sprint 6).

Ce module est le SEUL point du package `graph/` qui importe `langgraph` (en
dehors de `nodes.py`, qui ne fait qu'appeler des composants purs). La
logique de décision réside intégralement dans `ReasoningPolicy`
(`policy.py`, zéro import LangGraph) — ce fichier ne fait qu'y déléguer en
câblant nœuds et arêtes conformément à docs/graph_spec.md.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from reasoning.action_client import ActionClient, RetrievalClient
from reasoning.analyzer import QueryAnalyzer
from reasoning.critic import Critic
from reasoning.graph.edges import (
    route_after_analysis,
    route_after_critique,
    route_after_verification,
)
from reasoning.graph.nodes import (
    clarify_node,
    make_analyze_node,
    make_critique_node,
    make_generate_answer_node,
    make_plan_node,
    make_retrieve_node,
    make_verify_node,
)
from reasoning.graph.policy import (
    ROUTE_CLARIFY,
    ROUTE_END,
    ROUTE_GENERATE_ANSWER,
    ROUTE_PLAN,
    ROUTE_RETRIEVE,
    ReasoningPolicy,
)
from reasoning.graph.protocols import (
    AnalyzerProtocol,
    CriticProtocol,
    PlannerProtocol,
    VerifierProtocol,
)
from reasoning.graph.state import GraphState
from reasoning.planner import Planner
from reasoning.verifier import Verifier


def build_graph(
    analyzer: AnalyzerProtocol | None = None,
    planner: PlannerProtocol | None = None,
    critic: CriticProtocol | None = None,
    verifier: VerifierProtocol | None = None,
    retrieval_client: RetrievalClient | None = None,
) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    """Construit et compile le graphe LangGraph du module REASONING.

    Tous les composants sont injectables (défaut : instances standard) pour
    permettre la substitution par des doubles de test dans
    `tests/integration/test_graph.py` — notamment `retrieval_client`, le
    module ACTION n'étant pas encore branché (docs/graph_spec.md §7).

    Args:
        analyzer: Instance QueryAnalyzer (défaut : QueryAnalyzer()).
        planner: Instance Planner (défaut : Planner()).
        critic: Instance Critic (défaut : Critic()).
        verifier: Instance Verifier (défaut : Verifier()).
        retrieval_client: Client de retrieval conforme au protocole
            `RetrievalClient` (défaut : ActionClient() — HTTP réel).

    Returns:
        Le graphe LangGraph compilé, prêt à être invoqué via `.invoke()`
        ou `.ainvoke()` avec un `GraphState` initial.
    """
    analyzer = analyzer or QueryAnalyzer()
    planner = planner or Planner()
    critic = critic or Critic()
    verifier = verifier or Verifier()
    retrieval_client = retrieval_client or ActionClient()
    policy = ReasoningPolicy()

    graph: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)

    graph.add_node("analyze_query", make_analyze_node(analyzer, policy))
    graph.add_node("clarify", clarify_node)
    graph.add_node("plan", make_plan_node(planner))
    graph.add_node("retrieve", make_retrieve_node(retrieval_client))
    graph.add_node("critique", make_critique_node(critic, policy))
    graph.add_node("generate_answer", make_generate_answer_node())
    graph.add_node("verify", make_verify_node(verifier, policy))

    graph.set_entry_point("analyze_query")

    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {ROUTE_CLARIFY: "clarify", ROUTE_PLAN: "plan"},
    )
    graph.add_edge("clarify", END)
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {ROUTE_RETRIEVE: "retrieve", ROUTE_GENERATE_ANSWER: "generate_answer"},
    )
    graph.add_edge("generate_answer", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {ROUTE_END: END, ROUTE_PLAN: "plan"},
    )

    return graph.compile()


__all__ = ["build_graph"]
