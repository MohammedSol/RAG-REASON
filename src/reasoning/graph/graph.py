"""
Assemblage du graphe LangGraph — module REASONING (Sprint 6).

Ce module est le SEUL point du package `graph/` qui importe `langgraph` (en
dehors de `nodes.py`, qui ne fait qu'appeler des composants purs). La
logique de décision réside intégralement dans `ReasoningPolicy`
(`policy.py`, zéro import LangGraph) — ce fichier ne fait qu'y déléguer en
câblant nœuds et arêtes conformément à docs/graph_spec.md.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from reasoning.action_client import (
    ActionClient,
    ActionUnavailableError,
    RetrievalClient,
)
from reasoning.analyzer import QueryAnalyzer
from reasoning.contracts.action_interface import RetrievalRequest, RetrievalResponse
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

load_dotenv()


def _use_real_action() -> bool:
    """Indique si le nœud `retrieve` doit appeler le module ACTION réel.

    Drapeau `USE_REAL_ACTION`, lu dans l'environnement selon le même
    mécanisme que `OLLAMA_BASE_URL` ou `DEFAULT_REASONING_MODEL` — aucun
    système de configuration parallèle n'est introduit.

    **Valeur par défaut : FAUX.** Tant que le drapeau n'est pas positionné,
    le graphe se comporte exactement comme avant le Sprint I2 : aucun appel
    réseau n'est tenté, et il faut injecter explicitement un client (double
    de test ou `ActionClient`) via `build_graph(retrieval_client=...)`.
    Cette valeur par défaut garantit qu'aucune suite de tests, ni aucun
    usage existant, ne se met soudainement à dépendre d'un service externe.

    Returns:
        True uniquement si `USE_REAL_ACTION` vaut 1/true/yes/on
        (comparaison insensible à la casse).
    """
    return os.getenv("USE_REAL_ACTION", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class _UnconfiguredRetrievalClient:
    """Client de repli, actif quand `USE_REAL_ACTION` est faux.

    Échoue explicitement plutôt que de retourner un résultat vide : un
    graphe exécuté sans client de retrieval configuré est une erreur de
    câblage, pas un cas métier. Le nœud `retrieve` intercepte cette
    exception comme n'importe quelle défaillance et applique son repli
    fail-closed, mais le message rend la cause immédiatement lisible.
    """

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Lève systématiquement : aucun client n'a été configuré."""
        raise ActionUnavailableError(
            "aucun client de retrieval configuré : USE_REAL_ACTION est faux "
            "et aucun client n'a été injecté via build_graph(retrieval_client=...). "
            "Positionner USE_REAL_ACTION=true pour appeler le module ACTION."
        )


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
    `tests/integration/test_graph.py`.

    Args:
        analyzer: Instance QueryAnalyzer (défaut : QueryAnalyzer()).
        planner: Instance Planner (défaut : Planner()).
        critic: Instance Critic (défaut : Critic()).
        verifier: Instance Verifier (défaut : Verifier()).
        retrieval_client: Client de retrieval conforme au protocole
            `RetrievalClient`. Si omis, le choix dépend du drapeau
            `USE_REAL_ACTION` : `ActionClient` (module ACTION réel) quand il
            est vrai, sinon un repli inerte qui échoue explicitement sans
            tenter le moindre appel réseau. **Le défaut est le repli inerte**
            — aucun usage existant ne devient dépendant d'un service externe
            du seul fait du Sprint I2.

    Returns:
        Le graphe LangGraph compilé, prêt à être invoqué via `.invoke()`
        ou `.ainvoke()` avec un `GraphState` initial.
    """
    analyzer = analyzer or QueryAnalyzer()
    planner = planner or Planner()
    critic = critic or Critic()
    verifier = verifier or Verifier()
    # Sélection du client de retrieval :
    #   1. client injecté explicitement (double de test, ou ActionClient) ;
    #   2. sinon USE_REAL_ACTION=true  -> ActionClient (module ACTION réel) ;
    #   3. sinon                        -> repli inerte, aucun appel réseau.
    if retrieval_client is None:
        retrieval_client = (
            ActionClient() if _use_real_action() else _UnconfiguredRetrievalClient()
        )
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
