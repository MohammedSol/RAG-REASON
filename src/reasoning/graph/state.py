"""
État d'exécution du graphe LangGraph — `GraphState`.

Conforme à la décision de conception documentée dans `docs/graph_spec.md
§5.1` : `AgentState` (contrat Pydantic figé, `contracts/internal_models.py`)
reste la SEULE source de vérité pour les champs partagés avec le reste du
module REASONING (analysis, plan, evaluations, verification,
feedback_loop_count). `GraphState` est un `TypedDict` d'exécution
**propre à LangGraph** qui enveloppe `AgentState` sans le dupliquer, et
ajoute uniquement les champs de plomberie nécessaires au graphe
(accumulation multi-hop, compteur local de retries par étape).
"""

from __future__ import annotations

from typing import TypedDict

from reasoning.contracts.action_interface import RetrievalResponse, RetrievedChunk
from reasoning.contracts.internal_models import AgentState


class GraphState(TypedDict):
    """État complet circulant entre les nœuds du graphe LangGraph.

    Note d'implémentation (docs/graph_spec.md §6) : les fonctions de
    routage conditionnel de `edges.py` ne peuvent pas muter l'état
    (contrainte LangGraph — le callable `path` d'`add_conditional_edges`
    ne fait que LIRE l'état et retourner un nom de nœud). La décision de
    routage est donc calculée par le NŒUD précédent (qui, lui, peut muter
    l'état) via `ReasoningPolicy`, et stockée dans `next_route`. La
    fonction d'arête correspondante se contente de relire ce champ.

    Attributes:
        agent_state: Le contrat `AgentState` figé — original_query, analysis,
            plan, evaluations, verification, feedback_loop_count (le budget
            global unique, cf. docs/graph_spec.md §3).
        retrieved_chunks: Chunks accumulés sur tous les hops du plan
            d'exécution — consommés par le nœud `generate_answer` et le
            `Verifier` (qui a besoin de la liste complète des sources).
        retry_counts: Compteur LOCAL de re-retrievals par `step_id`,
            comparé à `Critic.max_retries` (garde locale, cf. §3.3).
        pending_step_ids: File d'attente des `step_id` restant à traiter,
            triée topologiquement (Kahn) une fois lors du nœud `plan`.
        current_step_id: L'identifiant de la `PlanStep` en cours de
            traitement par `retrieve`/`critique`, ou `None` si aucune étape
            n'est encore active.
        last_retrieval_response: La `RetrievalResponse` du dernier appel
            `retrieve`, consommée immédiatement par `critique`.
        answer: La réponse candidate produite par `generate_answer`, ou
            `None` avant son exécution.
        next_route: Nom du prochain nœud, calculé par le nœud courant via
            `ReasoningPolicy` — relu tel quel par la fonction d'arête
            correspondante dans `edges.py`.
    """

    agent_state: AgentState
    retrieved_chunks: list[RetrievedChunk]
    retry_counts: dict[str, int]
    pending_step_ids: list[str]
    current_step_id: str | None
    last_retrieval_response: RetrievalResponse | None
    answer: str | None
    next_route: str


def build_initial_state(original_query: str) -> GraphState:
    """Construit l'état initial du graphe pour une nouvelle requête.

    Args:
        original_query: La requête utilisateur brute.

    Returns:
        GraphState initialisé — `agent_state` vierge (analysis=None,
        plan=None, etc.), aucun chunk accumulé, aucun retry consommé.
    """
    return GraphState(
        agent_state=AgentState(original_query=original_query),
        retrieved_chunks=[],
        retry_counts={},
        pending_step_ids=[],
        current_step_id=None,
        last_retrieval_response=None,
        answer=None,
        next_route="",
    )


__all__ = ["GraphState", "build_initial_state"]
