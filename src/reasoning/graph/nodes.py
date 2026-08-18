"""
Fonctions de nœuds du graphe LangGraph — module REASONING (Sprint 6).

Chaque nœud est produit par une fonction fabrique (`make_*_node`) qui reçoit
les composants injectés (Analyzer, Planner, Critic, Verifier, client de
retrieval), permettant l'injection de doubles de test sans dépendance à
Ollama ni à un serveur ACTION réel (cf. docs/graph_spec.md §7).

Les nœuds appellent des composants **purs et sans état** (règle non
négociable de la mission) : ils ne font que lire l'état d'entrée, appeler le
composant, et retourner une mise à jour partielle de `GraphState`. Toute
décision de routage est déléguée à `ReasoningPolicy` (framework-free) — les
nœuds ne font qu'appliquer les effets de bord (avancer la file d'attente,
incrémenter les compteurs) cohérents avec la décision retournée par la
policy, car les fonctions d'arête de LangGraph ne peuvent pas muter l'état
(cf. docs/graph_spec.md §5.1, note sur `next_route`).
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any, Protocol

import httpx
from dotenv import load_dotenv
from litellm import completion
from pydantic import ValidationError

from reasoning.action_client import ActionClientError, RetrievalClient
from reasoning.contracts.action_interface import RetrievalRequest, RetrievalResponse
from reasoning.contracts.internal_models import ExecutionPlan, VerificationResult
from reasoning.graph.policy import ReasoningPolicy
from reasoning.graph.protocols import (
    AnalyzerProtocol,
    CriticProtocol,
    PlannerProtocol,
    VerifierProtocol,
)
from reasoning.graph.state import GraphState

load_dotenv()

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_REASONING_MODEL: str = os.getenv(
    "DEFAULT_REASONING_MODEL", "ollama/qwen2.5:7b"
)

# Nombre de chunks demandés par retrieval (paramètre par défaut du nœud retrieve).
_DEFAULT_TOP_K: int = 5

# Nombre maximum de caractères par chunk injecté dans le prompt de synthèse.
_MAX_CHUNK_CHARS: int = 600


class NodeFn(Protocol):
    """Signature structurelle d'un nœud LangGraph (compatible `add_node`).

    Un `Protocol` est utilisé plutôt qu'un alias `Callable[...]` car mypy
    strict ne parvient pas à résoudre la surcharge générique `add_node` de
    LangGraph (`StateNode[NodeInputT]`, une union de Protocols) face à un
    alias `Callable` — limitation connue de l'inférence de surcharge de
    mypy sur les unions de Protocols génériques. Un Protocol structurel
    résout correctement l'appariement.
    """

    async def __call__(self, state: GraphState) -> dict[str, Any]: ...


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────


def _topological_order(plan: ExecutionPlan) -> list[str]:
    """Calcule l'ordre topologique (Kahn) des `step_id` d'un plan validé.

    Le plan est supposé déjà validé comme DAG acyclique par le Planner
    (`_validate_dag`, planner_spec.md §9) — cette fonction ne revalide pas
    l'absence de cycle, elle produit uniquement un ordre d'exécution
    séquentiel cohérent avec les dépendances.

    Args:
        plan: L'ExecutionPlan dont on calcule l'ordre.

    Returns:
        Liste de `step_id` dans un ordre respectant `depends_on`.
    """
    in_degree: dict[str, int] = {s.step_id: 0 for s in plan.steps}
    adjacency: dict[str, list[str]] = {s.step_id: [] for s in plan.steps}

    for step in plan.steps:
        for dep in step.depends_on:
            if dep not in in_degree:
                continue
            adjacency[dep].append(step.step_id)
            in_degree[step.step_id] += 1

    queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    order: list[str] = []

    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for neighbor in adjacency[node_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order


def _empty_response(query_id: str) -> RetrievalResponse:
    """Construit une RetrievalResponse défensive vide (échec réseau ACTION)."""
    return RetrievalResponse(query_id=query_id, chunks=[], retrieval_score=None)


def _format_chunks_for_generation(chunks: list[Any]) -> str:
    """Formate les chunks accumulés pour le prompt de synthèse finale."""
    lines: list[str] = []
    for chunk in chunks:
        content = chunk.content
        if len(content) > _MAX_CHUNK_CHARS:
            content = content[:_MAX_CHUNK_CHARS] + "... [truncated]"
        lines.append(f"  [source={chunk.source}]: {content}")
    return "\n".join(lines) if lines else "  (no context retrieved)"


# ─────────────────────────────────────────────────────────────────────────────
# Nœud : analyze_query
# ─────────────────────────────────────────────────────────────────────────────


def make_analyze_node(
    analyzer: AnalyzerProtocol,
    policy: ReasoningPolicy,
) -> NodeFn:
    """Fabrique le nœud `analyze_query`."""

    async def _node(state: GraphState) -> dict[str, Any]:
        agent_state = state["agent_state"]
        analysis = analyzer.analyze(agent_state.original_query)
        new_agent_state = agent_state.model_copy(update={"analysis": analysis})
        next_route = policy.route_after_analysis(
            reasoning_budget=analysis.reasoning_budget
        )
        return {"agent_state": new_agent_state, "next_route": next_route}

    return _node


# ─────────────────────────────────────────────────────────────────────────────
# Nœud : clarify (AMBIGUOUS — court-circuit sans retrieval)
# ─────────────────────────────────────────────────────────────────────────────


async def clarify_node(state: GraphState) -> dict[str, Any]:
    """Nœud terminal pour les requêtes AMBIGUOUS (reasoning_budget=0).

    Ne fait aucun appel LLM ni retrieval (analyzer_spec.md §2.1). Construit
    directement un VerificationResult demandant une clarification.
    """
    agent_state = state["agent_state"]
    clarification = (
        "Your query is ambiguous or too broad to answer reliably. "
        "Please clarify the specific entity, scope, or intent of your question."
    )
    verification = VerificationResult(
        is_grounded=False,
        faithfulness_score=0.0,
        unsupported_claims=["ambiguous_query"],
        final_answer=clarification,
    )
    new_agent_state = agent_state.model_copy(update={"verification": verification})
    return {"agent_state": new_agent_state}


# ─────────────────────────────────────────────────────────────────────────────
# Nœud : plan
# ─────────────────────────────────────────────────────────────────────────────


def make_plan_node(planner: PlannerProtocol) -> NodeFn:
    """Fabrique le nœud `plan`.

    Gère la re-planification (docs/graph_spec.md §4.7) : si un
    `VerificationResult` insatisfaisant est déjà présent dans l'état (retour
    depuis `verify`), la requête envoyée au Planner est augmentée du
    feedback (`unsupported_claims`) sans modifier `planner.py` (composant
    figé).
    """

    async def _node(state: GraphState) -> dict[str, Any]:
        agent_state = state["agent_state"]
        if agent_state.analysis is None:
            raise RuntimeError(
                "plan_node appelé sans AnalysisResult préalable "
                "(violation du flux du graphe)."
            )

        query = agent_state.original_query
        verification = agent_state.verification
        if (
            verification is not None
            and not verification.is_grounded
            and verification.unsupported_claims
        ):
            claims = ", ".join(verification.unsupported_claims)
            query = (
                f"{agent_state.original_query} "
                f"(Additional verification needed on: {claims})"
            )

        plan = planner.decompose(query, agent_state.analysis)
        order = _topological_order(plan)

        new_agent_state = agent_state.model_copy(update={"plan": plan})
        return {
            "agent_state": new_agent_state,
            "pending_step_ids": order,
            "current_step_id": None,
            "retry_counts": {},
        }

    return _node


# ─────────────────────────────────────────────────────────────────────────────
# Nœud : retrieve
# ─────────────────────────────────────────────────────────────────────────────


def make_retrieve_node(client: RetrievalClient, top_k: int = _DEFAULT_TOP_K) -> NodeFn:
    """Fabrique le nœud `retrieve` — client HTTP vers le module ACTION.

    Repli fail-closed (docs/graph_spec.md §7) : toute erreur réseau ou de
    validation produit une `RetrievalResponse` vide plutôt qu'un crash —
    le `Critic` traite alors le cas "aucun chunk" déjà spécifié.
    """

    async def _node(state: GraphState) -> dict[str, Any]:
        agent_state = state["agent_state"]
        plan = agent_state.plan
        pending = state["pending_step_ids"]
        if plan is None or not pending:
            raise RuntimeError(
                "retrieve_node appelé sans plan actif ou file d'étapes vide."
            )

        step_id = pending[0]
        step = next((s for s in plan.steps if s.step_id == step_id), None)
        if step is None:
            raise RuntimeError(f"PlanStep '{step_id}' introuvable dans le plan.")

        hop_index = plan.steps.index(step)
        request = RetrievalRequest(
            query_id=plan.plan_id,
            sub_query=step.sub_query,
            hop_index=hop_index,
            top_k=top_k,
        )

        try:
            response = client.retrieve(request)
        except ActionClientError as exc:
            # Défaillance ATTENDUE et déjà qualifiée par le client (Sprint I2) :
            # module injoignable, code HTTP non-2xx, corps non conforme ou
            # query_id non corrélé. Ces cas font partie du fonctionnement
            # normal d'un appel réseau — les journaliser en ERROR noierait les
            # anomalies véritablement imprévues dans le bruit d'exploitation.
            logger.warning(
                "retrieve[%s] : module ACTION injoignable ou en erreur "
                "(%s: %s) — réponse vide défensive.",
                step_id,
                type(exc).__name__,
                exc,
            )
            response = _empty_response(plan.plan_id)
        except (httpx.HTTPError, OSError, TimeoutError) as exc:
            logger.warning(
                "retrieve[%s] : erreur réseau (%s: %s) — réponse vide défensive.",
                step_id,
                type(exc).__name__,
                exc,
            )
            response = _empty_response(plan.plan_id)
        except ValidationError as exc:
            logger.warning(
                "retrieve[%s] : réponse ACTION invalide (%s) — réponse vide défensive.",
                step_id,
                exc,
            )
            response = _empty_response(plan.plan_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "retrieve[%s] : erreur inattendue (%s: %s) — réponse vide défensive.",
                step_id,
                type(exc).__name__,
                exc,
            )
            response = _empty_response(plan.plan_id)

        new_chunks = [*state["retrieved_chunks"], *response.chunks]
        return {
            "current_step_id": step_id,
            "last_retrieval_response": response,
            "retrieved_chunks": new_chunks,
        }

    return _node


# ─────────────────────────────────────────────────────────────────────────────
# Nœud : critique
# ─────────────────────────────────────────────────────────────────────────────


def make_critique_node(critic: CriticProtocol, policy: ReasoningPolicy) -> NodeFn:
    """Fabrique le nœud `critique`.

    Applique la décision de `ReasoningPolicy.route_after_critique` (budget
    global unique + garde locale `max_retries`, docs/graph_spec.md §3) et
    met à jour la file d'attente / le compteur de retries en conséquence.
    """

    async def _node(state: GraphState) -> dict[str, Any]:
        agent_state = state["agent_state"]
        response = state["last_retrieval_response"]
        step_id = state["current_step_id"]
        plan = agent_state.plan

        if response is None or step_id is None or plan is None:
            raise RuntimeError(
                "critique_node appelé hors séquence (retrieve non exécuté)."
            )

        step = next((s for s in plan.steps if s.step_id == step_id), None)
        if step is None:
            raise RuntimeError(f"PlanStep '{step_id}' introuvable dans le plan.")

        evaluation = critic.evaluate(step, response)
        new_evaluations = [*agent_state.evaluations, evaluation]
        new_feedback_loop_count = agent_state.feedback_loop_count + 1

        reasoning_budget = (
            agent_state.analysis.reasoning_budget if agent_state.analysis else 0
        )
        pending = state["pending_step_ids"]
        retry_count = state["retry_counts"].get(step_id, 0)
        has_next_step = len(pending) > 1

        decision = policy.route_after_critique(
            reasoning_budget=reasoning_budget,
            feedback_loop_count=new_feedback_loop_count,
            is_sufficient=evaluation.is_sufficient,
            retry_count=retry_count,
            max_retries=critic.max_retries,
            has_next_step=has_next_step,
        )

        new_agent_state = agent_state.model_copy(
            update={
                "evaluations": new_evaluations,
                "feedback_loop_count": new_feedback_loop_count,
            }
        )

        new_retry_counts = dict(state["retry_counts"])
        new_pending = list(pending)

        if decision.advance_step:
            if new_pending and new_pending[0] == step_id:
                new_pending.pop(0)
            new_retry_counts.pop(step_id, None)
        else:
            new_retry_counts[step_id] = retry_count + 1

        return {
            "agent_state": new_agent_state,
            "retry_counts": new_retry_counts,
            "pending_step_ids": new_pending,
            "next_route": decision.route,
        }

    return _node


# ─────────────────────────────────────────────────────────────────────────────
# Nœud : generate_answer
# ─────────────────────────────────────────────────────────────────────────────

_GENERATION_PROMPT: str = (
    "You are a helpful assistant answering questions using ONLY the "
    "provided context. If the context is insufficient, say so plainly — "
    "do not invent information.\n\n"
    "Question:\n  {query}\n\n"
    "Context:\n{context}\n\n"
    "Answer the question concisely, in English, using only the context above."
)


def make_generate_answer_node(
    model: str = _DEFAULT_REASONING_MODEL,
    api_base: str = _OLLAMA_BASE_URL,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> NodeFn:
    """Fabrique le nœud `generate_answer`.

    Décision de conception (docs/graph_spec.md §2.1) : ce nœud n'est pas un
    composant dédié à sa propre spec — il appelle directement le LLM et
    retourne le texte brut comme réponse candidate. Pas de parsing TOON :
    la sortie EST le résultat final, il n'y a aucun champ structuré à en
    extraire (contrairement à Analyzer/Planner/Critic/Verifier).
    """

    async def _node(state: GraphState) -> dict[str, Any]:
        agent_state = state["agent_state"]
        context = _format_chunks_for_generation(state["retrieved_chunks"])
        prompt = _GENERATION_PROMPT.format(
            query=agent_state.original_query, context=context
        )

        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_base=api_base,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                answer = "I could not generate an answer from the available context."
        except (OSError, TimeoutError, RuntimeError) as exc:
            logger.warning(
                "generate_answer : erreur réseau/LLM (%s: %s) — réponse de repli.",
                type(exc).__name__,
                exc,
            )
            answer = "I could not generate an answer due to a technical failure."
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "generate_answer : erreur inattendue (%s: %s) — réponse de repli.",
                type(exc).__name__,
                exc,
            )
            answer = "I could not generate an answer due to a technical failure."

        return {"answer": answer}

    return _node


# ─────────────────────────────────────────────────────────────────────────────
# Nœud : verify
# ─────────────────────────────────────────────────────────────────────────────


def make_verify_node(verifier: VerifierProtocol, policy: ReasoningPolicy) -> NodeFn:
    """Fabrique le nœud `verify`."""

    async def _node(state: GraphState) -> dict[str, Any]:
        agent_state = state["agent_state"]
        answer = state["answer"] or ""
        sources = state["retrieved_chunks"]

        verification = verifier.verify(answer, sources)
        new_feedback_loop_count = agent_state.feedback_loop_count + 1

        reasoning_budget = (
            agent_state.analysis.reasoning_budget if agent_state.analysis else 0
        )
        next_route = policy.route_after_verification(
            reasoning_budget=reasoning_budget,
            feedback_loop_count=new_feedback_loop_count,
            is_grounded=verification.is_grounded,
        )

        new_agent_state = agent_state.model_copy(
            update={
                "verification": verification,
                "feedback_loop_count": new_feedback_loop_count,
            }
        )
        return {"agent_state": new_agent_state, "next_route": next_route}

    return _node


__all__ = [
    "clarify_node",
    "make_analyze_node",
    "make_critique_node",
    "make_generate_answer_node",
    "make_plan_node",
    "make_retrieve_node",
    "make_verify_node",
]
