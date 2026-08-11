"""
ReasoningPolicy — Logique de routage pure du graphe REASONING.

Contrainte d'architecture non négociable (cf. plan.md et la mission) :
    Ce module ne doit JAMAIS importer le paquet d'orchestration externe.
    Il prend en entrée des valeurs primitives (jamais un objet du
    framework de graphe), et retourne un nom de nœud sous forme de
    chaîne. LangGraph n'est qu'un adaptateur d'exécution appelant ces
    méthodes depuis `graph/nodes.py`.

    Vérifiable : une recherche du nom du paquet d'orchestration (tout en
    minuscules) dans ce fichier ne doit renvoyer aucune occurrence.

Cf. docs/graph_spec.md §3 et §6 pour la spécification complète du budget
global unique et de sa composition avec la garde locale `max_retries`.
"""

from __future__ import annotations

from typing import NamedTuple

# Noms de nœuds — chaînes partagées entre policy.py, nodes.py et graph.py.
ROUTE_CLARIFY = "clarify"
ROUTE_PLAN = "plan"
ROUTE_RETRIEVE = "retrieve"
ROUTE_GENERATE_ANSWER = "generate_answer"
ROUTE_END = "end"


class CritiqueDecision(NamedTuple):
    """Décision de routage après `critique`, avec l'effet de bord à appliquer.

    Attributes:
        route: Nom du prochain nœud (`ROUTE_RETRIEVE` ou `ROUTE_GENERATE_ANSWER`).
        advance_step: `True` si l'étape courante doit être retirée de la file
            d'attente (`pending_step_ids`) — contexte suffisant, garde locale
            ou garde globale déclenchée. `False` si l'étape doit être
            re-tentée (re-retrieval sur la même étape).
    """

    route: str
    advance_step: bool


class ReasoningPolicy:
    """Décisions de routage pures pour le graphe REASONING.

    Aucune méthode ne lit ni ne modifie d'état LangGraph : chaque méthode
    reçoit les valeurs strictement nécessaires à la décision et retourne un
    nom de nœud. Cette classe est testable unitairement sans LangGraph ni
    LLM.
    """

    def route_after_analysis(self, *, reasoning_budget: int) -> str:
        """Décide du routage après `analyze_query`.

        Args:
            reasoning_budget: Le budget alloué par l'Analyzer.
                `0` signifie `QueryType.AMBIGUOUS` (analyzer_spec.md §2.1).

        Returns:
            `ROUTE_CLARIFY` si aucun retrieval n'est autorisé, `ROUTE_PLAN`
            sinon (tous les autres types passent par le Planner — y compris
            SIMPLE, qui court-circuite lui-même tout appel LLM, cf.
            docs/graph_spec.md §4.1).
        """
        if reasoning_budget <= 0:
            return ROUTE_CLARIFY
        return ROUTE_PLAN

    def route_after_critique(
        self,
        *,
        reasoning_budget: int,
        feedback_loop_count: int,
        is_sufficient: bool,
        retry_count: int,
        max_retries: int,
        has_next_step: bool,
    ) -> CritiqueDecision:
        """Décide du routage après `critique` (cf. docs/graph_spec.md §3.3).

        La garde globale (budget épuisé) est TOUJOURS prioritaire sur la
        garde locale (`max_retries`) et sur le verdict `is_sufficient`.

        Args:
            reasoning_budget: Plafond global (`AgentState.analysis.reasoning_budget`).
            feedback_loop_count: Compteur global déjà incrémenté pour ce
                passage par `critique` (cf. §3.2 — incrément fait par le nœud
                appelant AVANT cet appel).
            is_sufficient: Verdict du Critic pour l'étape courante.
            retry_count: Nombre de re-retrievals DÉJÀ effectués pour cette
                étape (avant la décision courante).
            max_retries: Plafond local du Critic pour cette étape.
            has_next_step: `True` s'il reste au moins une autre étape dans
                le plan après l'étape courante.

        Returns:
            `CritiqueDecision(route=ROUTE_GENERATE_ANSWER, advance_step=True)`
            si le budget global est épuisé, ou si le plan est épuisé
            (contexte suffisant ou retries épuisés sans étape suivante).
            `CritiqueDecision(route=ROUTE_RETRIEVE, advance_step=True)` pour
            avancer vers l'étape suivante. `CritiqueDecision(route=ROUTE_RETRIEVE,
            advance_step=False)` pour re-tenter la même étape.
        """
        if feedback_loop_count >= reasoning_budget:
            return CritiqueDecision(route=ROUTE_GENERATE_ANSWER, advance_step=True)

        if is_sufficient:
            route = ROUTE_RETRIEVE if has_next_step else ROUTE_GENERATE_ANSWER
            return CritiqueDecision(route=route, advance_step=True)

        if retry_count >= max_retries:
            route = ROUTE_RETRIEVE if has_next_step else ROUTE_GENERATE_ANSWER
            return CritiqueDecision(route=route, advance_step=True)

        return CritiqueDecision(route=ROUTE_RETRIEVE, advance_step=False)

    def route_after_verification(
        self,
        *,
        reasoning_budget: int,
        feedback_loop_count: int,
        is_grounded: bool,
    ) -> str:
        """Décide du routage après `verify` (cf. docs/graph_spec.md §4.6).

        Args:
            reasoning_budget: Plafond global.
            feedback_loop_count: Compteur global déjà incrémenté pour ce
                passage par `verify`.
            is_grounded: Verdict du Verifier.

        Returns:
            `ROUTE_END` si la réponse est fondée OU si le budget global est
            épuisé (garde globale — sortie forcée, réponse potentiellement
            non-garantie fondée mais `VerificationResult.is_grounded=False`
            reste visible par l'appelant). `ROUTE_PLAN` sinon
            (re-planification).
        """
        if is_grounded:
            return ROUTE_END
        if feedback_loop_count >= reasoning_budget:
            return ROUTE_END
        return ROUTE_PLAN


__all__ = [
    "ROUTE_CLARIFY",
    "ROUTE_END",
    "ROUTE_GENERATE_ANSWER",
    "ROUTE_PLAN",
    "ROUTE_RETRIEVE",
    "ReasoningPolicy",
]
