"""
Planner — Composant de décomposition Plan-and-Solve du module REASONING.

Ce composant est le deuxième nœud du graphe de raisonnement. Il reçoit
une requête utilisateur et son `AnalysisResult` (produit par le Query
Analyzer) et produit un `ExecutionPlan` : un graphe acyclique dirigé (DAG)
de sous-requêtes atomiques, chacune destinée au module ACTION (retrieval).

Architecture (conforme à docs/planner_spec.md) :
    Niveau 0 : Court-circuit Python si QueryType.SIMPLE (0 appel LLM)
    Niveau 1 : Décomposition LLM via LiteLLM → Ollama (DEFAULT_REASONING_MODEL)
    Niveau 2 : Fallback séquentiel Python si le LLM échoue ou génère un DAG invalide

Modèle utilisé : Qwen 2.5 7B (plus capable que le 3B de l'Analyzer) car la
décomposition multi-hop est cognitivement plus exigeante que la classification.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import deque

from dotenv import load_dotenv
from litellm import completion
from pydantic import ValidationError

from reasoning.contracts.internal_models import (
    AnalysisResult,
    ExecutionPlan,
    PlanStep,
    QueryType,
    StepStatus,
)
from reasoning.planner.prompts import PLANNING_PROMPT
from reasoning.shared.toon_utils import ToonParseError, parse_toon_to_dict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_REASONING_MODEL: str = os.getenv(
    "DEFAULT_REASONING_MODEL", "ollama/qwen2.5:7b"
)

# Nombre maximum de tokens pour la réponse du Planner (plan multi-étapes)
_MAX_TOKENS: int = 1024


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────


def _generate_plan_id(query: str) -> str:
    """Génère un identifiant de plan court et reproductible depuis la requête.

    Args:
        query: La requête utilisateur originale.

    Returns:
        Un identifiant hexadécimal de 12 caractères, préfixé par ``plan-``.
    """
    digest = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()
    return f"plan-{digest[:12]}"


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────


class Planner:
    """Composant Plan-and-Solve qui décompose une requête complexe en DAG.

    Utilise une architecture à trois niveaux :
    - Niveau 0 : Court-circuit Python pour les requêtes SIMPLE (sans LLM)
    - Niveau 1 : Appel LLM (Qwen 7B) pour MULTI_HOP et COMPARATIVE
    - Niveau 2 : Fallback séquentiel si le LLM génère un plan invalide

    Attributes:
        model: Identifiant LiteLLM du modèle de raisonnement (Qwen 7B).
        api_base: URL de base de l'API Ollama.
        max_tokens: Limite de tokens pour la réponse LLM.
        temperature: Température de génération (0.0 = déterministe).
    """

    def __init__(
        self,
        model: str = _DEFAULT_REASONING_MODEL,
        api_base: str = _OLLAMA_BASE_URL,
        max_tokens: int = _MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ── API publique ──────────────────────────────────────────────────────────

    def decompose(self, query: str, analysis: AnalysisResult) -> ExecutionPlan:
        """Point d'entrée du Planner : produit un ExecutionPlan depuis une requête.

        Applique la règle de court-circuit : si la requête est SIMPLE, retourne
        immédiatement un plan mono-étape sans appel LLM. Sinon, délègue au LLM.

        Args:
            query: La requête utilisateur originale.
            analysis: L'AnalysisResult produit par le Query Analyzer.

        Returns:
            Un ExecutionPlan valide (DAG acyclique, budget respecté).
        """
        if analysis.query_type == QueryType.SIMPLE:
            logger.info("Court-circuit SIMPLE : plan mono-étape sans appel LLM.")
            return self._build_simple_plan(query)

        logger.info(
            "Décomposition LLM pour query_type=%s budget=%d",
            analysis.query_type,
            analysis.reasoning_budget,
        )
        try:
            return self._plan_with_llm(query, analysis)
        except (OSError, TimeoutError, RuntimeError) as exc:
            logger.warning(
                "Erreur réseau/LLM (%s: %s) — fallback séquentiel.",
                type(exc).__name__,
                exc,
            )
        except (ToonParseError, ValidationError, ValueError) as exc:
            logger.warning(
                "Réponse LLM non parseable (%s: %s) — fallback séquentiel.",
                type(exc).__name__,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Erreur inattendue (%s: %s) — fallback séquentiel.",
                type(exc).__name__,
                exc,
            )

        return self._build_sequential_fallback(query, analysis)

    # ── Niveau 0 : Court-circuit SIMPLE ──────────────────────────────────────

    @staticmethod
    def _build_simple_plan(query: str) -> ExecutionPlan:
        """Construit un ExecutionPlan mono-étape sans appel LLM.

        Utilisé exclusivement pour les requêtes de type SIMPLE.

        Args:
            query: La requête utilisateur originale.

        Returns:
            ExecutionPlan contenant une unique PlanStep sans dépendances.
        """
        step = PlanStep(
            step_id="step_1",
            sub_query=query,
            depends_on=[],
            status=StepStatus.PENDING,
        )
        return ExecutionPlan(
            plan_id=_generate_plan_id(query),
            original_query=query,
            steps=[step],
            dependencies_graph={"step_1": []},
        )

    # ── Niveau 1 : Décomposition LLM ─────────────────────────────────────────

    def _plan_with_llm(self, query: str, analysis: AnalysisResult) -> ExecutionPlan:
        """Appelle le LLM pour décomposer la requête et parse la réponse TOON.

        Args:
            query: La requête utilisateur originale.
            analysis: L'AnalysisResult contenant le budget et les entités.

        Returns:
            ExecutionPlan parsé, validé et conforme au budget.

        Raises:
            ToonParseError: Si la réponse ne contient aucun bloc TOON parseable.
            ValidationError: Si un bloc TOON ne satisfait pas le contrat Pydantic.
            ValueError: Si aucune étape n'est extraite de la réponse.
            OSError | TimeoutError: Si Ollama est inaccessible.
        """
        prompt = PLANNING_PROMPT.format(
            query=query,
            reasoning_budget=analysis.reasoning_budget,
        )

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        raw_content: str = response.choices[0].message.content or ""
        logger.debug("Réponse brute du Planner LLM : %s", raw_content)

        return self._parse_toon_plan(raw_content, query, analysis)

    # ── Parsing de la réponse TOON ────────────────────────────────────────────

    def _parse_toon_plan(
        self, raw: str, query: str, analysis: AnalysisResult
    ) -> ExecutionPlan:
        """Parse une réponse LLM brute en ExecutionPlan via les blocs TOON.

        Extrait tous les blocs ``<<<...>>>`` de la réponse. Le premier bloc est
        traité comme l'en-tête du plan (`plan_rationale`, `total_steps`), les
        blocs suivants comme des étapes (`step_id`, `sub_query`, `depends_on`).

        Les étapes excédentaires (au-delà du budget) sont tronquées avec un
        avertissement. Si le DAG résultant est invalide (cycle ou référence
        inconnue), bascule sur le fallback séquentiel.

        Args:
            raw: La chaîne brute retournée par le LLM.
            query: La requête utilisateur originale (pour le plan_id).
            analysis: L'AnalysisResult pour récupérer le budget et les entités.

        Returns:
            ExecutionPlan valide.

        Raises:
            ValueError: Si aucun bloc d'étape n'est trouvé dans la réponse.
            ToonParseError: Si un bloc est non parseable.
            ValidationError: Si les champs ne satisfont pas le contrat Pydantic.
        """
        # Extraction de tous les blocs <<<...>>>
        raw_blocks = re.findall(r"<<<(.*?)>>>", raw, re.DOTALL)

        if not raw_blocks:
            raise ValueError(
                "Aucun bloc TOON (<<<...>>>) trouvé dans la réponse du Planner."
            )

        # Le premier bloc est l'en-tête (optionnel mais attendu)
        header_data = parse_toon_to_dict(f"<<<{raw_blocks[0]}>>>")
        step_blocks = raw_blocks[1:]

        # Si le premier bloc ressemble à une étape (contient step_id), pas d'en-tête
        if "step_id" in header_data:
            step_blocks = raw_blocks

        if not step_blocks:
            raise ValueError(
                "Aucun bloc d'étape trouvé dans la réponse TOON du Planner."
            )

        budget = analysis.reasoning_budget

        # Parse et instanciation des PlanStep
        steps: list[PlanStep] = []
        for raw_block in step_blocks[:budget]:
            block_data = parse_toon_to_dict(f"<<<{raw_block}>>>")

            step_id_raw = block_data.get("step_id")
            sub_query_raw = block_data.get("sub_query")

            if not step_id_raw or not sub_query_raw:
                logger.warning(
                    "Bloc TOON ignoré : step_id ou sub_query manquant. Data=%s",
                    block_data,
                )
                continue

            # depends_on : None/vide → liste vide, liste TOON → list[str]
            raw_deps = block_data.get("depends_on")
            depends_on: list[str]
            if isinstance(raw_deps, list):
                depends_on = [str(d) for d in raw_deps]
            elif raw_deps is None:
                depends_on = []
            else:
                depends_on = [str(raw_deps)]

            steps.append(
                PlanStep(
                    step_id=str(step_id_raw),
                    sub_query=str(sub_query_raw),
                    depends_on=depends_on,
                    status=StepStatus.PENDING,
                )
            )

        if not steps:
            raise ValueError(
                "Aucune PlanStep valide construite depuis la réponse du Planner."
            )

        # Avertissement si des étapes ont été tronquées pour respecter le budget
        if len(step_blocks) > budget:
            logger.warning(
                "Plan tronqué : %d étapes reçues, budget=%d → %d étapes conservées.",
                len(step_blocks),
                budget,
                len(steps),
            )

        # Construction du graphe de dépendances
        dep_graph: dict[str, list[str]] = {s.step_id: s.depends_on for s in steps}

        plan = ExecutionPlan(
            plan_id=_generate_plan_id(query),
            original_query=query,
            steps=steps,
            dependencies_graph=dep_graph,
        )

        # Validation du DAG — bascule sur le fallback si invalide
        if not self._validate_dag(plan):
            logger.warning(
                "DAG invalide (cycle ou référence inconnue) " "— fallback séquentiel."
            )
            return self._build_sequential_fallback(query, analysis)

        logger.info(
            "Plan valide généré : plan_id=%s, %d étapes.",
            plan.plan_id,
            len(plan.steps),
        )
        return plan

    # ── Validation du DAG (algorithme de Kahn) ────────────────────────────────

    def _validate_dag(self, plan: ExecutionPlan) -> bool:
        """Vérifie que le plan forme un graphe acyclique dirigé (DAG) valide.

        Implémente l'algorithme de tri topologique de Kahn en Python pur.
        Vérifie simultanément :
        - L'absence de cycle
        - Que chaque référence dans `depends_on` pointe vers un step_id existant

        Args:
            plan: L'ExecutionPlan à valider.

        Returns:
            True si le graphe est un DAG valide, False si un cycle ou une
            référence inconnue est détectée.
        """
        known_ids: set[str] = {s.step_id for s in plan.steps}

        # Vérification préalable : toutes les dépendances référencent des step_id connus
        for step in plan.steps:
            for dep in step.depends_on:
                if dep not in known_ids:
                    logger.warning(
                        "Dépendance inconnue : '%s' référencée par '%s' n'existe pas.",
                        dep,
                        step.step_id,
                    )
                    return False

        # Construction du graphe d'adjacence inversé et du vecteur in-degree
        in_degree: dict[str, int] = {s.step_id: 0 for s in plan.steps}
        # adjacency[u] = liste des nœuds v tels que v dépend de u (u → v)
        adjacency: dict[str, list[str]] = {s.step_id: [] for s in plan.steps}

        for step in plan.steps:
            for dep in step.depends_on:
                adjacency[dep].append(step.step_id)
                in_degree[step.step_id] += 1

        # Algorithme de Kahn : traitement des nœuds de degré entrant 0
        queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
        visited_count = 0

        while queue:
            node = queue.popleft()
            visited_count += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Si tous les nœuds ont été visités, pas de cycle
        is_valid = visited_count == len(plan.steps)
        if not is_valid:
            logger.warning(
                "Cycle détecté dans le DAG (%d/%d nœuds visités).",
                visited_count,
                len(plan.steps),
            )
        return is_valid

    # ── Niveau 2 : Fallback séquentiel ───────────────────────────────────────

    def _build_sequential_fallback(
        self, query: str, analysis: AnalysisResult
    ) -> ExecutionPlan:
        """Construit un plan séquentiel dégradé sans cycle garanti.

        Invoqué si le LLM génère un plan invalide (cycle, référence inconnue,
        TOON non parseable). Crée une chaîne linéaire d'étapes basées sur les
        `detected_entities` de l'AnalysisResult, plafonnée par le budget.

        Si aucune entité n'est disponible, crée une étape unique avec la requête
        complète.

        Args:
            query: La requête utilisateur originale.
            analysis: L'AnalysisResult pour récupérer les entités et le budget.

        Returns:
            ExecutionPlan séquentiel valide (DAG linéaire sans cycle).
        """
        budget = max(1, analysis.reasoning_budget)
        entities = analysis.detected_entities[:budget]

        steps: list[PlanStep] = []

        if entities:
            for i, entity in enumerate(entities, start=1):
                sub_q = (
                    f"Quelles informations sur '{entity}' "
                    f"sont pertinentes pour répondre à : {query}"
                )
                steps.append(
                    PlanStep(
                        step_id=f"step_{i}",
                        sub_query=sub_q,
                        depends_on=[f"step_{i - 1}"] if i > 1 else [],
                        status=StepStatus.PENDING,
                    )
                )
        else:
            steps = [
                PlanStep(
                    step_id="step_1",
                    sub_query=query,
                    depends_on=[],
                    status=StepStatus.PENDING,
                )
            ]

        dep_graph: dict[str, list[str]] = {s.step_id: s.depends_on for s in steps}

        logger.info(
            "Fallback séquentiel : plan_id=%s, %d étape(s).",
            _generate_plan_id(query),
            len(steps),
        )

        return ExecutionPlan(
            plan_id=_generate_plan_id(query),
            original_query=query,
            steps=steps,
            dependencies_graph=dep_graph,
        )
