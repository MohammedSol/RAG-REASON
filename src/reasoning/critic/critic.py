"""
Critic — Évaluateur de contexte (Self-RAG Feedback Loop).

Ce composant est le quatrième nœud du graphe LangGraph. Il évalue la qualité
du contexte récupéré par le module ACTION pour une étape de plan donnée, et
produit un verdict structuré (`CriticEvaluation`) indiquant si le contexte
est suffisant et, dans le cas contraire, un feedback actionnable pour le
Planner.

Architecture (conforme à docs/critic_spec.md §7.3) :
    Niveau 1 : Appel LLM via LiteLLM → Ollama (DEFAULT_REASONING_MODEL)
               avec prompt Chain-of-Thought structuré.
    Niveau 2 : Fallback défensif si le LLM échoue (ToonParseError,
               ValidationError, Timeout) → CriticEvaluation avec
               is_sufficient=False et relevance_score=0.0.

Cas spécial (sans appel LLM) :
    Si RetrievalResponse.chunks est vide → évaluation immédiate sans LLM.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from litellm import completion
from pydantic import ValidationError

from reasoning.contracts.action_interface import RetrievalResponse
from reasoning.contracts.internal_models import CriticEvaluation, PlanStep
from reasoning.critic.prompts import EVALUATION_PROMPT
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

# Nombre maximum de tokens pour la réponse du Critic.
# Le bloc TOON de sortie est court (3 lignes), 256 tokens suffisent largement.
_MAX_TOKENS: int = 256

# Nombre maximum de caractères par chunk dans le prompt (troncature défensive).
# Évite de dépasser le contexte du modèle sur des chunks longs.
_MAX_CHUNK_CHARS: int = 800


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────


class Critic:
    """Évaluateur de contexte pour le feedback loop Self-RAG.

    Évalue si le contexte récupéré par le module ACTION est suffisant pour
    répondre à une sous-question issue du plan d'exécution. Produit un
    `CriticEvaluation` structuré avec un score de pertinence, les aspects
    manquants identifiés, et un feedback actionnable pour le Planner.

    Architecture hybride :
    - Niveau 1 : Appel LLM (Qwen 7B) avec prompt Chain-of-Thought
    - Niveau 2 : Fallback défensif si le LLM échoue ou timeout

    Attributes:
        model: Identifiant LiteLLM du modèle de raisonnement (Qwen 7B).
        api_base: URL de base de l'API Ollama.
        max_tokens: Limite de tokens pour la réponse LLM.
        temperature: Température de génération (0.0 = déterministe).
        sufficiency_threshold: Seuil de relevance_score pour is_sufficient=True.
        max_retries: Nombre maximum de re-retrievals par PlanStep (garde locale).
    """

    def __init__(
        self,
        model: str = _DEFAULT_REASONING_MODEL,
        api_base: str = _OLLAMA_BASE_URL,
        max_tokens: int = _MAX_TOKENS,
        temperature: float = 0.0,
        # Seuil par défaut conforme à critic_spec.md §3.1 — paramétrable.
        sufficiency_threshold: float = 0.70,
        # Valeur par défaut conforme à critic_spec.md §5.1 — paramétrable.
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.sufficiency_threshold = sufficiency_threshold
        self.max_retries = max_retries

    # ── API publique ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        step: PlanStep,
        response: RetrievalResponse,
    ) -> CriticEvaluation:
        """Évalue la qualité du contexte récupéré pour une étape de plan.

        Tente d'abord l'évaluation LLM via un prompt Chain-of-Thought TOON.
        Bascule sur un fallback défensif en cas d'échec du parsing ou de
        timeout. Cas spécial : si la réponse ne contient aucun chunk, retourne
        immédiatement is_sufficient=False sans appel LLM.

        Args:
            step: L'étape de plan évaluée (fournit step_id et sub_query).
            response: La réponse du module ACTION (chunks récupérés).

        Returns:
            CriticEvaluation avec is_sufficient, relevance_score,
            missing_aspects et feedback.
        """
        # ── Cas spécial : aucun chunk récupéré (critic_spec.md §8) ────────────
        if not response.chunks:
            logger.warning(
                "Critic[%s] : aucun chunk reçu — is_sufficient=False sans LLM.",
                step.step_id,
            )
            return CriticEvaluation(
                step_id=step.step_id,
                is_sufficient=False,
                relevance_score=0.0,
                missing_aspects=["no chunks retrieved"],
                feedback="No chunks were retrieved for this step.",
            )

        # ── Niveau 1 : Évaluation LLM ─────────────────────────────────────────
        try:
            return self._evaluate_with_llm(step, response)
        except (OSError, TimeoutError, RuntimeError) as exc:
            logger.warning(
                "Critic[%s] : erreur réseau/LLM (%s: %s) — fallback défensif.",
                step.step_id,
                type(exc).__name__,
                exc,
            )
        except (ToonParseError, ValidationError, ValueError) as exc:
            logger.warning(
                "Critic[%s] : réponse LLM non parseable (%s: %s) — fallback.",
                step.step_id,
                type(exc).__name__,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Critic[%s] : erreur inattendue (%s: %s) — fallback défensif.",
                step.step_id,
                type(exc).__name__,
                exc,
            )

        # ── Niveau 2 : Fallback défensif (critic_spec.md §7.3) ───────────────
        return self._build_defensive_fallback(step.step_id)

    # ── Niveau 1 : Évaluation LLM ─────────────────────────────────────────────

    def _evaluate_with_llm(
        self,
        step: PlanStep,
        response: RetrievalResponse,
    ) -> CriticEvaluation:
        """Appelle le LLM avec le prompt CoT et parse la réponse TOON.

        Args:
            step: L'étape de plan contenant sub_query et step_id.
            response: La réponse du module ACTION avec les chunks.

        Returns:
            CriticEvaluation parsée depuis la réponse TOON du LLM.

        Raises:
            ToonParseError: Si la réponse ne contient pas de bloc TOON valide.
            ValidationError: Si les valeurs TOON ne satisfont pas le contrat.
            OSError | TimeoutError: Si Ollama est inaccessible.
        """
        chunks_text = self._format_chunks(response)
        prompt = EVALUATION_PROMPT.format(
            sub_query=step.sub_query,
            n_chunks=len(response.chunks),
            chunks_content=chunks_text,
        )

        llm_response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        raw_content: str = (llm_response.choices[0].message.content or "").strip()
        logger.debug("Critic[%s] réponse LLM brute : %s", step.step_id, raw_content)

        data = parse_toon_to_dict(raw_content)
        return self._build_evaluation(step.step_id, data)

    # ── Construction de CriticEvaluation ──────────────────────────────────────

    def _build_evaluation(
        self,
        step_id: str,
        data: dict[str, object],
    ) -> CriticEvaluation:
        """Construit un CriticEvaluation depuis un dictionnaire parsé (TOON).

        Calcule is_sufficient en comparant relevance_score au seuil paramétrable.
        Le champ step_id est toujours injecté depuis PlanStep.step_id —
        il n'est pas demandé au LLM (critic_spec.md §6.3).

        Args:
            step_id: Identifiant de l'étape, repris depuis PlanStep.step_id.
            data: Dictionnaire issu de parse_toon_to_dict().

        Returns:
            CriticEvaluation instancié et validé par Pydantic.

        Raises:
            ValidationError: Si relevance_score est hors [0.0, 1.0].
            ValueError: Si relevance_score est absent ou non numérique.
        """
        raw_score = data.get("relevance_score")
        if raw_score is None:
            raise ValueError(
                f"Champ 'relevance_score' absent de la réponse TOON : {data}"
            )

        # Mypy strict : float() n'accepte pas object directement —
        # on normalise via str() pour couvrir int, float et str parsés par TOON.
        if not isinstance(raw_score, int | float | str):
            raise ValueError(
                f"'relevance_score' de type inattendu : {type(raw_score)!r}"
            )

        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'relevance_score' non convertible en float : {raw_score!r}"
            ) from exc

        # Extraction de missing_aspects : liste ou chaîne vide
        raw_aspects = data.get("missing_aspects")
        if isinstance(raw_aspects, list):
            missing_aspects = [str(a) for a in raw_aspects if str(a).strip()]
        elif raw_aspects is None or raw_aspects == "":
            missing_aspects = []
        else:
            missing_aspects = [str(raw_aspects).strip()]

        # Extraction du feedback
        raw_feedback = data.get("feedback")
        feedback = str(raw_feedback).strip() if raw_feedback else ""

        # is_sufficient calculé en Python — PAS généré par le LLM
        is_sufficient = score >= self.sufficiency_threshold

        evaluation = CriticEvaluation(
            step_id=step_id,
            is_sufficient=is_sufficient,
            relevance_score=score,
            missing_aspects=missing_aspects,
            feedback=feedback,
        )

        logger.info(
            "Critic[%s] : relevance_score=%.2f is_sufficient=%s",
            step_id,
            score,
            is_sufficient,
        )
        return evaluation

    @staticmethod
    def _build_defensive_fallback(step_id: str) -> CriticEvaluation:
        """Construit un CriticEvaluation de secours quand le LLM échoue.

        Retourne systématiquement is_sufficient=False avec relevance_score=0.0
        pour déclencher un re-retrieval (si le compteur max_retries le permet).

        Args:
            step_id: Identifiant de l'étape évaluée.

        Returns:
            CriticEvaluation défensif avec is_sufficient=False.
        """
        return CriticEvaluation(
            step_id=step_id,
            is_sufficient=False,
            relevance_score=0.0,
            missing_aspects=["parsing_failed"],
            feedback="LLM evaluation failed — forcing re-retrieval.",
        )

    @staticmethod
    def _format_chunks(response: RetrievalResponse) -> str:
        """Formate les chunks récupérés en texte lisible pour le prompt CoT.

        Chaque chunk est présenté avec son numéro, sa source, son score de
        pertinence individuel et son contenu (tronqué à _MAX_CHUNK_CHARS si
        nécessaire pour rester dans le contexte du modèle 7B).

        Args:
            response: La réponse du module ACTION contenant les chunks.

        Returns:
            Chaîne de caractères prête à être injectée dans le prompt.
        """
        lines: list[str] = []
        for i, chunk in enumerate(response.chunks, start=1):
            content = chunk.content
            if len(content) > _MAX_CHUNK_CHARS:
                content = content[:_MAX_CHUNK_CHARS] + "... [truncated]"
            lines.append(
                f"  [Chunk {i} — {chunk.source}"
                f" | score={chunk.relevance_score:.2f}]: {content}"
            )
        return "\n".join(lines)
