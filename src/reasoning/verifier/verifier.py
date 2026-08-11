"""
Verifier — Vérificateur de fidélité (Groundedness Check).

Ce composant est le cinquième nœud du graphe LangGraph. Il est exécuté après
la génération de la réponse candidate, avant sa présentation à l'utilisateur.
Sa mission est de détecter les hallucinations en vérifiant que chaque
affirmation factuelle de la réponse est traçable dans les chunks source
récupérés par le module ACTION.

Architecture (conforme à docs/verifier_spec.md §9.3) :
    Niveau 1 : Appel LLM via LiteLLM → Ollama (DEFAULT_REASONING_MODEL) avec
               prompt Chain-of-Thought structuré, décomposition et vérification
               des claims en un seul appel.
    Niveau 2 : Fallback défensif si le LLM échoue (ToonParseError,
               ValidationError, Timeout) → VerificationResult avec
               is_grounded=False et faithfulness_score=0.0.

Cas spécial (sans appel LLM) :
    Si `sources` est vide → verdict défensif immédiat (verifier_spec.md §5.3).

Invariant absolu (verifier_spec.md §7) :
    `final_answer` est TOUJOURS une copie stricte de `answer` reçu en entrée —
    le Verifier ne modifie, ne tronque et ne reformule jamais la réponse.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from litellm import completion
from pydantic import ValidationError

from reasoning.contracts.action_interface import RetrievedChunk
from reasoning.contracts.internal_models import VerificationResult
from reasoning.shared.toon_utils import ToonParseError, parse_toon_records
from reasoning.verifier.prompts import VERIFICATION_PROMPT

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_REASONING_MODEL: str = os.getenv(
    "DEFAULT_REASONING_MODEL", "ollama/qwen2.5:7b"
)

# Nombre maximum de tokens pour la réponse du Verifier.
# Le bloc TOON de sortie peut être long si la réponse contient de nombreux
# claims (conforme à verifier_spec.md §9.4).
_MAX_TOKENS: int = 512

# Nombre maximum de caractères par chunk dans le prompt (troncature défensive).
# Valeur proposée par verifier_spec.md §8.6.
_MAX_CHUNK_CHARS: int = 600


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────


class Verifier:
    """Vérificateur de fidélité (Groundedness) pour le graphe Self-RAG.

    Vérifie que chaque affirmation factuelle de la réponse candidate est
    traçable dans les chunks source récupérés au cours du plan d'exécution.
    Produit un `VerificationResult` structuré avec un score de fidélité, les
    affirmations non-fondées identifiées, et la réponse finale (inchangée).

    Architecture hybride :
    - Niveau 1 : Appel LLM (Qwen 7B) avec prompt Chain-of-Thought
    - Niveau 2 : Fallback défensif si le LLM échoue ou timeout

    Attributes:
        model: Identifiant LiteLLM du modèle de raisonnement (Qwen 7B).
        api_base: URL de base de l'API Ollama.
        max_tokens: Limite de tokens pour la réponse LLM.
        temperature: Température de génération (0.0 = déterministe).
        faithfulness_threshold: Seuil de faithfulness_score pour is_grounded=True.
    """

    def __init__(
        self,
        model: str = _DEFAULT_REASONING_MODEL,
        api_base: str = _OLLAMA_BASE_URL,
        max_tokens: int = _MAX_TOKENS,
        temperature: float = 0.0,
        # Seuil par défaut conforme à verifier_spec.md §6.1 — paramétrable.
        faithfulness_threshold: float = 0.80,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.faithfulness_threshold = faithfulness_threshold

    # ── API publique ──────────────────────────────────────────────────────────

    def verify(
        self,
        answer: str,
        sources: list[RetrievedChunk],
    ) -> VerificationResult:
        """Vérifie la fidélité d'une réponse candidate par rapport aux sources.

        Tente d'abord la vérification LLM via un prompt Chain-of-Thought TOON
        multi-enregistrements. Bascule sur un fallback défensif en cas
        d'échec du parsing ou de timeout. Cas spécial : si `sources` est
        vide, retourne immédiatement is_grounded=False sans appel LLM.

        Args:
            answer: La réponse candidate générée par le nœud generate_answer.
            sources: Les chunks source récupérés au cours du plan d'exécution.

        Returns:
            VerificationResult avec is_grounded, faithfulness_score,
            unsupported_claims et final_answer (copie stricte de `answer`).
        """
        # ── Cas spécial : aucune source disponible (verifier_spec.md §5.3) ────
        if not sources:
            logger.warning(
                "Verifier : aucune source disponible — is_grounded=False sans LLM."
            )
            return VerificationResult(
                is_grounded=False,
                faithfulness_score=0.0,
                unsupported_claims=["no sources available for verification"],
                final_answer=answer,
            )

        # ── Niveau 1 : Vérification LLM ────────────────────────────────────────
        try:
            return self._verify_with_llm(answer, sources)
        except (OSError, TimeoutError, RuntimeError) as exc:
            logger.warning(
                "Verifier : erreur réseau/LLM (%s: %s) — fallback défensif.",
                type(exc).__name__,
                exc,
            )
        except (ToonParseError, ValidationError, ValueError) as exc:
            logger.warning(
                "Verifier : réponse LLM non parseable (%s: %s) — fallback.",
                type(exc).__name__,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Verifier : erreur inattendue (%s: %s) — fallback défensif.",
                type(exc).__name__,
                exc,
            )

        # ── Niveau 2 : Fallback défensif (verifier_spec.md §9.3) ──────────────
        return self._build_defensive_fallback(answer)

    # ── Niveau 1 : Vérification LLM ────────────────────────────────────────────

    def _verify_with_llm(
        self,
        answer: str,
        sources: list[RetrievedChunk],
    ) -> VerificationResult:
        """Appelle le LLM avec le prompt CoT et parse la réponse TOON.

        Args:
            answer: La réponse candidate à vérifier.
            sources: Les chunks source pour la confrontation claim↔source.

        Returns:
            VerificationResult construit depuis les enregistrements TOON.

        Raises:
            ToonParseError: Si la réponse ne contient pas de bloc TOON valide.
            ValidationError: Si les valeurs TOON ne satisfont pas le contrat.
            ValueError: Si un enregistrement est incomplet.
            OSError | TimeoutError: Si Ollama est inaccessible.
        """
        chunks_text = self._format_chunks(sources)
        prompt = VERIFICATION_PROMPT.format(
            answer=answer,
            n_chunks=len(sources),
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
        logger.debug("Verifier réponse LLM brute : %s", raw_content)

        records = parse_toon_records(raw_content)
        return self._build_verification(answer, records)

    # ── Construction de VerificationResult ─────────────────────────────────────

    def _build_verification(
        self,
        answer: str,
        records: list[dict[str, Any]],
    ) -> VerificationResult:
        """Construit un VerificationResult depuis les enregistrements TOON parsés.

        Calcule faithfulness_score = claims_supported / total_claims et
        is_grounded = (faithfulness_score >= threshold). Cas limite
        total_claims=0 : faithfulness_score=1.0, is_grounded=True
        (verifier_spec.md §5.2).

        Args:
            answer: La réponse candidate (copiée telle quelle en final_answer).
            records: Liste d'enregistrements issus de parse_toon_records().

        Returns:
            VerificationResult instancié et validé par Pydantic.

        Raises:
            ValueError: Si `claim_text` ou `is_supported` est absent d'un
                enregistrement.
        """
        total_claims = len(records)

        # Cas limite : aucun claim vérifiable (verifier_spec.md §5.2)
        if total_claims == 0:
            logger.info(
                "Verifier : aucun claim identifié — is_grounded=True (trivial)."
            )
            return VerificationResult(
                is_grounded=True,
                faithfulness_score=1.0,
                unsupported_claims=[],
                final_answer=answer,
            )

        claims_supported = 0
        unsupported_claims: list[str] = []

        for record in records:
            raw_claim = record.get("claim_text")
            if raw_claim is None:
                raise ValueError(
                    f"Champ 'claim_text' absent d'un enregistrement TOON : {record}"
                )
            claim_text = str(raw_claim).strip()

            raw_supported = record.get("is_supported")
            if raw_supported is None:
                raise ValueError(
                    f"Champ 'is_supported' absent d'un enregistrement TOON : {record}"
                )
            # Piège de typage (verifier_spec.md §8.4) : is_supported est
            # TOUJOURS une chaîne après parsing TOON, jamais un bool Python.
            # Comparaison explicite obligatoire — ne jamais écrire
            # `if raw_supported:` ("false" est truthy en Python).
            is_supported = str(raw_supported).strip().lower() == "true"

            if is_supported:
                claims_supported += 1
            else:
                unsupported_claims.append(claim_text)

        faithfulness_score = claims_supported / total_claims
        is_grounded = faithfulness_score >= self.faithfulness_threshold

        result = VerificationResult(
            is_grounded=is_grounded,
            faithfulness_score=faithfulness_score,
            unsupported_claims=unsupported_claims,
            final_answer=answer,
        )

        logger.info(
            "Verifier : faithfulness_score=%.2f is_grounded=%s (%d/%d claims)",
            faithfulness_score,
            is_grounded,
            claims_supported,
            total_claims,
        )
        return result

    @staticmethod
    def _build_defensive_fallback(answer: str) -> VerificationResult:
        """Construit un VerificationResult de secours quand le LLM échoue.

        Retourne systématiquement is_grounded=False avec faithfulness_score=0.0.
        `final_answer` reste une copie stricte de `answer` — invariant absolu
        du composant (verifier_spec.md §7.1).

        Args:
            answer: La réponse candidate reçue en entrée de `verify()`.

        Returns:
            VerificationResult défensif avec is_grounded=False.
        """
        return VerificationResult(
            is_grounded=False,
            faithfulness_score=0.0,
            unsupported_claims=["verification_failed"],
            final_answer=answer,
        )

    @staticmethod
    def _format_chunks(sources: list[RetrievedChunk]) -> str:
        """Formate les chunks source en texte lisible pour le prompt CoT.

        Chaque chunk est présenté avec son `chunk_id` (repris tel quel par le
        LLM dans `source_chunk_id`), sa source, et son contenu (tronqué à
        `_MAX_CHUNK_CHARS` pour rester dans le contexte du modèle 7B).

        Args:
            sources: Les chunks source à formater.

        Returns:
            Chaîne de caractères prête à être injectée dans le prompt.
        """
        lines: list[str] = []
        for chunk in sources:
            content = chunk.content
            if len(content) > _MAX_CHUNK_CHARS:
                content = content[:_MAX_CHUNK_CHARS] + "... [truncated]"
            lines.append(
                f"  [chunk_id={chunk.chunk_id} | source={chunk.source}]: {content}"
            )
        return "\n".join(lines)
