"""
Baseline RAG naïf — Sprint I5-B.

Point de comparaison du système complet : question → UN SEUL appel
`/retrieve` → génération directe. Aucun Analyzer, aucun Planner, aucun
Critic, aucun Verifier, aucune boucle de rétroaction.

C'est l'architecture que le module REASONING est censé surpasser. Le cahier
des charges vise un gain de faithfulness d'environ +15 points.

ÉQUITÉ DE LA COMPARAISON
========================
Le gain mesuré ne vaut que si les deux systèmes diffèrent par leur SEULE
architecture. Chaque paramètre qui pourrait expliquer un écart autrement est
donc aligné sur le système complet, à l'identique :

    dimension            système complet                    baseline
    ────────────────────────────────────────────────────────────────────
    corpus               HotpotQA distractor, 1966 art.      identique
    moteur               module ACTION, POST /retrieve       identique
    client HTTP          reasoning.action_client.ActionClient identique
    top_k                5 (`nodes._DEFAULT_TOP_K`)          5
    modèle               `DEFAULT_REASONING_MODEL` (7B)      identique
    temperature          0.0                                 0.0
    max_tokens           512                                 512
    prompt de génération `nodes._GENERATION_PROMPT`          RÉUTILISÉ TEL QUEL
    gestion d'erreur     repli fail-closed, réponse dégradée identique
    traçabilité          Langfuse via `observability`        identique

Le prompt n'est pas recopié : il est IMPORTÉ depuis `reasoning.graph.nodes`.
Le recopier laisserait dériver les deux formulations, et l'écart mesuré
porterait alors en partie sur la rédaction du prompt plutôt que sur
l'architecture. C'est le point le plus important de cette liste.

CE QUI DIFFÈRE, ET C'EST TOUT LE PROPOS
----------------------------------------
Le baseline envoie la question de l'utilisateur telle quelle au retriever.
Le système complet la classe, la décompose en sous-requêtes, juge le
contexte obtenu, relance si nécessaire, transmet l'entité résolue d'un saut
au suivant, puis vérifie la réponse contre ses sources.

COÛT ATTENDU
------------
1 appel LLM par question, contre 4 (SIMPLE) à 8 (MULTI_HOP) pour le système
complet — mesures du Sprint I5-A. C'est la moitié « coût » du compromis
qualité/coût exigé par le cahier des charges ; le comptage réel est relevé
par `reasoning.observability`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from reasoning.action_client import ActionClient, ActionClientError
from reasoning.contracts.action_interface import RetrievalRequest, RetrievedChunk
from reasoning.graph.nodes import (
    _DEFAULT_REASONING_MODEL,
    _DEFAULT_TOP_K,
    _GENERATION_PROMPT,
    _OLLAMA_BASE_URL,
    _format_chunks_for_generation,
)

logger = logging.getLogger(__name__)

# Alignés sur `make_generate_answer_node` (nodes.py). Toute divergence
# fausserait la comparaison.
_TEMPERATURE: float = 0.0
_MAX_TOKENS: int = 512

# Réponses de repli, calquées sur celles du nœud `generate_answer`.
_FALLBACK_EMPTY = "I could not generate an answer from the available context."
_FALLBACK_ERROR = "I could not generate an answer due to a technical failure."


@dataclass
class NaiveRagResult:
    """Résultat d'une question traitée par le baseline.

    Attributes:
        question: La question posée, telle quelle.
        answer: La réponse générée.
        chunks: Les chunks récupérés en un unique appel au retriever.
        retrieval_ms: Durée de l'appel au module ACTION.
        generation_ms: Durée de l'appel au LLM.
        total_ms: Durée totale.
        error: Message d'anomalie si le traitement s'est dégradé, sinon None.
    """

    question: str
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    retrieval_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0
    error: str | None = None

    @property
    def contexts(self) -> list[str]:
        """Contenus textuels des chunks — format attendu par RAGAS."""
        return [chunk.content for chunk in self.chunks]

    @property
    def sources(self) -> list[str]:
        """Fichiers d'origine des chunks."""
        return [chunk.source for chunk in self.chunks]


class NaiveRag:
    """RAG en une passe : retrieve puis generate, sans aucun raisonnement.

    Attributes:
        client: Client HTTP vers le module ACTION — le même que le graphe.
        model: Modèle de génération, aligné sur `generate_answer`.
        top_k: Nombre de chunks demandés, aligné sur le graphe.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = _DEFAULT_REASONING_MODEL,
        api_base: str = _OLLAMA_BASE_URL,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self.client = client if client is not None else ActionClient()
        self.model = model
        self.api_base = api_base
        self.top_k = top_k

    def answer(self, question: str, query_id: str) -> NaiveRagResult:
        """Traite une question de bout en bout.

        Ne lève jamais : toute défaillance produit un résultat dégradé,
        exactement comme le repli fail-closed du graphe. Une campagne de
        plusieurs heures ne doit pas s'interrompre sur une question.

        Args:
            question: La question de l'utilisateur, envoyée telle quelle.
            query_id: Identifiant de corrélation, repris dans la requête.

        Returns:
            Le résultat, éventuellement dégradé (`error` renseigné).
        """
        started = time.perf_counter()

        chunks, retrieval_ms, error = self._retrieve(question, query_id)
        answer, generation_ms, gen_error = self._generate(question, chunks)

        return NaiveRagResult(
            question=question,
            answer=answer,
            chunks=chunks,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=round((time.perf_counter() - started) * 1000),
            error=error or gen_error,
        )

    # ── Étapes ───────────────────────────────────────────────────────────

    def _retrieve(
        self, question: str, query_id: str
    ) -> tuple[list[RetrievedChunk], int, str | None]:
        """Un unique appel au module ACTION, sur la question brute."""
        request = RetrievalRequest(
            query_id=query_id,
            sub_query=question,  # aucune décomposition : c'est le propos
            hop_index=0,
            top_k=self.top_k,
        )
        started = time.perf_counter()
        try:
            response = self.client.retrieve(request)
        except ActionClientError as exc:
            logger.warning(
                "naive_rag[%s] : module ACTION injoignable ou en erreur "
                "(%s: %s) — génération sans contexte.",
                query_id,
                type(exc).__name__,
                exc,
            )
            return [], round((time.perf_counter() - started) * 1000), str(exc)
        except Exception as exc:  # noqa: BLE001 — la campagne ne doit pas casser
            logger.error(
                "naive_rag[%s] : erreur inattendue au retrieval (%s: %s).",
                query_id,
                type(exc).__name__,
                exc,
            )
            return [], round((time.perf_counter() - started) * 1000), str(exc)

        return (
            list(response.chunks),
            round((time.perf_counter() - started) * 1000),
            None,
        )

    def _generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> tuple[str, int, str | None]:
        """Génération directe, avec le prompt du système complet."""
        # `completion` est résolu à l'appel et non à l'import : c'est ce qui
        # permet à `observability.instrument()` de tracer et compter cet
        # appel comme ceux du système complet.
        from reasoning.graph import nodes

        # Accès par `getattr` : `nodes` fait `from litellm import completion`,
        # que mypy --strict ne considère pas comme un ré-export explicite.
        completion = getattr(nodes, "completion")  # noqa: B009

        prompt = _GENERATION_PROMPT.format(
            query=question, context=_format_chunks_for_generation(chunks)
        )
        started = time.perf_counter()
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.api_base,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                answer = _FALLBACK_EMPTY
        except Exception as exc:  # noqa: BLE001 — la campagne ne doit pas casser
            logger.warning(
                "naive_rag : génération en échec (%s: %s) — réponse de repli.",
                type(exc).__name__,
                exc,
            )
            return (
                _FALLBACK_ERROR,
                round((time.perf_counter() - started) * 1000),
                str(exc),
            )

        return answer, round((time.perf_counter() - started) * 1000), None


__all__ = ["NaiveRag", "NaiveRagResult"]
