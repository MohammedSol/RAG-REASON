"""
Tests d'intégration — Sprint 5.3 : Verifier réel (Ollama live).

Ces tests appellent RÉELLEMENT l'instance Ollama locale via LiteLLM.
Ils nécessitent :
    - `ollama serve` en arrière-plan
    - Le modèle DEFAULT_REASONING_MODEL téléchargé (`ollama pull qwen2.5:7b`)
    - Les variables d'environnement du fichier `.env`

Exécution :
    uv run pytest tests/integration/test_verifier_live.py -v -m integration

Pour ignorer ces tests si Ollama n'est pas disponible :
    uv run pytest tests/ -v -m "not integration"

Note sur la détection d'indisponibilité d'Ollama :
    Le Verifier applique un repli *fail-closed* (verifier_spec.md §9.3) : si
    Ollama est injoignable, `verify()` ne lève PAS d'exception, il retourne
    `is_grounded=False`, `faithfulness_score=0.0` et
    `unsupported_claims=["verification_failed"]`. Sans précaution, les paires
    2 et 3 — qui attendent précisément `is_grounded=False` — passeraient au
    vert alors qu'aucune vérification n'a eu lieu. Deux garde-fous sont donc
    posés : une sonde de connectivité explicite (`_require_ollama`) et
    l'assertion que le marqueur `verification_failed` est absent des
    résultats.
"""

from __future__ import annotations

import os

import httpx
import pytest
from reasoning.contracts.action_interface import RetrievedChunk
from reasoning.contracts.internal_models import VerificationResult
from reasoning.verifier import Verifier

# ─────────────────────────────────────────────────────────────────────────────
# Sonde de connectivité Ollama — échec explicite, jamais de skip silencieux
# ─────────────────────────────────────────────────────────────────────────────

_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Sentinelle produite par le repli défensif du Verifier quand l'appel LLM
# échoue (verifier_spec.md §9.3). Sa présence signale une panne
# d'infrastructure, pas un verdict de non-fondement légitime.
_FALLBACK_MARKER = "verification_failed"


@pytest.fixture(scope="module")
def require_ollama() -> None:
    """Échoue explicitement — sans skip — si Ollama est injoignable.

    Raises:
        Failed: Si le service Ollama ne répond pas sur `OLLAMA_BASE_URL`.
    """
    try:
        response = httpx.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        response.raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.fail(
            "Ollama est indisponible sur "
            f"{_OLLAMA_BASE_URL} ({type(exc).__name__}: {exc}). "
            "Les tests d'intégration du Verifier ne peuvent pas être "
            "exécutés. Lancer `ollama serve` puis réessayer — ces tests ne "
            "sont volontairement PAS skippés silencieusement.",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def live_verifier() -> Verifier:
    """Instance Verifier en mode réel, partagée sur tout le module."""
    return Verifier()


# ─────────────────────────────────────────────────────────────────────────────
# Sources de référence — partagées par les trois paires
# ─────────────────────────────────────────────────────────────────────────────

# Les `chunk_id` ci-dessous sont les SEULS identifiants que le LLM peut
# légitimement citer dans `source_chunk_id`.
_SOURCES: list[RetrievedChunk] = [
    RetrievedChunk(
        chunk_id="openai-history-001",
        content=(
            "OpenAI was founded in December 2015 by Elon Musk, Sam Altman, "
            "Greg Brockman, Ilya Sutskever, Wojciech Zaremba, and John Schulman. "
            "The company is headquartered in San Francisco, California. "
            "It was established as a non-profit before transitioning to a "
            "capped-profit structure in 2019."
        ),
        source="openai_company_history.pdf",
        relevance_score=0.94,
    ),
    RetrievedChunk(
        chunk_id="gpt4-report-002",
        content=(
            "GPT-4 was released by OpenAI in March 2023. It is a large "
            "multimodal model capable of processing both text and image "
            "inputs, and it exhibits human-level performance on several "
            "professional and academic benchmarks."
        ),
        source="gpt4_technical_report.pdf",
        relevance_score=0.89,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Les trois paires (answer, sources) de référence — verifier_spec.md §12
# ─────────────────────────────────────────────────────────────────────────────

# Paire 1 — réponse manifestement fondée : chaque affirmation est
# explicitement présente dans les chunks ci-dessus.
_ANSWER_GROUNDED: str = (
    "OpenAI was founded in December 2015. Its headquarters are in San "
    "Francisco, California. GPT-4 was released by OpenAI in March 2023 and "
    "can process both text and image inputs."
)

# Paire 2 — réponse partiellement fondée : les deux premières affirmations
# sont dans les sources, la troisième (nombre de paramètres) n'y figure nulle
# part et constitue une extrapolation non traçable.
_ANSWER_PARTIAL: str = (
    "OpenAI was founded in December 2015. GPT-4 was released in March 2023. "
    "GPT-4 was trained on exactly 1.8 trillion parameters using 25,000 "
    "NVIDIA A100 GPUs over a period of 90 days."
)

# Paire 3 — réponse entièrement hallucinée : aucune affirmation n'est
# soutenue, et plusieurs contredisent frontalement les sources.
_ANSWER_HALLUCINATED: str = (
    "OpenAI was founded in Paris in 1998 by Alan Turing and Ada Lovelace. "
    "GPT-4 was released in 2009 and runs exclusively on quantum hardware "
    "manufactured in Antarctica."
)


# ─────────────────────────────────────────────────────────────────────────────
# Résultats mis en cache au niveau module — un seul appel LLM par paire
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def result_grounded(
    live_verifier: Verifier, require_ollama: None
) -> VerificationResult:
    """Résultat de la paire 1 (réponse manifestement fondée)."""
    return live_verifier.verify(_ANSWER_GROUNDED, _SOURCES)


@pytest.fixture(scope="module")
def result_partial(live_verifier: Verifier, require_ollama: None) -> VerificationResult:
    """Résultat de la paire 2 (une affirmation absente des sources)."""
    return live_verifier.verify(_ANSWER_PARTIAL, _SOURCES)


@pytest.fixture(scope="module")
def result_hallucinated(
    live_verifier: Verifier, require_ollama: None
) -> VerificationResult:
    """Résultat de la paire 3 (réponse entièrement hallucinée)."""
    return live_verifier.verify(_ANSWER_HALLUCINATED, _SOURCES)


# ─────────────────────────────────────────────────────────────────────────────
# Tests d'intégration
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestVerifierLive:
    """Tests de bout-en-bout avec Ollama réel.

    Nécessite : `ollama serve` + modèle qwen2.5:7b téléchargé.
    """

    # ── Paire 1 — réponse manifestement fondée ────────────────────────────

    def test_grounded_answer_is_accepted(
        self, result_grounded: VerificationResult
    ) -> None:
        """Paire 1 : toutes les affirmations sont traçables → is_grounded=True."""
        assert result_grounded.is_grounded is True
        assert result_grounded.unsupported_claims == []

    def test_grounded_answer_preserves_final_answer(
        self, result_grounded: VerificationResult
    ) -> None:
        """Invariant verifier_spec.md §7.1 — final_answer identique à answer."""
        assert result_grounded.final_answer == _ANSWER_GROUNDED

    # ── Paire 2 — une affirmation absente des sources ─────────────────────

    def test_partial_answer_is_rejected(
        self, result_partial: VerificationResult
    ) -> None:
        """Paire 2 : l'affirmation non traçable fait tomber le verdict."""
        assert result_partial.is_grounded is False
        assert len(result_partial.unsupported_claims) > 0

    def test_partial_answer_is_not_an_infrastructure_failure(
        self, result_partial: VerificationResult
    ) -> None:
        """Le verdict négatif vient du LLM, pas du repli défensif.

        Sans ce contrôle, une panne d'Ollama produirait exactement le même
        `is_grounded=False` et le test passerait au vert à tort.
        """
        assert _FALLBACK_MARKER not in result_partial.unsupported_claims

    def test_partial_answer_preserves_final_answer(
        self, result_partial: VerificationResult
    ) -> None:
        """Invariant verifier_spec.md §7.1 — jamais tronquée, même rejetée."""
        assert result_partial.final_answer == _ANSWER_PARTIAL

    # ── Paire 3 — réponse entièrement hallucinée ──────────────────────────

    def test_hallucinated_answer_is_rejected(
        self, result_hallucinated: VerificationResult
    ) -> None:
        """Paire 3 : aucune affirmation soutenue → rejet avec claims listés."""
        assert result_hallucinated.is_grounded is False
        assert len(result_hallucinated.unsupported_claims) > 0

    def test_hallucinated_answer_is_not_an_infrastructure_failure(
        self, result_hallucinated: VerificationResult
    ) -> None:
        """Le verdict négatif vient du LLM, pas du repli défensif."""
        assert _FALLBACK_MARKER not in result_hallucinated.unsupported_claims

    def test_hallucinated_answer_preserves_final_answer(
        self, result_hallucinated: VerificationResult
    ) -> None:
        """Invariant verifier_spec.md §7.1 — restituée intacte."""
        assert result_hallucinated.final_answer == _ANSWER_HALLUCINATED

    # ── Ordre relatif des scores — critère stable malgré la variabilité 7B ─

    def test_hallucinated_scores_strictly_below_partial(
        self,
        result_partial: VerificationResult,
        result_hallucinated: VerificationResult,
    ) -> None:
        """Une hallucination totale doit scorer sous une réponse partielle.

        On compare un ORDRE plutôt qu'un seuil absolu : le score exact d'un
        modèle 7B local varie d'une exécution à l'autre, mais le classement
        relatif entre « 2 affirmations sur 3 traçables » et « aucune
        affirmation traçable » reste stable.
        """
        assert result_hallucinated.faithfulness_score < (
            result_partial.faithfulness_score
        )
