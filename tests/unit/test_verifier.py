"""
Tests unitaires — Sprint 5.3 : Verifier (mocks LiteLLM).

Ces tests sont rapides et isolés : ils ne nécessitent PAS Ollama.
Toutes les réponses LLM sont simulées via `unittest.mock.patch`.

Exécution :
    uv run pytest tests/unit/test_verifier.py -v --cov=reasoning.verifier
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from reasoning.contracts.action_interface import RetrievedChunk
from reasoning.contracts.internal_models import VerificationResult
from reasoning.shared.toon_utils import dump_dict_to_toon
from reasoning.verifier import Verifier

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — constructeurs de réponses TOON multi-enregistrements mockées
# ─────────────────────────────────────────────────────────────────────────────


def _make_records_response(records: list[dict[str, Any]]) -> MagicMock:
    """Simule une réponse LLM au format TOON multi-enregistrements du Verifier.

    Args:
        records: Liste de dicts, un par claim (claim_text, is_supported,
            source_chunk_id).

    Returns:
        MagicMock dont choices[0].message.content est un bloc TOON
        conforme au format Option C (enregistrements séparés par '---').
    """
    mock_response = MagicMock()
    if not records:
        toon = "<<<\n>>>"
    else:
        sections: list[str] = []
        for record in records:
            block = dump_dict_to_toon(record)
            inner = block.removeprefix("<<<\n").removesuffix("\n>>>")
            sections.append(inner)
        toon = "<<<\n" + "\n---\n".join(sections) + "\n>>>"
    mock_response.choices[0].message.content = toon
    return mock_response


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures partagées — données de test réalistes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def verifier() -> Verifier:
    """Instance Verifier avec paramètres neutres pour les tests unitaires."""
    return Verifier(
        model="ollama/test-model",
        api_base="http://localhost:11434",
    )


@pytest.fixture
def answer_openai() -> str:
    """Réponse candidate réaliste sur l'historique d'OpenAI et GPT-4."""
    return (
        "OpenAI was founded in December 2015 by Sam Altman and Elon Musk. "
        "GPT-4 was released in March 2023. The model supports text and "
        "image inputs."
    )


@pytest.fixture
def sources_openai() -> list[RetrievedChunk]:
    """Chunks source réalistes couvrant l'historique d'OpenAI et GPT-4."""
    return [
        RetrievedChunk(
            chunk_id="openai-history-001",
            content=(
                "OpenAI was founded in December 2015 by Elon Musk, Sam Altman, "
                "Greg Brockman, Ilya Sutskever, Wojciech Zaremba, and John Schulman."
            ),
            source="openai_history.pdf",
            relevance_score=0.93,
        ),
        RetrievedChunk(
            chunk_id="gpt4-report-002",
            content=(
                "GPT-4 was released by OpenAI in March 2023. It is a large "
                "multimodal model capable of processing both text and image inputs."
            ),
            source="gpt4_technical_report.pdf",
            relevance_score=0.90,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Réponse entièrement fondée
# ─────────────────────────────────────────────────────────────────────────────


class TestFullyGroundedAnswer:
    """Cas 1 — Réponse 100% fondée : is_grounded=True, faithfulness_score=1.0."""

    @patch("reasoning.verifier.verifier.completion")
    def test_fully_grounded_returns_is_grounded_true(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """3 claims, tous supportés → is_grounded=True, faithfulness_score=1.0."""
        mock_completion.return_value = _make_records_response(
            [
                {
                    "claim_text": "OpenAI was founded in December 2015 by Sam Altman and Elon Musk",
                    "is_supported": "true",
                    "source_chunk_id": "openai-history-001",
                },
                {
                    "claim_text": "GPT-4 was released in March 2023",
                    "is_supported": "true",
                    "source_chunk_id": "gpt4-report-002",
                },
                {
                    "claim_text": "The model supports text and image inputs",
                    "is_supported": "true",
                    "source_chunk_id": "gpt4-report-002",
                },
            ]
        )

        result = verifier.verify(answer_openai, sources_openai)

        assert isinstance(result, VerificationResult)
        assert result.is_grounded is True
        assert result.faithfulness_score == pytest.approx(1.0)
        assert result.unsupported_claims == []
        mock_completion.assert_called_once()

    @patch("reasoning.verifier.verifier.completion")
    def test_final_answer_never_modified(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """final_answer est TOUJOURS une copie stricte de answer (verifier_spec.md §7)."""
        mock_completion.return_value = _make_records_response(
            [
                {
                    "claim_text": "OpenAI was founded in December 2015",
                    "is_supported": "true",
                    "source_chunk_id": "openai-history-001",
                }
            ]
        )

        result = verifier.verify(answer_openai, sources_openai)

        assert result.final_answer == answer_openai


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Réponse partiellement fondée
# ─────────────────────────────────────────────────────────────────────────────


class TestPartiallyGroundedAnswer:
    """Cas 2 — Réponse partiellement fondée : score intermédiaire."""

    @patch("reasoning.verifier.verifier.completion")
    def test_one_unsupported_claim_out_of_three(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """3 claims, 1 non-supporté → faithfulness_score = 2/3 ≈ 0.667."""
        mock_completion.return_value = _make_records_response(
            [
                {
                    "claim_text": "OpenAI was founded in December 2015",
                    "is_supported": "true",
                    "source_chunk_id": "openai-history-001",
                },
                {
                    "claim_text": "GPT-4 was released in March 2023",
                    "is_supported": "true",
                    "source_chunk_id": "gpt4-report-002",
                },
                {
                    "claim_text": "GPT-4 has one trillion parameters",
                    "is_supported": "false",
                    "source_chunk_id": "",
                },
            ]
        )

        result = verifier.verify(answer_openai, sources_openai)

        assert result.faithfulness_score == pytest.approx(2 / 3)
        # Seuil par défaut 0.80 : 0.667 < 0.80 → non fondé
        assert result.is_grounded is False

    @patch("reasoning.verifier.verifier.completion")
    def test_unsupported_claims_list_contains_exact_text(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """unsupported_claims contient le texte exact du claim non-supporté."""
        mock_completion.return_value = _make_records_response(
            [
                {
                    "claim_text": "OpenAI was founded in December 2015",
                    "is_supported": "true",
                    "source_chunk_id": "openai-history-001",
                },
                {
                    "claim_text": "GPT-4 has one trillion parameters",
                    "is_supported": "false",
                    "source_chunk_id": "",
                },
            ]
        )

        result = verifier.verify(answer_openai, sources_openai)

        assert len(result.unsupported_claims) == 1
        assert result.unsupported_claims[0] == "GPT-4 has one trillion parameters"
        # Le claim supporté ne doit PAS figurer dans unsupported_claims
        assert "OpenAI was founded in December 2015" not in result.unsupported_claims


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Réponse entièrement non-fondée (hallucination totale)
# ─────────────────────────────────────────────────────────────────────────────


class TestFullyUngroundedAnswer:
    """Cas 3 — Réponse entièrement hallucinée : score=0.0, is_grounded=False."""

    @patch("reasoning.verifier.verifier.completion")
    def test_two_claims_zero_supported(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """2 claims, 0 supporté → faithfulness_score=0.0, is_grounded=False."""
        mock_completion.return_value = _make_records_response(
            [
                {
                    "claim_text": "OpenAI was founded in Paris",
                    "is_supported": "false",
                    "source_chunk_id": "",
                },
                {
                    "claim_text": "GPT-4 was released in 2019",
                    "is_supported": "false",
                    "source_chunk_id": "",
                },
            ]
        )

        result = verifier.verify(answer_openai, sources_openai)

        assert result.faithfulness_score == pytest.approx(0.0)
        assert result.is_grounded is False
        assert len(result.unsupported_claims) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Cas limite : zéro claim vérifiable
# ─────────────────────────────────────────────────────────────────────────────


class TestNoClaimsAnswer:
    """Cas 4 — Réponse sans claim factuel : trivialement fondée (verifier_spec.md §5.2)."""

    @patch("reasoning.verifier.verifier.completion")
    def test_empty_toon_block_means_no_claims(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """Bloc TOON vide (<<<\\n>>>) → total_claims=0 → is_grounded=True, score=1.0."""
        mock_completion.return_value = _make_records_response([])

        result = verifier.verify(
            "This is a purely rhetorical statement.", sources_openai
        )

        assert result.is_grounded is True
        assert result.faithfulness_score == pytest.approx(1.0)
        assert result.unsupported_claims == []


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Cas spécial : sources vides (sans appel LLM)
# ─────────────────────────────────────────────────────────────────────────────


class TestEmptySources:
    """Cas particulier — Aucune source disponible : court-circuit sans appel LLM."""

    @patch("reasoning.verifier.verifier.completion")
    def test_empty_sources_no_llm_call(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
    ) -> None:
        """sources=[] → is_grounded=False sans appel LLM (verifier_spec.md §5.3)."""
        result = verifier.verify(answer_openai, [])

        assert isinstance(result, VerificationResult)
        assert result.is_grounded is False
        assert result.faithfulness_score == pytest.approx(0.0)
        assert result.unsupported_claims == ["no sources available for verification"]
        mock_completion.assert_not_called()

    @patch("reasoning.verifier.verifier.completion")
    def test_empty_sources_final_answer_preserved(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
    ) -> None:
        """Même sans sources, final_answer reste identique à answer."""
        result = verifier.verify(answer_openai, [])

        assert result.final_answer == answer_openai


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Fallback défensif : échec LLM
# ─────────────────────────────────────────────────────────────────────────────


class TestDefensiveFallback:
    """Cas 5 — Fallback défensif : échec LLM → is_grounded=False, score=0.0."""

    @patch("reasoning.verifier.verifier.completion")
    def test_fallback_on_llm_timeout(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """Timeout LLM → fallback défensif, pas de crash, is_grounded=False."""
        mock_completion.side_effect = TimeoutError("Ollama timeout")

        result = verifier.verify(answer_openai, sources_openai)

        assert isinstance(result, VerificationResult)
        assert result.is_grounded is False
        assert result.faithfulness_score == pytest.approx(0.0)
        assert result.unsupported_claims == ["verification_failed"]
        assert result.final_answer == answer_openai

    @patch("reasoning.verifier.verifier.completion")
    def test_fallback_on_connection_error(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """OSError (connexion refusée) → fallback défensif."""
        mock_completion.side_effect = OSError("Connection refused")

        result = verifier.verify(answer_openai, sources_openai)

        assert result.is_grounded is False
        assert result.faithfulness_score == pytest.approx(0.0)

    @patch("reasoning.verifier.verifier.completion")
    def test_fallback_on_toon_parse_error(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """Réponse LLM sans bloc TOON → ToonParseError → fallback défensif."""
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = (
            "I believe this answer is well supported by the sources provided."
        )
        mock_completion.return_value = mock_response

        result = verifier.verify(answer_openai, sources_openai)

        assert isinstance(result, VerificationResult)
        assert result.is_grounded is False
        assert result.faithfulness_score == pytest.approx(0.0)

    @patch("reasoning.verifier.verifier.completion")
    def test_fallback_on_missing_claim_text_field(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """Enregistrement TOON sans claim_text → ValueError → fallback."""
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = (
            "<<<\nis_supported :: true\nsource_chunk_id :: openai-history-001\n>>>"
        )
        mock_completion.return_value = mock_response

        result = verifier.verify(answer_openai, sources_openai)

        assert isinstance(result, VerificationResult)
        assert result.is_grounded is False
        assert result.faithfulness_score == pytest.approx(0.0)

    @patch("reasoning.verifier.verifier.completion")
    def test_fallback_on_missing_is_supported_field(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """Enregistrement TOON sans is_supported → ValueError → fallback."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "<<<\nclaim_text :: OpenAI was founded in 2015\n"
            "source_chunk_id :: openai-history-001\n>>>"
        )
        mock_completion.return_value = mock_response

        result = verifier.verify(answer_openai, sources_openai)

        assert isinstance(result, VerificationResult)
        assert result.is_grounded is False
        assert result.faithfulness_score == pytest.approx(0.0)

    @patch("reasoning.verifier.verifier.completion")
    def test_fallback_reason_marker_present(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """Le fallback défensif signale explicitement l'échec dans unsupported_claims."""
        mock_completion.side_effect = RuntimeError("Unexpected internal error")

        result = verifier.verify(answer_openai, sources_openai)

        assert result.unsupported_claims == ["verification_failed"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Piège de typage : is_supported est une chaîne, pas un booléen
# ─────────────────────────────────────────────────────────────────────────────


class TestIsSupportedStringTrap:
    """Vérifie que le Verifier gère is_supported comme une chaîne (verifier_spec.md §8.4)."""

    @patch("reasoning.verifier.verifier.completion")
    def test_is_supported_case_insensitive(
        self,
        mock_completion: MagicMock,
        verifier: Verifier,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """is_supported en majuscules ('TRUE'/'FALSE') est correctement interprété."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "<<<\n"
            "claim_text :: OpenAI was founded in December 2015\n"
            "is_supported :: TRUE\n"
            "source_chunk_id :: openai-history-001\n"
            "---\n"
            "claim_text :: GPT-4 has infinite context length\n"
            "is_supported :: FALSE\n"
            "source_chunk_id ::\n"
            ">>>"
        )
        mock_completion.return_value = mock_response

        result = verifier.verify(answer_openai, sources_openai)

        assert result.faithfulness_score == pytest.approx(0.5)
        assert result.unsupported_claims == ["GPT-4 has infinite context length"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Seuil configurable (faithfulness_threshold)
# ─────────────────────────────────────────────────────────────────────────────


class TestCustomThreshold:
    """Vérifie que faithfulness_threshold est paramétrable et non codé en dur."""

    def test_default_threshold_value(self) -> None:
        """La valeur par défaut de faithfulness_threshold est 0.80 (verifier_spec.md §6.1)."""
        verifier = Verifier()
        assert verifier.faithfulness_threshold == pytest.approx(0.80)

    def test_custom_threshold_is_stored(self) -> None:
        """faithfulness_threshold est correctement stocké comme attribut d'instance."""
        verifier = Verifier(faithfulness_threshold=0.90)
        assert verifier.faithfulness_threshold == pytest.approx(0.90)

    @patch("reasoning.verifier.verifier.completion")
    def test_lenient_threshold_accepts_075_score(
        self,
        mock_completion: MagicMock,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """Avec threshold=0.70, un score de 0.75 (3/4) → is_grounded=True."""
        lenient_verifier = Verifier(
            model="ollama/test-model", faithfulness_threshold=0.70
        )
        mock_completion.return_value = _make_records_response(
            [
                {
                    "claim_text": "claim 1",
                    "is_supported": "true",
                    "source_chunk_id": "c1",
                },
                {
                    "claim_text": "claim 2",
                    "is_supported": "true",
                    "source_chunk_id": "c1",
                },
                {
                    "claim_text": "claim 3",
                    "is_supported": "true",
                    "source_chunk_id": "c1",
                },
                {
                    "claim_text": "claim 4",
                    "is_supported": "false",
                    "source_chunk_id": "",
                },
            ]
        )

        result = lenient_verifier.verify(answer_openai, sources_openai)

        assert result.faithfulness_score == pytest.approx(0.75)
        assert result.is_grounded is True

    @patch("reasoning.verifier.verifier.completion")
    def test_strict_threshold_rejects_075_score(
        self,
        mock_completion: MagicMock,
        answer_openai: str,
        sources_openai: list[RetrievedChunk],
    ) -> None:
        """Avec threshold=0.90, le même score de 0.75 (3/4) → is_grounded=False."""
        strict_verifier = Verifier(
            model="ollama/test-model", faithfulness_threshold=0.90
        )
        mock_completion.return_value = _make_records_response(
            [
                {
                    "claim_text": "claim 1",
                    "is_supported": "true",
                    "source_chunk_id": "c1",
                },
                {
                    "claim_text": "claim 2",
                    "is_supported": "true",
                    "source_chunk_id": "c1",
                },
                {
                    "claim_text": "claim 3",
                    "is_supported": "true",
                    "source_chunk_id": "c1",
                },
                {
                    "claim_text": "claim 4",
                    "is_supported": "false",
                    "source_chunk_id": "",
                },
            ]
        )

        result = strict_verifier.verify(answer_openai, sources_openai)

        assert result.faithfulness_score == pytest.approx(0.75)
        assert result.is_grounded is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Paramètres du constructeur
# ─────────────────────────────────────────────────────────────────────────────


class TestConstructorParams:
    """Vérifie que tous les paramètres du constructeur sont correctement stockés."""

    def test_default_params(self) -> None:
        """Les valeurs par défaut correspondent à celles de verifier_spec.md."""
        verifier = Verifier()
        assert verifier.faithfulness_threshold == pytest.approx(0.80)
        assert verifier.temperature == pytest.approx(0.0)
        assert verifier.max_tokens == 512

    def test_custom_model_stored(self) -> None:
        """Le modèle personnalisé est stocké dans l'instance."""
        verifier = Verifier(model="ollama/qwen2.5:7b")
        assert verifier.model == "ollama/qwen2.5:7b"

    def test_custom_api_base_stored(self) -> None:
        """L'URL api_base personnalisée est stockée dans l'instance."""
        verifier = Verifier(api_base="http://custom-ollama:11434")
        assert verifier.api_base == "http://custom-ollama:11434"
