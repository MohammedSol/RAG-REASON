"""
Tests unitaires — Sprint 4.3 : Critic (mocks LiteLLM).

Ces tests sont rapides et isolés : ils ne nécessitent PAS Ollama.
Toutes les réponses LLM sont simulées via `unittest.mock.patch`.

Exécution :
    uv run pytest tests/unit/test_critic.py -v --cov=reasoning.critic
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from reasoning.contracts.action_interface import RetrievalResponse, RetrievedChunk
from reasoning.contracts.internal_models import CriticEvaluation, PlanStep
from reasoning.critic import Critic
from reasoning.shared.toon_utils import dump_dict_to_toon

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — constructeurs de réponses mockées
# ─────────────────────────────────────────────────────────────────────────────


def _make_llm_response(payload: dict[str, Any]) -> MagicMock:
    """Simule une réponse LLM au format TOON.

    Args:
        payload: Dictionnaire représentant les champs de l'évaluation Critic.

    Returns:
        MagicMock dont l'attribut choices[0].message.content est un bloc TOON.
    """
    mock_response = MagicMock()
    mock_response.choices[0].message.content = dump_dict_to_toon(payload)
    return mock_response


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures partagées — données de test réalistes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def critic() -> Critic:
    """Instance Critic avec paramètres neutres pour les tests unitaires."""
    return Critic(
        model="ollama/test-model",
        api_base="http://localhost:11434",
    )


@pytest.fixture
def step_founding() -> PlanStep:
    """PlanStep réaliste : sous-question sur la date de fondation d'OpenAI."""
    return PlanStep(
        step_id="step_2",
        sub_query="What year was the company that created GPT-4 founded?",
        depends_on=["step_1"],
    )


@pytest.fixture
def step_architecture() -> PlanStep:
    """PlanStep réaliste : sous-question sur l'architecture de BERT."""
    return PlanStep(
        step_id="step_1",
        sub_query="What is the technical architecture of BERT?",
        depends_on=[],
    )


@pytest.fixture
def response_relevant() -> RetrievalResponse:
    """RetrievalResponse avec contexte directement pertinent pour la question."""
    return RetrievalResponse(
        query_id="q-openai-founding",
        chunks=[
            RetrievedChunk(
                chunk_id="openai-history-001",
                content=(
                    "OpenAI was founded in December 2015 by Elon Musk, Sam Altman, "
                    "Greg Brockman, Ilya Sutskever, Wojciech Zaremba, and John Schulman. "
                    "The company is headquartered in San Francisco, California."
                ),
                source="openai_company_history.pdf",
                relevance_score=0.92,
            ),
            RetrievedChunk(
                chunk_id="openai-gpt4-002",
                content=(
                    "GPT-4 was released by OpenAI in March 2023. It is a large multimodal "
                    "model capable of processing both text and image inputs."
                ),
                source="gpt4_technical_report.pdf",
                relevance_score=0.88,
            ),
        ],
        retrieval_score=0.90,
    )


@pytest.fixture
def response_irrelevant() -> RetrievalResponse:
    """RetrievalResponse avec contexte hors-sujet (produits, pas historique)."""
    return RetrievalResponse(
        query_id="q-openai-founding",
        chunks=[
            RetrievedChunk(
                chunk_id="openai-products-010",
                content=(
                    "OpenAI offers a range of products including the ChatGPT consumer "
                    "application, the OpenAI API, and the DALL-E image generation service."
                ),
                source="openai_products_overview.pdf",
                relevance_score=0.41,
            ),
            RetrievedChunk(
                chunk_id="openai-pricing-011",
                content=(
                    "OpenAI pricing for the API is based on token usage. GPT-4 Turbo costs "
                    "$0.01 per 1K input tokens and $0.03 per 1K output tokens."
                ),
                source="openai_pricing_2024.pdf",
                relevance_score=0.35,
            ),
        ],
        retrieval_score=0.38,
    )


@pytest.fixture
def response_partial() -> RetrievalResponse:
    """RetrievalResponse avec contexte partiel : confirme l'entité, pas la date."""
    return RetrievalResponse(
        query_id="q-openai-founding",
        chunks=[
            RetrievedChunk(
                chunk_id="openai-gpt4-003",
                content=(
                    "OpenAI is the company behind GPT-4, one of the most capable "
                    "large language models to date. GPT-4 was released in early 2023."
                ),
                source="ai_landscape_2024.txt",
                relevance_score=0.60,
            ),
        ],
        retrieval_score=0.60,
    )


@pytest.fixture
def response_empty() -> RetrievalResponse:
    """RetrievalResponse sans aucun chunk récupéré."""
    return RetrievalResponse(
        query_id="q-openai-founding",
        chunks=[],
        retrieval_score=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Chemin nominal : contexte suffisant
# ─────────────────────────────────────────────────────────────────────────────


class TestSufficientContext:
    """Cas 1 — Contexte pertinent : is_sufficient=True, score >= seuil (0.70)."""

    @patch("reasoning.critic.critic.completion")
    def test_sufficient_context_returns_is_sufficient_true(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """Contexte pertinent → is_sufficient=True, relevance_score >= 0.70."""
        mock_completion.return_value = _make_llm_response(
            {
                "relevance_score": 0.88,
                "missing_aspects": "",
                "feedback": "",
            }
        )

        result = critic.evaluate(step_founding, response_relevant)

        assert isinstance(result, CriticEvaluation)
        assert result.is_sufficient is True
        assert result.relevance_score == pytest.approx(0.88)
        assert result.relevance_score >= critic.sufficiency_threshold
        assert result.feedback == ""
        mock_completion.assert_called_once()

    @patch("reasoning.critic.critic.completion")
    def test_step_id_is_propagated_from_planstep(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """step_id est repris depuis PlanStep.step_id — pas généré par le LLM."""
        mock_completion.return_value = _make_llm_response(
            {"relevance_score": 0.85, "missing_aspects": "", "feedback": ""}
        )

        result = critic.evaluate(step_founding, response_relevant)

        # step_founding.step_id == "step_2"
        assert result.step_id == "step_2"
        assert result.step_id == step_founding.step_id

    @patch("reasoning.critic.critic.completion")
    def test_score_exactly_at_threshold_is_sufficient(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """Score exactement à 0.70 (la limite) → is_sufficient=True (≥, pas >)."""
        mock_completion.return_value = _make_llm_response(
            {"relevance_score": 0.70, "missing_aspects": "", "feedback": ""}
        )

        result = critic.evaluate(step_founding, response_relevant)

        assert result.is_sufficient is True
        assert result.relevance_score == pytest.approx(0.70)

    @patch("reasoning.critic.critic.completion")
    def test_sufficient_context_missing_aspects_is_empty(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """Quand le contexte est suffisant, missing_aspects est vide."""
        mock_completion.return_value = _make_llm_response(
            {"relevance_score": 0.92, "missing_aspects": "", "feedback": ""}
        )

        result = critic.evaluate(step_founding, response_relevant)

        assert result.missing_aspects == []


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Contexte hors-sujet : is_sufficient=False, feedback non vide
# ─────────────────────────────────────────────────────────────────────────────


class TestIrrelevantContext:
    """Cas 2 — Contexte hors-sujet : is_sufficient=False, feedback non vide."""

    @patch("reasoning.critic.critic.completion")
    def test_irrelevant_context_returns_is_sufficient_false(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_irrelevant: RetrievalResponse,
    ) -> None:
        """Contexte hors-sujet → is_sufficient=False, score < 0.70."""
        mock_completion.return_value = _make_llm_response(
            {
                "relevance_score": 0.35,
                "missing_aspects": "founding year of OpenAI",
                "feedback": (
                    "Retrieved chunks cover products and pricing but do not mention "
                    "the founding date or year of establishment of OpenAI."
                ),
            }
        )

        result = critic.evaluate(step_founding, response_irrelevant)

        assert isinstance(result, CriticEvaluation)
        assert result.is_sufficient is False
        assert result.relevance_score == pytest.approx(0.35)
        assert result.relevance_score < critic.sufficiency_threshold
        mock_completion.assert_called_once()

    @patch("reasoning.critic.critic.completion")
    def test_irrelevant_context_feedback_is_not_empty(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_irrelevant: RetrievalResponse,
    ) -> None:
        """Contexte hors-sujet → feedback non vide (actionnable pour le Planner)."""
        mock_completion.return_value = _make_llm_response(
            {
                "relevance_score": 0.35,
                "missing_aspects": "founding year of OpenAI",
                "feedback": (
                    "Retrieved chunks cover products and pricing but do not mention "
                    "the founding date or year of establishment of OpenAI."
                ),
            }
        )

        result = critic.evaluate(step_founding, response_irrelevant)

        assert result.feedback != ""
        assert len(result.feedback) > 10  # feedback substantiel, pas juste " "

    @patch("reasoning.critic.critic.completion")
    def test_score_just_below_threshold_is_not_sufficient(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_irrelevant: RetrievalResponse,
    ) -> None:
        """Score juste en dessous de 0.70 → is_sufficient=False (< strict)."""
        mock_completion.return_value = _make_llm_response(
            {
                "relevance_score": 0.69,
                "missing_aspects": "founding year",
                "feedback": "Context is incomplete.",
            }
        )

        result = critic.evaluate(step_founding, response_irrelevant)

        assert result.is_sufficient is False
        assert result.relevance_score == pytest.approx(0.69)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Contexte partiel : missing_aspects correctement rempli
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialContext:
    """Cas 3 — Contexte partiel : missing_aspects non vide et cohérent."""

    @patch("reasoning.critic.critic.completion")
    def test_partial_context_missing_aspects_populated(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_partial: RetrievalResponse,
    ) -> None:
        """Contexte partiel → is_sufficient=False, missing_aspects non vide."""
        mock_completion.return_value = _make_llm_response(
            {
                "relevance_score": 0.50,
                "missing_aspects": "founding year of OpenAI | headquarters location",
                "feedback": (
                    "Context confirms OpenAI created GPT-4 but does not mention "
                    "the founding year or headquarters of the company."
                ),
            }
        )

        result = critic.evaluate(step_founding, response_partial)

        assert result.is_sufficient is False
        assert len(result.missing_aspects) > 0
        # Les aspects manquants doivent correspondre au contexte mocké :
        # la date de fondation et le siège ne sont pas dans response_partial
        assert any("founding" in aspect.lower() for aspect in result.missing_aspects)

    @patch("reasoning.critic.critic.completion")
    def test_partial_context_multiple_missing_aspects(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_partial: RetrievalResponse,
    ) -> None:
        """Plusieurs aspects manquants → liste avec plusieurs entrées."""
        mock_completion.return_value = _make_llm_response(
            {
                "relevance_score": 0.45,
                "missing_aspects": (
                    "founding year of OpenAI | "
                    "headquarters location | "
                    "names of original founders"
                ),
                "feedback": "Multiple key facts are missing from the retrieved context.",
            }
        )

        result = critic.evaluate(step_founding, response_partial)

        assert result.is_sufficient is False
        assert len(result.missing_aspects) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Garde anti-boucle : attribut max_retries
# ─────────────────────────────────────────────────────────────────────────────


class TestMaxRetriesGuard:
    """Cas 4 — Garde anti-boucle : max_retries exposé comme attribut paramétrable."""

    def test_default_max_retries_value(self) -> None:
        """La valeur par défaut de max_retries est 2 (critic_spec.md §5.1)."""
        critic = Critic()
        assert critic.max_retries == 2

    def test_custom_max_retries_is_stored(self) -> None:
        """max_retries est correctement stocké comme attribut d'instance."""
        critic = Critic(max_retries=1)
        assert critic.max_retries == 1

    def test_max_retries_accessible_for_external_counter(self) -> None:
        """Le routeur LangGraph peut lire critic.max_retries pour piloter le compteur."""
        critic = Critic(max_retries=3)
        # Simule la logique du routeur conditionnel :
        # compteur local initialisé à 0, incrémenté à chaque appel
        retries_for_step: dict[str, int] = {}
        step_id = "step_1"

        for _ in range(3):  # 3 tentatives simulées
            retries_for_step[step_id] = retries_for_step.get(step_id, 0) + 1

        # Après max_retries tentatives, la garde doit forcer la sortie
        assert retries_for_step[step_id] >= critic.max_retries

    def test_custom_sufficiency_threshold_stored(self) -> None:
        """sufficiency_threshold est paramétrable et stocké correctement."""
        critic = Critic(sufficiency_threshold=0.85)
        assert critic.sufficiency_threshold == pytest.approx(0.85)

    @patch("reasoning.critic.critic.completion")
    def test_custom_threshold_changes_is_sufficient_decision(
        self,
        mock_completion: MagicMock,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """Avec threshold=0.85, un score de 0.80 → is_sufficient=False."""
        strict_critic = Critic(
            model="ollama/test-model",
            sufficiency_threshold=0.85,
        )
        mock_completion.return_value = _make_llm_response(
            {"relevance_score": 0.80, "missing_aspects": "", "feedback": ""}
        )

        result = strict_critic.evaluate(step_founding, response_relevant)

        assert result.is_sufficient is False
        assert result.relevance_score == pytest.approx(0.80)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Fallback défensif : échec LLM
# ─────────────────────────────────────────────────────────────────────────────


class TestDefensiveFallback:
    """Cas 5 — Fallback défensif : échec LLM → is_sufficient=False, score=0.0."""

    @patch("reasoning.critic.critic.completion")
    def test_fallback_on_llm_timeout(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """Timeout LLM → fallback défensif, pas de crash, is_sufficient=False."""
        mock_completion.side_effect = TimeoutError("Ollama timeout")

        result = critic.evaluate(step_founding, response_relevant)

        assert isinstance(result, CriticEvaluation)
        assert result.is_sufficient is False
        assert result.relevance_score == pytest.approx(0.0)
        assert result.step_id == step_founding.step_id

    @patch("reasoning.critic.critic.completion")
    def test_fallback_on_connection_error(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """OSError (connexion refusée) → fallback défensif."""
        mock_completion.side_effect = OSError("Connection refused")

        result = critic.evaluate(step_founding, response_relevant)

        assert isinstance(result, CriticEvaluation)
        assert result.is_sufficient is False
        assert result.relevance_score == pytest.approx(0.0)

    @patch("reasoning.critic.critic.completion")
    def test_fallback_on_malformed_toon_response(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """Réponse LLM sans bloc TOON → ToonParseError → fallback défensif."""
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "I think the context is good enough to answer the question."
        mock_completion.return_value = mock_response

        result = critic.evaluate(step_founding, response_relevant)

        assert isinstance(result, CriticEvaluation)
        assert result.is_sufficient is False
        assert result.relevance_score == pytest.approx(0.0)

    @patch("reasoning.critic.critic.completion")
    def test_fallback_on_missing_relevance_score_in_toon(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """TOON valide mais sans relevance_score → ValueError → fallback."""
        mock_response = MagicMock()
        # Bloc TOON bien formé mais champ obligatoire absent
        mock_response.choices[0].message.content = (
            "<<<\n"
            "missing_aspects :: founding year\n"
            "feedback :: Context is insufficient.\n"
            ">>>"
        )
        mock_completion.return_value = mock_response

        result = critic.evaluate(step_founding, response_relevant)

        assert isinstance(result, CriticEvaluation)
        assert result.is_sufficient is False
        assert result.relevance_score == pytest.approx(0.0)

    @patch("reasoning.critic.critic.completion")
    def test_fallback_defensive_step_id_preserved(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """Le fallback défensif préserve le step_id de la PlanStep originale."""
        mock_completion.side_effect = RuntimeError("Unexpected internal error")

        result = critic.evaluate(step_founding, response_relevant)

        assert result.step_id == step_founding.step_id

    @patch("reasoning.critic.critic.completion")
    def test_fallback_feedback_is_informative(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_relevant: RetrievalResponse,
    ) -> None:
        """Le fallback produit un feedback signalant l'échec LLM (pas une chaîne vide)."""
        mock_completion.side_effect = TimeoutError("Ollama timeout")

        result = critic.evaluate(step_founding, response_relevant)

        # Le fallback défensif doit produire un feedback non vide
        assert result.feedback != ""
        assert "failed" in result.feedback.lower() or "re-retrieval" in result.feedback


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Cas spécial : chunks vides (sans appel LLM)
# ─────────────────────────────────────────────────────────────────────────────


class TestEmptyChunks:
    """Cas particulier — Aucun chunk récupéré : court-circuit sans appel LLM."""

    @patch("reasoning.critic.critic.completion")
    def test_empty_chunks_no_llm_call(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_empty: RetrievalResponse,
    ) -> None:
        """Aucun chunk → is_sufficient=False sans appel LLM (critic_spec.md §8)."""
        result = critic.evaluate(step_founding, response_empty)

        assert isinstance(result, CriticEvaluation)
        assert result.is_sufficient is False
        assert result.relevance_score == pytest.approx(0.0)
        # Le LLM NE doit PAS être appelé dans ce cas
        mock_completion.assert_not_called()

    @patch("reasoning.critic.critic.completion")
    def test_empty_chunks_step_id_preserved(
        self,
        mock_completion: MagicMock,
        critic: Critic,
        step_founding: PlanStep,
        response_empty: RetrievalResponse,
    ) -> None:
        """Même sans chunks, le step_id est préservé dans l'évaluation."""
        result = critic.evaluate(step_founding, response_empty)
        assert result.step_id == step_founding.step_id


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Paramètres du constructeur
# ─────────────────────────────────────────────────────────────────────────────


class TestConstructorParams:
    """Vérifie que tous les paramètres du constructeur sont correctement stockés."""

    def test_default_params(self) -> None:
        """Les valeurs par défaut correspondent à celles de critic_spec.md."""
        critic = Critic()
        assert critic.sufficiency_threshold == pytest.approx(0.70)
        assert critic.max_retries == 2
        assert critic.temperature == pytest.approx(0.0)
        assert critic.max_tokens == 256

    def test_custom_model_stored(self) -> None:
        """Le modèle personnalisé est stocké dans l'instance."""
        critic = Critic(model="ollama/qwen2.5:7b")
        assert critic.model == "ollama/qwen2.5:7b"
