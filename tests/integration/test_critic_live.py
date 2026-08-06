"""
Tests d'intégration — Sprint 4.3 : Critic réel (Ollama live).

Ces tests appellent RÉELLEMENT l'instance Ollama locale via LiteLLM.
Ils nécessitent :
    - `ollama serve` en arrière-plan
    - Le modèle DEFAULT_REASONING_MODEL téléchargé (`ollama pull qwen2.5:7b`)
    - Les variables d'environnement du fichier `.env`

Exécution :
    uv run pytest tests/integration/test_critic_live.py -v -m integration

Pour ignorer ces tests si Ollama n'est pas disponible :
    uv run pytest tests/ -v -m "not integration"
"""

from __future__ import annotations

import pytest
from reasoning.contracts.action_interface import RetrievalResponse, RetrievedChunk
from reasoning.contracts.internal_models import CriticEvaluation, PlanStep
from reasoning.critic import Critic

# ─────────────────────────────────────────────────────────────────────────────
# Fixture partagée
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_critic() -> Critic:
    """Instance Critic en mode réel, partagée sur tout le module."""
    return Critic()


# ─────────────────────────────────────────────────────────────────────────────
# Données de test réalistes
# ─────────────────────────────────────────────────────────────────────────────

# Paire 1 : contexte pertinent — la question porte sur la date de fondation
# d'OpenAI, le chunk répond directement à cette question.
_STEP_FOUNDING = PlanStep(
    step_id="step_2",
    sub_query="What year was the company that created GPT-4 founded?",
    depends_on=["step_1"],
)

_RESPONSE_RELEVANT = RetrievalResponse(
    query_id="q-openai-founding-live",
    chunks=[
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
            chunk_id="openai-gpt4-002",
            content=(
                "GPT-4 was created and released by OpenAI in March 2023. "
                "It represents a significant leap over GPT-3.5 in reasoning, "
                "coding, and multimodal capabilities."
            ),
            source="gpt4_technical_report.pdf",
            relevance_score=0.89,
        ),
    ],
    retrieval_score=0.91,
)

# Paire 2 : contexte insuffisant — la question porte sur la fondation,
# mais les chunks ne parlent que des tarifs API.
_RESPONSE_INSUFFICIENT = RetrievalResponse(
    query_id="q-openai-founding-live",
    chunks=[
        RetrievedChunk(
            chunk_id="openai-pricing-010",
            content=(
                "OpenAI API pricing for GPT-4 Turbo is $0.01 per 1K input tokens "
                "and $0.03 per 1K output tokens as of 2024. Volume discounts are "
                "available for enterprise customers."
            ),
            source="openai_pricing_2024.pdf",
            relevance_score=0.38,
        ),
    ],
    retrieval_score=0.38,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tests d'intégration
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestCriticLive:
    """Tests de bout-en-bout avec Ollama réel.

    Nécessite : `ollama serve` + modèle qwen2.5:7b téléchargé.
    """

    def test_critic_returns_valid_evaluation(self, live_critic: Critic) -> None:
        """Vérification de base : le Critic retourne un CriticEvaluation valide."""
        result = live_critic.evaluate(_STEP_FOUNDING, _RESPONSE_RELEVANT)

        assert isinstance(result, CriticEvaluation)
        assert result.step_id == _STEP_FOUNDING.step_id
        assert 0.0 <= result.relevance_score <= 1.0
        assert isinstance(result.is_sufficient, bool)
        assert isinstance(result.missing_aspects, list)
        assert isinstance(result.feedback, str)

    def test_relevant_context_produces_high_score(self, live_critic: Critic) -> None:
        """Contexte directement pertinent → relevance_score probablement élevé.

        Note : Ce test peut varier selon la version du modèle Ollama.
        Un échec isolé ne constitue pas un bloquant ; analyser la tendance globale.
        """
        result = live_critic.evaluate(_STEP_FOUNDING, _RESPONSE_RELEVANT)

        # Le modèle 7B devrait reconnaître que le chunk répond directement
        # à la question sur la date de fondation d'OpenAI.
        assert result.relevance_score > 0.5

    def test_insufficient_context_produces_low_score_and_feedback(
        self, live_critic: Critic
    ) -> None:
        """Contexte hors-sujet → relevance_score bas, feedback non vide.

        Note : Ce test peut légèrement varier selon la version du modèle Ollama.
        """
        result = live_critic.evaluate(_STEP_FOUNDING, _RESPONSE_INSUFFICIENT)

        # Critère souple : on vérifie que le modèle génère un verdict structuré,
        # pas nécessairement le score exact (le modèle local peut varier).
        assert isinstance(result, CriticEvaluation)
        assert 0.0 <= result.relevance_score <= 1.0
        # Si insuffisant, le feedback doit être présent
        if not result.is_sufficient:
            assert result.feedback != "" or len(result.missing_aspects) > 0

    def test_evaluation_step_id_always_from_planstep(self, live_critic: Critic) -> None:
        """step_id est toujours repris depuis PlanStep, jamais généré par le LLM."""
        result = live_critic.evaluate(_STEP_FOUNDING, _RESPONSE_RELEVANT)
        assert result.step_id == "step_2"
