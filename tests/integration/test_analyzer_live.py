"""
Tests d'intégration — Sprint 2.3 : Query Analyzer réel (Ollama live).

Ces tests appellent RÉELLEMENT l'instance Ollama locale via LiteLLM.
Ils nécessitent :
    - `ollama serve` en arrière-plan
    - Le modèle DEFAULT_FAST_MODEL téléchargé (`ollama pull qwen2.5:3b`)
    - Les variables d'environnement du fichier `.env`

Exécution :
    uv run pytest tests/integration/ -v -m integration

Pour ignorer ces tests si Ollama n'est pas disponible :
    uv run pytest tests/ -v -m "not integration"
"""

from __future__ import annotations

import pytest
from reasoning.analyzer import QueryAnalyzer
from reasoning.contracts.internal_models import AnalysisResult, QueryType

# ─────────────────────────────────────────────────────────────────────────────
# Jeu de données d'intégration : 15 requêtes avec classification attendue
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_CASES: list[tuple[str, QueryType]] = [
    # ── SIMPLE (5 cas) ────────────────────────────────────────────────────────
    (
        "Qu'est-ce que la rétropropagation ?",
        QueryType.SIMPLE,
    ),
    (
        "Quel est le rôle d'un Transformer dans le NLP ?",
        QueryType.SIMPLE,
    ),
    (
        "Quand a été fondée OpenAI ?",
        QueryType.SIMPLE,
    ),
    (
        "Définition de l'entropie croisée en machine learning.",
        QueryType.SIMPLE,
    ),
    (
        "Qu'est-ce que LangGraph ?",
        QueryType.SIMPLE,
    ),
    # ── MULTI_HOP (4 cas) ─────────────────────────────────────────────────────
    (
        "Qui a créé LangChain, et quel est son rôle chez LangSmith ?",
        QueryType.MULTI_HOP,
    ),
    (
        "Quel est le PDG de l'entreprise qui a publié GPT-4 ?",
        QueryType.MULTI_HOP,
    ),
    (
        "Depuis que l'attention a été introduite, comment a évolué la taille des LLMs ?",
        QueryType.MULTI_HOP,
    ),
    (
        "Qui a inventé le mécanisme d'attention, et dans quelle entreprise travaille-t-il ?",
        QueryType.MULTI_HOP,
    ),
    # ── COMPARATIVE (3 cas) ───────────────────────────────────────────────────
    (
        "Compare le RAG et le fine-tuning pour une application médicale.",
        QueryType.COMPARATIVE,
    ),
    (
        "Quelle est la différence entre BM25 et un retriever dense ?",
        QueryType.COMPARATIVE,
    ),
    (
        "LangGraph versus AutoGen : avantages et inconvénients ?",
        QueryType.COMPARATIVE,
    ),
    # ── AMBIGUOUS (3 cas) ─────────────────────────────────────────────────────
    (
        "Explique-moi les réseaux.",
        QueryType.AMBIGUOUS,
    ),
    (
        "Comment fonctionne Python ?",
        QueryType.AMBIGUOUS,
    ),
    (
        "Parle-moi de l'agent.",
        QueryType.AMBIGUOUS,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Fixture partagée
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live_analyzer() -> QueryAnalyzer:
    """Instance QueryAnalyzer en mode réel, partagée sur tout le module."""
    return QueryAnalyzer()


# ─────────────────────────────────────────────────────────────────────────────
# Tests d'intégration
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestAnalyzerLive:
    """Tests de bout-en-bout avec Ollama réel.

    Nécessite : `ollama serve` + modèle qwen2.5:3b téléchargé.
    """

    def test_analyzer_returns_valid_result(self, live_analyzer: QueryAnalyzer) -> None:
        """Vérification de base : l'analyzer retourne un AnalysisResult valide."""
        result = live_analyzer.analyze("Qu'est-ce que la rétropropagation ?")
        assert isinstance(result, AnalysisResult)
        assert result.query_type in QueryType.__members__.values()
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning_budget >= 0

    @pytest.mark.parametrize("query,expected_type", INTEGRATION_CASES)
    def test_classification_accuracy(
        self,
        live_analyzer: QueryAnalyzer,
        query: str,
        expected_type: QueryType,
    ) -> None:
        """Vérifie que le modèle classifie correctement les 15 requêtes de référence.

        Note : Ce test peut légèrement varier selon la version du modèle Ollama.
        Un échec isolé ne constitue pas un blocant ; analyser la tendance globale.
        """
        result = live_analyzer.analyze(query)

        assert isinstance(
            result, AnalysisResult
        ), f"Attendu AnalysisResult, obtenu {type(result)} pour : {query!r}"
        assert result.query_type == expected_type, (
            f"Requête : {query!r}\n"
            f"  Attendu  : {expected_type}\n"
            f"  Obtenu   : {result.query_type}\n"
            f"  Confiance: {result.confidence:.2f}"
        )

    def test_reasoning_budget_coherence(self, live_analyzer: QueryAnalyzer) -> None:
        """Le reasoning_budget doit être cohérent avec le query_type retourné."""
        budget_map = {
            QueryType.SIMPLE: 1,
            QueryType.MULTI_HOP: 3,
            QueryType.COMPARATIVE: 2,
            QueryType.AMBIGUOUS: 0,
        }

        test_queries = [
            "Qu'est-ce que BERT ?",
            "Compare BERT et GPT.",
            "Qui a fondé Hugging Face, et quel est son modèle phare ?",
        ]

        for query in test_queries:
            result = live_analyzer.analyze(query)
            expected_budget = budget_map[result.query_type]
            assert result.reasoning_budget == expected_budget, (
                f"Budget incohérent pour {query!r} : "
                f"type={result.query_type}, budget={result.reasoning_budget}, "
                f"attendu={expected_budget}"
            )

    def test_confidence_is_high_for_llm_path(
        self, live_analyzer: QueryAnalyzer
    ) -> None:
        """Quand le LLM répond correctement, confidence > 0.55 (> fallback)."""
        result = live_analyzer.analyze("Qu'est-ce que la rétropropagation ?")
        assert result.confidence > 0.55, (
            f"Confidence {result.confidence:.2f} suggère l'activation du fallback "
            "alors qu'Ollama est censé être disponible."
        )

    def test_detected_entities_is_list(self, live_analyzer: QueryAnalyzer) -> None:
        """detected_entities est toujours une liste (possiblement vide)."""
        result = live_analyzer.analyze("Quel est le rôle d'un Transformer ?")
        assert isinstance(result.detected_entities, list)

    def test_empty_query_no_llm_call(self, live_analyzer: QueryAnalyzer) -> None:
        """Une requête vide retourne AMBIGUOUS sans appeler Ollama."""
        result = live_analyzer.analyze("")
        assert result.query_type == QueryType.AMBIGUOUS
        assert result.reasoning_budget == 0
