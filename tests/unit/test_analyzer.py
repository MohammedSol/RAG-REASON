"""
Tests unitaires — Sprint 2.3 : Query Analyzer (mocks LiteLLM).

Ces tests sont rapides et isolés : ils ne nécessitent PAS Ollama.
Toutes les réponses LLM sont simulées via `unittest.mock.patch`.

Exécution :
    uv run pytest tests/unit/test_analyzer.py -v --cov=src/reasoning/analyzer
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from reasoning.analyzer import QueryAnalyzer
from reasoning.contracts.internal_models import AnalysisResult, QueryType
from reasoning.shared.toon_utils import dump_dict_to_toon

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — constructeurs de réponses mockées
# ─────────────────────────────────────────────────────────────────────────────


def _make_llm_response(payload: dict[str, Any]) -> MagicMock:
    """Simule une réponse LLM au format TOON.

    Args:
        payload: Dictionnaire représentant les champs de classification.

    Returns:
        MagicMock dont l'attribut choices[0].message.content est un bloc TOON.
    """
    mock_response = MagicMock()
    mock_response.choices[0].message.content = dump_dict_to_toon(payload)
    return mock_response


# ─────────────────────────────────────────────────────────────────────────────
# Fixture partagée
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    """Instance QueryAnalyzer avec paramètres neutres pour les tests."""
    return QueryAnalyzer(
        model="ollama/test-model",
        api_base="http://localhost:11434",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Succès LLM : classifications correctes
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMSuccessPath:
    """Tests du chemin nominal : le LLM répond avec un JSON valide."""

    @patch("reasoning.analyzer.analyzer.completion")
    def test_simple_query_classification(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Le LLM retourne SIMPLE → AnalysisResult.query_type == SIMPLE."""
        mock_completion.return_value = _make_llm_response(
            {
                "query_type": "SIMPLE",
                "confidence": 0.97,
                "detected_entities": ["rétropropagation"],
                "reasoning_budget": 1,
            }
        )

        result = analyzer.analyze("Qu'est-ce que la rétropropagation ?")

        assert isinstance(result, AnalysisResult)
        assert result.query_type == QueryType.SIMPLE
        assert result.confidence == pytest.approx(0.97)
        assert result.reasoning_budget == 1
        assert "rétropropagation" in result.detected_entities
        mock_completion.assert_called_once()

    @patch("reasoning.analyzer.analyzer.completion")
    def test_multi_hop_query_classification(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Le LLM retourne MULTI_HOP → reasoning_budget == 3."""
        mock_completion.return_value = _make_llm_response(
            {
                "query_type": "MULTI_HOP",
                "confidence": 0.92,
                "detected_entities": ["LangChain", "LangSmith"],
                "reasoning_budget": 3,
            }
        )

        result = analyzer.analyze(
            "Qui a créé LangChain, et quel est son rôle chez LangSmith ?"
        )

        assert result.query_type == QueryType.MULTI_HOP
        assert result.reasoning_budget == 3
        assert result.confidence == pytest.approx(0.92)
        assert "LangChain" in result.detected_entities

    @patch("reasoning.analyzer.analyzer.completion")
    def test_ambiguous_query_classification(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Le LLM retourne AMBIGUOUS → reasoning_budget == 0."""
        mock_completion.return_value = _make_llm_response(
            {
                "query_type": "AMBIGUOUS",
                "confidence": 0.82,
                "detected_entities": ["réseaux"],
                "reasoning_budget": 0,
            }
        )

        result = analyzer.analyze("Explique-moi les réseaux.")

        assert result.query_type == QueryType.AMBIGUOUS
        assert result.reasoning_budget == 0

    @patch("reasoning.analyzer.analyzer.completion")
    def test_comparative_query_classification(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Le LLM retourne COMPARATIVE → reasoning_budget == 2."""
        mock_completion.return_value = _make_llm_response(
            {
                "query_type": "COMPARATIVE",
                "confidence": 0.94,
                "detected_entities": ["RAG", "fine-tuning"],
                "reasoning_budget": 2,
            }
        )

        result = analyzer.analyze(
            "Compare le RAG et le fine-tuning pour une application médicale."
        )

        assert result.query_type == QueryType.COMPARATIVE
        assert result.reasoning_budget == 2

    @patch("reasoning.analyzer.analyzer.completion")
    def test_llm_response_with_toon_markdown_block(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Le TOON dans un bloc ```toon ... ``` doit être extrait correctement."""
        payload = {
            "query_type": "SIMPLE",
            "confidence": 0.88,
            "detected_entities": ["Transformer"],
            "reasoning_budget": 1,
        }
        mock_response = MagicMock()
        toon_content = (
            dump_dict_to_toon(payload).replace("<<<", "").replace(">>>", "").strip()
        )
        mock_response.choices[0].message.content = "```toon\n" + toon_content + "\n```"
        mock_completion.return_value = mock_response

        result = analyzer.analyze("Qu'est-ce qu'un Transformer ?")

        assert result.query_type == QueryType.SIMPLE
        assert result.confidence == pytest.approx(0.88)

    @patch("reasoning.analyzer.analyzer.completion")
    def test_llm_response_with_toon_degraded_in_text(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Lignes TOON noyées dans du texte libre — mode dégradé."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "Voici la classification :\n"
            "query_type :: MULTI_HOP\n"
            "confidence :: 0.85\n"
            "detected_entities :: GPT-4 |\n"
            "reasoning_budget :: 3\n"
            "Fin de la réponse."
        )
        mock_completion.return_value = mock_response

        result = analyzer.analyze(
            "Quel est le PDG de l'entreprise qui a publié GPT-4 ?"
        )

        assert result.query_type == QueryType.MULTI_HOP


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Activation du fallback heuristique
# ─────────────────────────────────────────────────────────────────────────────


class TestFallbackHeuristics:
    """Tests du chemin de secours : le LLM échoue, les heuristiques prennent le relais."""

    @patch("reasoning.analyzer.analyzer.completion")
    def test_fallback_on_llm_timeout(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Simulation d'un timeout LLM → fallback heuristique, pas de crash."""
        mock_completion.side_effect = TimeoutError("Ollama timeout")

        result = analyzer.analyze(
            "Quel est le rôle d'un mécanisme d'attention dans un Transformer ?"
        )

        assert isinstance(result, AnalysisResult)
        assert result.confidence == pytest.approx(0.55)
        assert result.query_type in QueryType.__members__.values()

    @patch("reasoning.analyzer.analyzer.completion")
    def test_fallback_on_connection_error(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Simulation d'une erreur de connexion réseau → fallback heuristique."""
        mock_completion.side_effect = OSError("Connection refused")

        result = analyzer.analyze(
            "Compare BERT et GPT pour une tâche de classification."
        )

        assert isinstance(result, AnalysisResult)
        assert result.confidence == pytest.approx(0.55)
        assert result.query_type == QueryType.COMPARATIVE

    @patch("reasoning.analyzer.analyzer.completion")
    def test_fallback_on_invalid_json_response(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Le LLM renvoie du texte non-JSON → ValueError → fallback heuristique."""
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "Désolé, je ne peux pas classifier cette requête."
        mock_completion.return_value = mock_response

        result = analyzer.analyze(
            "Qui a créé LangChain, et ensuite quel est son rôle ?"
        )

        assert isinstance(result, AnalysisResult)
        assert result.confidence == pytest.approx(0.55)
        # "et ensuite" est un marqueur MULTI_HOP → le fallback doit le détecter
        assert result.query_type == QueryType.MULTI_HOP

    @patch("reasoning.analyzer.analyzer.completion")
    def test_fallback_on_malformed_json(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Le LLM renvoie un JSON syntaxiquement invalide → fallback."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"query_type": "SIMPLE", broken}'
        mock_completion.return_value = mock_response

        result = analyzer.analyze("Qu'est-ce que la photosynthèse ?")

        assert isinstance(result, AnalysisResult)
        assert result.confidence == pytest.approx(0.55)

    @patch("reasoning.analyzer.analyzer.completion")
    def test_fallback_on_pydantic_validation_error(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Le LLM renvoie un JSON valide mais non conforme au schéma → fallback."""
        mock_completion.return_value = _make_llm_response(
            {
                "query_type": "INEXISTANT",  # valeur non membre de QueryType
                "confidence": 0.9,
                "reasoning_budget": 1,
            }
        )

        result = analyzer.analyze("Qu'est-ce que le NLP ?")

        assert isinstance(result, AnalysisResult)
        assert result.confidence == pytest.approx(0.55)

    @patch("reasoning.analyzer.analyzer.completion")
    def test_fallback_on_unexpected_exception(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """Une exception non prévue (RuntimeError) est capturée → fallback, pas de crash."""
        mock_completion.side_effect = RuntimeError("Erreur inattendue")

        result = analyzer.analyze("Qu'est-ce que le gradient descent ?")

        assert isinstance(result, AnalysisResult)
        assert result.confidence == pytest.approx(0.55)

    def test_fallback_detects_comparative(self, analyzer: QueryAnalyzer) -> None:
        """Appel direct de _classify_with_heuristics → détecte COMPARATIVE."""
        result = analyzer._classify_with_heuristics(
            "Compare les performances de BM25 versus un retriever dense."
        )
        assert result.query_type == QueryType.COMPARATIVE
        assert result.confidence == pytest.approx(0.55)
        assert result.reasoning_budget == 2

    def test_fallback_detects_multi_hop(self, analyzer: QueryAnalyzer) -> None:
        """Appel direct de _classify_with_heuristics → détecte MULTI_HOP via marqueur."""
        result = analyzer._classify_with_heuristics(
            "Qui a fondé OpenAI, et ensuite quel est son rôle actuel ?"
        )
        assert result.query_type == QueryType.MULTI_HOP
        assert result.reasoning_budget == 3

    def test_fallback_detects_simple_by_default(self, analyzer: QueryAnalyzer) -> None:
        """Appel direct de _classify_with_heuristics → SIMPLE comme valeur par défaut."""
        result = analyzer._classify_with_heuristics(
            "Quelle est la définition du token dans un LLM ?"
        )
        assert result.query_type == QueryType.SIMPLE
        assert result.reasoning_budget == 1

    def test_fallback_detects_ambiguous_short_query(
        self, analyzer: QueryAnalyzer
    ) -> None:
        """Requête trop courte (< 3 tokens) → AMBIGUOUS via heuristique."""
        result = analyzer._classify_with_heuristics("Explique python")
        assert result.query_type == QueryType.AMBIGUOUS
        assert result.reasoning_budget == 0

    def test_fallback_generic_verb_triggers_ambiguous(
        self, analyzer: QueryAnalyzer
    ) -> None:
        """Verbe générique + terme court → AMBIGUOUS via heuristique."""
        result = analyzer._classify_with_heuristics("Explique le cloud")
        assert result.query_type == QueryType.AMBIGUOUS


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Cas limites de l'API publique
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests des cas limites de la méthode analyze()."""

    def test_empty_string_returns_ambiguous(self, analyzer: QueryAnalyzer) -> None:
        """Une requête vide doit retourner AMBIGUOUS sans appeler le LLM."""
        with patch("reasoning.analyzer.analyzer.completion") as mock_completion:
            result = analyzer.analyze("")
            mock_completion.assert_not_called()

        assert result.query_type == QueryType.AMBIGUOUS
        assert result.confidence == pytest.approx(0.99)

    def test_whitespace_only_returns_ambiguous(self, analyzer: QueryAnalyzer) -> None:
        """Une requête d'espaces seuls retourne AMBIGUOUS."""
        with patch("reasoning.analyzer.analyzer.completion") as mock_completion:
            result = analyzer.analyze("   ")
            mock_completion.assert_not_called()

        assert result.query_type == QueryType.AMBIGUOUS

    @patch("reasoning.analyzer.analyzer.completion")
    def test_result_is_analysis_result_instance(
        self, mock_completion: MagicMock, analyzer: QueryAnalyzer
    ) -> None:
        """La valeur retournée est toujours une instance d'AnalysisResult."""
        mock_completion.return_value = _make_llm_response(
            {
                "query_type": "SIMPLE",
                "confidence": 0.9,
                "detected_entities": [],
                "reasoning_budget": 1,
            }
        )
        result = analyzer.analyze("Test")
        assert isinstance(result, AnalysisResult)

    def test_analyzer_default_params(self) -> None:
        """Les paramètres par défaut de QueryAnalyzer sont correctement initialisés.

        Valeurs attendues après Solution A (diagnostic_analyzer.md) :
        - max_tokens=64 : réduit l'overhead KV-cache (~20 tokens réels pour un bloc TOON)
        - timeout=15.0  : évite le blocage indéfini si Ollama est lent
        """
        a = QueryAnalyzer()
        assert a.temperature == pytest.approx(0.0)
        assert a.max_tokens == 64  # Optimisé depuis 256 — voir diagnostic_analyzer.md
        assert a.timeout == pytest.approx(15.0)
        assert "ollama" in a.model.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Méthodes statiques internes
# ─────────────────────────────────────────────────────────────────────────────


class TestInternalMethods:
    """Tests unitaires des méthodes statiques privées."""

    def test_extract_toon_with_delimiters(self) -> None:
        """Bloc TOON avec délimiteurs <<< >>> → retourné tel quel."""
        raw = "<<<\nquery_type :: SIMPLE\nconfidence :: 0.9\n>>>"
        result = QueryAnalyzer._extract_toon(raw)
        assert "<<<" in result and ">>>" in result
        assert "query_type :: SIMPLE" in result

    def test_extract_toon_from_markdown_block(self) -> None:
        """Bloc TOON dans un bloc Markdown → extraction correcte."""
        raw = "```toon\nquery_type :: SIMPLE\nreasoning_budget :: 1\n```"
        result = QueryAnalyzer._extract_toon(raw)
        assert "query_type :: SIMPLE" in result

    def test_extract_toon_degraded_mode(self) -> None:
        """Lignes clé :: valeur sans délimiteurs → mode dégradé."""
        raw = "query_type :: MULTI_HOP\nreasoning_budget :: 3"
        result = QueryAnalyzer._extract_toon(raw)
        assert "query_type :: MULTI_HOP" in result

    def test_extract_toon_raises_on_no_toon(self) -> None:
        """Texte sans bloc TOON ni ligne clé :: valeur → ValueError."""
        with pytest.raises(ValueError, match="Aucun bloc TOON"):
            QueryAnalyzer._extract_toon("Aucun contenu TOON ici du tout.")

    def test_extract_entities_filters_stop_words(self) -> None:
        """Les stop words ne doivent pas apparaître dans les entités."""
        tokens = ["quel", "est", "langchain", "framework", "pour"]
        entities = QueryAnalyzer._extract_entities(tokens)
        assert "quel" not in entities
        assert "langchain" in entities

    def test_extract_entities_max_five(self) -> None:
        """Maximum 5 entités retournées."""
        tokens = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        entities = QueryAnalyzer._extract_entities(tokens)
        assert len(entities) <= 5

    def test_is_ambiguous_short_query(self) -> None:
        """Requête < 3 tokens → True."""
        assert QueryAnalyzer._is_ambiguous("test", ["test"]) is True

    def test_is_ambiguous_polysemic_term_alone(self) -> None:
        """Terme polysémique seul comme token significatif → True."""
        assert (
            QueryAnalyzer._is_ambiguous("explique python", ["explique", "python"])
            is True
        )

    def test_is_ambiguous_false_for_specific_query(self) -> None:
        """Requête précise → False."""
        normalized = "quel est le rôle du gradient descent en machine learning"
        tokens = normalized.split()
        assert QueryAnalyzer._is_ambiguous(normalized, tokens) is False
