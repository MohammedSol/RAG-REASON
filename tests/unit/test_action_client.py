"""
Tests unitaires — Sprint 6.2 : ActionClient (client HTTP vers le module ACTION).

`httpx` est mocké — aucun serveur réel n'est requis (le module ACTION n'est
pas encore branché, cf. docs/graph_spec.md §7).

Exécution :
    uv run pytest tests/unit/test_action_client.py -v --cov=reasoning.action_client
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError
from reasoning.action_client import ActionClient
from reasoning.contracts.action_interface import RetrievalRequest, RetrievalResponse


@pytest.fixture
def client() -> ActionClient:
    """Instance ActionClient pointant vers une URL de test."""
    return ActionClient(base_url="http://action-module-test:8000", timeout=5.0)


@pytest.fixture
def request_founding() -> RetrievalRequest:
    """RetrievalRequest réaliste : recherche sur la fondation d'OpenAI."""
    return RetrievalRequest(
        query_id="q-openai-founding",
        sub_query="What year was OpenAI founded?",
        hop_index=0,
        top_k=5,
    )


class TestActionClientSuccess:
    """Cas nominal : le module ACTION répond avec un JSON valide."""

    @patch("reasoning.action_client.httpx.post")
    def test_retrieve_returns_validated_response(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Une réponse HTTP 200 valide est désérialisée en RetrievalResponse."""
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "query_id": "q-openai-founding",
            "chunks": [
                {
                    "chunk_id": "openai-history-001",
                    "content": "OpenAI was founded in December 2015.",
                    "source": "openai_history.pdf",
                    "relevance_score": 0.93,
                }
            ],
            "retrieval_score": 0.90,
        }
        mock_http_response.raise_for_status.return_value = None
        mock_post.return_value = mock_http_response

        response = client.retrieve(request_founding)

        assert isinstance(response, RetrievalResponse)
        assert response.query_id == "q-openai-founding"
        assert len(response.chunks) == 1
        assert response.chunks[0].chunk_id == "openai-history-001"

    @patch("reasoning.action_client.httpx.post")
    def test_retrieve_posts_to_correct_endpoint(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """La requête POST cible bien {base_url}/retrieve avec le JSON du contrat."""
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "query_id": "q-openai-founding",
            "chunks": [],
            "retrieval_score": None,
        }
        mock_http_response.raise_for_status.return_value = None
        mock_post.return_value = mock_http_response

        client.retrieve(request_founding)

        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == "http://action-module-test:8000/retrieve"
        assert call_kwargs["json"]["query_id"] == "q-openai-founding"
        assert call_kwargs["json"]["sub_query"] == "What year was OpenAI founded?"
        assert call_kwargs["timeout"] == pytest.approx(5.0)

    def test_base_url_trailing_slash_stripped(self) -> None:
        """Un base_url avec slash final ne produit pas de double slash dans l'URL."""
        client_with_slash = ActionClient(base_url="http://action-module:8000/")
        assert client_with_slash.base_url == "http://action-module:8000"


class TestActionClientErrors:
    """Cas d'erreur : réseau, timeout, statut HTTP, réponse invalide."""

    @patch("reasoning.action_client.httpx.post")
    def test_connection_error_propagates(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Une erreur de connexion est propagée (gérée par le nœud retrieve)."""
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(httpx.ConnectError):
            client.retrieve(request_founding)

    @patch("reasoning.action_client.httpx.post")
    def test_timeout_propagates(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Un timeout est propagé sans être avalé silencieusement."""
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(httpx.TimeoutException):
            client.retrieve(request_founding)

    @patch("reasoning.action_client.httpx.post")
    def test_http_error_status_propagates(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Un statut HTTP d'erreur (5xx) lève HTTPStatusError via raise_for_status."""
        mock_http_response = MagicMock()
        mock_http_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        mock_post.return_value = mock_http_response

        with pytest.raises(httpx.HTTPStatusError):
            client.retrieve(request_founding)

    @patch("reasoning.action_client.httpx.post")
    def test_malformed_response_raises_validation_error(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Un JSON ne respectant pas le contrat RetrievalResponse lève ValidationError."""
        mock_http_response = MagicMock()
        mock_http_response.raise_for_status.return_value = None
        # Champ 'chunks' manquant — viole le contrat RetrievalResponse
        mock_http_response.json.return_value = {"query_id": "q-openai-founding"}
        mock_post.return_value = mock_http_response

        with pytest.raises(ValidationError):
            client.retrieve(request_founding)


class TestConstructorParams:
    """Paramètres du constructeur d'ActionClient."""

    def test_default_timeout_value(self) -> None:
        client = ActionClient()
        assert client.timeout == pytest.approx(10.0)

    def test_custom_timeout_stored(self) -> None:
        client = ActionClient(timeout=3.5)
        assert client.timeout == pytest.approx(3.5)
