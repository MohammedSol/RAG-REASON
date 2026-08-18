"""
Tests unitaires — Sprint I2 : ActionClient (client HTTP vers le module ACTION).

`httpx.post` est mocké — aucun appel réseau réel n'est effectué.

Contrat vérifié ici (Sprint I2) : le client échoue FRANCHEMENT. Toute
anomalie lève une exception typée dérivant de `ActionClientError` ; le client
ne retourne jamais de réponse vide déguisée en succès. Les tests couvrent les
six cas exigés : réponse nominale, HTTP 400, HTTP 500, timeout, `query_id`
non corrélé, corps non conforme au contrat.

Exécution :
    uv run pytest tests/unit/test_action_client.py -v --cov=reasoning.action_client
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from reasoning.action_client import (
    ActionClient,
    ActionClientError,
    ActionHTTPError,
    ActionProtocolError,
    ActionUnavailableError,
)
from reasoning.contracts.action_interface import RetrievalRequest, RetrievalResponse

# ─────────────────────────────────────────────────────────────────────────────
# Helpers et fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _http_response(
    status_code: int = 200,
    payload: Any = None,
    text: str = "",
    json_raises: bool = False,
) -> MagicMock:
    """Fabrique une fausse réponse httpx.

    Args:
        status_code: Code HTTP simulé.
        payload: Objet retourné par `.json()`.
        text: Corps textuel, utilisé dans les messages d'erreur.
        json_raises: Si True, `.json()` lève `ValueError` (corps non-JSON).
    """
    response = MagicMock()
    response.status_code = status_code
    response.is_success = 200 <= status_code < 300
    response.text = text
    if json_raises:
        response.json.side_effect = ValueError("Expecting value: line 1 column 1")
    else:
        response.json.return_value = payload
    return response


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


def _valid_payload(query_id: str = "q-openai-founding") -> dict[str, Any]:
    """Corps de réponse conforme au contrat RetrievalResponse."""
    return {
        "query_id": query_id,
        "chunks": [
            {
                "chunk_id": "openai-history-001",
                "content": "OpenAI was founded in December 2015.",
                "source": "OpenAI.txt",
                "relevance_score": 0.93,
            }
        ],
        "retrieval_score": 0.90,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cas 1 — Réponse nominale
# ─────────────────────────────────────────────────────────────────────────────


class TestNominalResponse:
    """Cas 1 — le module ACTION répond correctement."""

    @patch("reasoning.action_client.httpx.post")
    def test_valid_response_is_deserialized(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Une réponse 200 conforme est désérialisée en RetrievalResponse."""
        mock_post.return_value = _http_response(200, _valid_payload())

        response = client.retrieve(request_founding)

        assert isinstance(response, RetrievalResponse)
        assert response.query_id == "q-openai-founding"
        assert len(response.chunks) == 1
        assert response.chunks[0].chunk_id == "openai-history-001"
        assert response.chunks[0].source == "OpenAI.txt"
        assert response.chunks[0].relevance_score == pytest.approx(0.93)
        assert response.retrieval_score == pytest.approx(0.90)

    @patch("reasoning.action_client.httpx.post")
    def test_request_is_posted_to_retrieve_endpoint(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """La requête cible {base_url}/retrieve avec le JSON du contrat."""
        mock_post.return_value = _http_response(200, _valid_payload())

        client.retrieve(request_founding)

        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == "http://action-module-test:8000/retrieve"
        assert call_kwargs["json"]["query_id"] == "q-openai-founding"
        assert call_kwargs["json"]["sub_query"] == "What year was OpenAI founded?"
        assert call_kwargs["json"]["hop_index"] == 0
        assert call_kwargs["json"]["top_k"] == 5
        assert call_kwargs["timeout"] == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Cas 2 et 3 — Codes HTTP d'erreur
# ─────────────────────────────────────────────────────────────────────────────


class TestHTTPErrors:
    """Cas 2 et 3 — tout code non-2xx lève ActionHTTPError."""

    @patch("reasoning.action_client.httpx.post")
    def test_http_400_raises_typed_error(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """HTTP 400 (RetrievalError côté ACTION) → ActionHTTPError."""
        mock_post.return_value = _http_response(
            400, text='{"detail":"moteur en erreur","query_id":"q-openai-founding"}'
        )

        with pytest.raises(ActionHTTPError) as exc_info:
            client.retrieve(request_founding)

        assert exc_info.value.status_code == 400
        assert "moteur en erreur" in exc_info.value.body
        assert isinstance(exc_info.value, ActionClientError)

    @patch("reasoning.action_client.httpx.post")
    def test_http_500_raises_typed_error(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """HTTP 500 → ActionHTTPError portant le code exact."""
        mock_post.return_value = _http_response(500, text="Internal Server Error")

        with pytest.raises(ActionHTTPError) as exc_info:
            client.retrieve(request_founding)

        assert exc_info.value.status_code == 500
        assert "Internal Server Error" in exc_info.value.body

    @patch("reasoning.action_client.httpx.post")
    def test_error_status_never_returns_empty_response(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Un code d'erreur ne doit jamais devenir une réponse vide silencieuse."""
        mock_post.return_value = _http_response(503, text="Service Unavailable")

        with pytest.raises(ActionHTTPError):
            client.retrieve(request_founding)
        # `.json()` n'est jamais consulté sur un statut d'erreur : le client
        # s'arrête avant toute tentative de désérialisation.
        mock_post.return_value.json.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Cas 4 — Indisponibilité du module ACTION
# ─────────────────────────────────────────────────────────────────────────────


class TestUnavailable:
    """Cas 4 — timeout et erreurs réseau → ActionUnavailableError."""

    @patch("reasoning.action_client.httpx.post")
    def test_timeout_raises_typed_error(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Un timeout lève ActionUnavailableError, jamais httpx brut."""
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(ActionUnavailableError) as exc_info:
            client.retrieve(request_founding)

        assert "5s dépassé" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, httpx.TimeoutException)

    @patch("reasoning.action_client.httpx.post")
    def test_connection_error_raises_typed_error(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Connexion refusée → ActionUnavailableError mentionnant l'URL."""
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(ActionUnavailableError) as exc_info:
            client.retrieve(request_founding)

        assert "http://action-module-test:8000/retrieve" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


# ─────────────────────────────────────────────────────────────────────────────
# Cas 5 — Corrélation query_id
# ─────────────────────────────────────────────────────────────────────────────


class TestQueryIdCorrelation:
    """Cas 5 — un query_id non corrélé est une erreur, pas un détail."""

    @patch("reasoning.action_client.httpx.post")
    def test_mismatched_query_id_raises(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Réponse portant un autre query_id → ActionProtocolError."""
        mock_post.return_value = _http_response(
            200, _valid_payload(query_id="q-une-autre-sous-question")
        )

        with pytest.raises(ActionProtocolError) as exc_info:
            client.retrieve(request_founding)

        message = str(exc_info.value)
        assert "q-openai-founding" in message
        assert "q-une-autre-sous-question" in message

    @patch("reasoning.action_client.httpx.post")
    def test_matching_query_id_is_accepted(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Contrôle symétrique : un query_id identique passe sans erreur."""
        mock_post.return_value = _http_response(200, _valid_payload())

        response = client.retrieve(request_founding)

        assert response.query_id == request_founding.query_id


# ─────────────────────────────────────────────────────────────────────────────
# Cas 6 — Corps non conforme au contrat
# ─────────────────────────────────────────────────────────────────────────────


class TestProtocolViolations:
    """Cas 6 — un corps non conforme lève ActionProtocolError."""

    @patch("reasoning.action_client.httpx.post")
    def test_missing_contract_field_raises(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Champ `chunks` absent → ActionProtocolError, pas ValidationError nue."""
        mock_post.return_value = _http_response(200, {"query_id": "q-openai-founding"})

        with pytest.raises(ActionProtocolError) as exc_info:
            client.retrieve(request_founding)

        assert "RetrievalResponse" in str(exc_info.value)

    @patch("reasoning.action_client.httpx.post")
    def test_wrong_field_type_raises(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """`relevance_score` textuel non numérique → ActionProtocolError."""
        payload = _valid_payload()
        payload["chunks"][0]["relevance_score"] = "tres pertinent"
        mock_post.return_value = _http_response(200, payload)

        with pytest.raises(ActionProtocolError):
            client.retrieve(request_founding)

    @patch("reasoning.action_client.httpx.post")
    def test_non_json_body_raises(
        self,
        mock_post: MagicMock,
        client: ActionClient,
        request_founding: RetrievalRequest,
    ) -> None:
        """Corps non décodable en JSON → ActionProtocolError."""
        mock_post.return_value = _http_response(200, json_raises=True)

        with pytest.raises(ActionProtocolError) as exc_info:
            client.retrieve(request_founding)

        assert "JSON" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# Constructeur
# ─────────────────────────────────────────────────────────────────────────────


class TestConstructorParams:
    """Paramètres du constructeur d'ActionClient."""

    def test_default_timeout_covers_cold_start(self) -> None:
        """Le timeout par défaut (30 s) couvre la construction d'index (~5 s)."""
        assert ActionClient().timeout == pytest.approx(30.0)

    def test_custom_timeout_stored(self) -> None:
        """Un timeout explicite est conservé tel quel."""
        assert ActionClient(timeout=3.5).timeout == pytest.approx(3.5)

    def test_base_url_trailing_slash_stripped(self) -> None:
        """Un base_url avec slash final ne produit pas de double slash."""
        assert (
            ActionClient(base_url="http://action-module:8000/").base_url
            == "http://action-module:8000"
        )

    def test_all_errors_share_a_common_base(self) -> None:
        """Les trois exceptions dérivent d'ActionClientError, interceptable seul."""
        assert issubclass(ActionUnavailableError, ActionClientError)
        assert issubclass(ActionHTTPError, ActionClientError)
        assert issubclass(ActionProtocolError, ActionClientError)
