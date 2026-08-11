"""
Client HTTP vers le module ACTION (astraexec).

Sérialise un `RetrievalRequest` (contrat figé, `contracts/action_interface.py`)
en JSON, l'envoie en `POST` vers l'endpoint `/retrieve` du module ACTION, et
désérialise la réponse en `RetrievalResponse`. Conforme au contrat décrit
dans `docs/ACTION_INTEGRATION_HANDOFF.md §7.1`.

Le module ACTION n'étant pas encore branché au moment de ce sprint, ce client
est conçu pour être substitué par un double de test (`Protocol` structurel)
sans dépendance à un serveur réel — voir `tests/integration/test_graph.py`.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import httpx
from dotenv import load_dotenv

from reasoning.contracts.action_interface import RetrievalRequest, RetrievalResponse

load_dotenv()

logger = logging.getLogger(__name__)

_ACTION_BASE_URL: str = os.getenv("ACTION_BASE_URL", "http://localhost:8000")
_DEFAULT_TIMEOUT: float = 10.0


class RetrievalClient(Protocol):
    """Interface structurelle d'un client de retrieval.

    Permet d'injecter un double de test (in-memory) dans le nœud `retrieve`
    du graphe sans dépendre d'une implémentation HTTP concrète — le module
    ACTION n'est pas encore disponible au moment de ce sprint.
    """

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Exécute une requête de retrieval et retourne la réponse."""
        ...


class ActionClient:
    """Client HTTP synchrone vers le module ACTION (`POST /retrieve`).

    Attributes:
        base_url: URL de base du module ACTION (ex: http://localhost:8000).
        timeout: Délai maximum d'attente en secondes avant de lever une
            exception (défensif — pas de blocage indéfini du graphe).
    """

    def __init__(
        self,
        base_url: str = _ACTION_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Envoie une RetrievalRequest au module ACTION et retourne la réponse.

        Args:
            request: La requête de retrieval, conforme au contrat figé.

        Returns:
            RetrievalResponse validée par Pydantic.

        Raises:
            httpx.RequestError: Erreur réseau (connexion refusée, DNS, etc.).
            httpx.TimeoutException: Le module ACTION n'a pas répondu à temps.
            httpx.HTTPStatusError: Réponse HTTP avec un code d'erreur (4xx/5xx).
            ValidationError: La réponse reçue ne satisfait pas le contrat
                `RetrievalResponse`.
        """
        url = f"{self.base_url}/retrieve"
        logger.debug(
            "ActionClient : POST %s (query_id=%s, hop_index=%d)",
            url,
            request.query_id,
            request.hop_index,
        )

        response = httpx.post(
            url,
            json=request.model_dump(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        return RetrievalResponse.model_validate(response.json())


__all__ = ["ActionClient", "RetrievalClient"]
