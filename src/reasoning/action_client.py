"""
Client HTTP vers le module ACTION (astraexec).

Sérialise un `RetrievalRequest` (contrat figé, `contracts/action_interface.py`)
en JSON, l'envoie en `POST` vers l'endpoint `/retrieve` du module ACTION, et
désérialise la réponse en `RetrievalResponse`. Conforme au contrat décrit
dans `docs/ACTION_INTEGRATION_HANDOFF.md §7.1`.

Principe de conception (Sprint I2) : ce client échoue FRANCHEMENT. Toute
anomalie — code HTTP non-2xx, timeout, réponse malformée, `query_id` qui ne
correspond pas — lève une exception typée dérivant de `ActionClientError`.
Il ne retourne jamais de réponse vide déguisée en succès : c'est au nœud
`retrieve` du graphe, et à lui seul, de décider du repli fail-closed.
Confondre « aucun résultat » et « le service est tombé » rendrait toute
mesure de qualité du retrieval ininterprétable.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from reasoning.contracts.action_interface import RetrievalRequest, RetrievalResponse

load_dotenv()

logger = logging.getLogger(__name__)

_ACTION_BASE_URL: str = os.getenv("ACTION_BASE_URL", "http://localhost:8000")

# Timeout par défaut : 30 s.
#
# Justification mesurée (Sprint I1) : le module ACTION construit son index au
# tout premier appel — 1966 documents, 3239 chunks, ~5 s mesurées, auxquelles
# s'ajoutent le démarrage d'uvicorn et la lecture disque. Les appels suivants
# répondent en quelques dizaines de millisecondes. 30 s couvre donc largement
# le pire cas (démarrage à froid) tout en garantissant que le graphe ne se
# bloque jamais indéfiniment : la borne reste finie et connue.
_DEFAULT_TIMEOUT: float = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions typées
# ─────────────────────────────────────────────────────────────────────────────


class ActionClientError(Exception):
    """Erreur de communication avec le module ACTION.

    Classe de base : un appelant peut intercepter ce seul type pour traiter
    indifféremment toutes les défaillances du client.
    """


class ActionUnavailableError(ActionClientError):
    """Le module ACTION est injoignable ou n'a pas répondu à temps.

    Couvre les erreurs réseau (connexion refusée, DNS) et les timeouts.
    """


class ActionHTTPError(ActionClientError):
    """Le module ACTION a répondu avec un code HTTP non-2xx.

    Attributes:
        status_code: Le code HTTP reçu.
        body: Le corps de la réponse, tronqué, pour diagnostic.
    """

    def __init__(self, status_code: int, body: str = "") -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"module ACTION → HTTP {status_code} : {body[:200]}")


class ActionProtocolError(ActionClientError):
    """La réponse ne respecte pas le contrat attendu.

    Couvre deux cas distincts mais de même gravité : un corps qui n'est pas
    un `RetrievalResponse` valide, et une réponse dont le `query_id` ne
    correspond pas à celui envoyé.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Protocole et implémentation
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalClient(Protocol):
    """Interface structurelle d'un client de retrieval.

    Permet d'injecter un double de test (in-memory) dans le nœud `retrieve`
    du graphe sans dépendre d'une implémentation HTTP concrète.
    """

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Exécute une requête de retrieval et retourne la réponse."""
        ...


class ActionClient:
    """Client HTTP synchrone vers le module ACTION (`POST /retrieve`).

    Attributes:
        base_url: URL de base du module ACTION (défaut : `ACTION_BASE_URL`
            de l'environnement, sinon `http://localhost:8000`).
        timeout: Délai maximum d'attente en secondes.
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
            RetrievalResponse validée par Pydantic, dont le `query_id` a été
            vérifié comme identique à celui de la requête.

        Raises:
            ActionUnavailableError: Module injoignable ou délai dépassé.
            ActionHTTPError: Réponse avec un code HTTP non-2xx.
            ActionProtocolError: Corps non conforme au contrat, ou `query_id`
                ne correspondant pas à celui envoyé.
        """
        url = f"{self.base_url}/retrieve"
        logger.debug(
            "ActionClient : POST %s (query_id=%s, hop_index=%d)",
            url,
            request.query_id,
            request.hop_index,
        )

        # ── Transport ────────────────────────────────────────────────────────
        try:
            response = httpx.post(
                url,
                json=request.model_dump(),
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise ActionUnavailableError(
                f"module ACTION : délai de {self.timeout:g}s dépassé sur {url}"
            ) from exc
        except httpx.RequestError as exc:
            raise ActionUnavailableError(
                f"module ACTION injoignable sur {url} ({type(exc).__name__}: {exc})"
            ) from exc

        # ── Statut HTTP : tout non-2xx est une erreur franche ────────────────
        if not response.is_success:
            raise ActionHTTPError(response.status_code, response.text)

        # ── Conformité du corps au contrat ───────────────────────────────────
        try:
            payload = response.json()
        except ValueError as exc:
            raise ActionProtocolError(
                f"réponse du module ACTION non décodable en JSON : {exc}"
            ) from exc

        try:
            retrieval = RetrievalResponse.model_validate(payload)
        except ValidationError as exc:
            raise ActionProtocolError(
                f"réponse non conforme au contrat RetrievalResponse : {exc}"
            ) from exc

        # ── Corrélation requête/réponse ──────────────────────────────────────
        # Contrôle indispensable dès que plusieurs retrievals partent en
        # parallèle (multi-hop) : une réponse mal corrélée injecterait le
        # contexte d'une sous-question dans une autre, produisant une réponse
        # finale plausible mais fondée sur les mauvaises sources — un défaut
        # bien plus coûteux qu'une erreur franche.
        if retrieval.query_id != request.query_id:
            raise ActionProtocolError(
                "corrélation rompue : query_id envoyé "
                f"{request.query_id!r}, reçu {retrieval.query_id!r}"
            )

        return retrieval


__all__ = [
    "ActionClient",
    "ActionClientError",
    "ActionHTTPError",
    "ActionProtocolError",
    "ActionUnavailableError",
    "RetrievalClient",
]
