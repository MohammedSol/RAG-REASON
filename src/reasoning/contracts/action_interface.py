# CONTRACT v1.0 — NE PAS MODIFIER SANS REVUE
"""
Contrats d'interface JSON entre le module REASONING et le module ACTION.

Ces schémas Pydantic définissent le protocole de communication exclusif
entre les deux modules. Toute modification doit faire l'objet d'une
revue conjointe avec le développeur du module ACTION.

Flux :
    REASONING  →  RetrievalRequest  →  MODULE ACTION
    REASONING  ←  RetrievalResponse ←  MODULE ACTION
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Un chunk de texte retourné par le module ACTION après retrieval.

    Attributes:
        chunk_id: Identifiant unique du chunk dans la base vectorielle.
        content: Contenu textuel brut du chunk.
        source: Origine du chunk (nom de document, URL, identifiant de source).
        relevance_score: Score de pertinence calculé par le module ACTION (0.0 – 1.0).
    """

    chunk_id: str
    content: str
    source: str
    relevance_score: float


class RetrievalRequest(BaseModel):
    """Requête de retrieval envoyée par le module REASONING vers le module ACTION.

    Attributes:
        query_id: Identifiant unique de la requête globale (traçabilité multi-hop).
        sub_query: Sous-requête textuelle à envoyer au retriever.
        hop_index: Indice du saut courant dans le plan d'exécution (0-indexed).
        filters: Filtres optionnels à appliquer sur les métadonnées des chunks.
        top_k: Nombre maximum de chunks à retourner. Doit être strictement positif.
        metadata: Données supplémentaires libres transmises au module ACTION.
    """

    query_id: str
    sub_query: str
    hop_index: int
    filters: dict[str, Any] | None = None
    top_k: int = Field(gt=0, description="Nombre de chunks à retourner (doit être > 0)")
    metadata: dict[str, Any] | None = None


class RetrievalResponse(BaseModel):
    """Réponse du module ACTION suite à une RetrievalRequest.

    Attributes:
        query_id: Identifiant de la requête d'origine (doit correspondre à la Request).
        chunks: Liste des chunks récupérés, ordonnés par pertinence décroissante.
        retrieval_score: Score global de confiance du retrieval (optionnel).
        metadata: Données supplémentaires retournées par le module ACTION.
    """

    query_id: str
    chunks: list[RetrievedChunk]
    retrieval_score: float | None = None
    metadata: dict[str, Any] | None = None
