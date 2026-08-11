# RAG-REASON × astraexec — Document de Handoff Technique
## Intégration Module REASONING ↔ Module ACTION

| | |
|---|---|
| **Document** | `docs/ACTION_INTEGRATION_HANDOFF.md` |
| **Version** | 1.0 |
| **Date** | Août 2026 |
| **Auteur** | Module REASONING (Mohammed Solimani) |
| **Destinataire** | Module ACTION (Ihssane — repo `astraexec`) |
| **Statut** | À valider conjointement avant implémentation du Sprint 6 |

---

## 1. Contexte du projet

RAG-REASON est un module de raisonnement avancé pour agent RAG, développé en binôme avec le module ACTION (`astraexec`). L'architecture est à **deux modules séparés** : le module REASONING (ce dépôt) orchestre la logique de raisonnement (classification, planification, critique, vérification) via un graphe LangGraph, tandis que le module ACTION (repo `astraexec`) est responsable de l'accès réel à la base vectorielle et de l'exécution des recherches. La **séparation par contrat** a été définie au Sprint 1 et constitue le fondement du projet : aucun appel direct de fonction entre les modules — tout passe par des objets Pydantic sérialisés via HTTP.

L'objectif de ce document est de fournir à Ihssane et à son assistant le contexte minimal pour comprendre ce qui est attendu côté ACTION, et d'aligner l'implémentation avant le Sprint 6 (Orchestration LangGraph), lors duquel les deux modules seront branchés ensemble pour la première fois.

---

## 2. Ce qui est implémenté côté REASONING à ce jour

Les sprints 0 à 4 sont terminés et validés (CI verte : Ruff, Mypy strict, Pytest). Voici les composants actifs :

| Composant | Rôle en une phrase | Signature publique exacte |
|---|---|---|
| **QueryAnalyzer** | Classifie la requête utilisateur selon la taxonomie `QueryType` (SIMPLE / MULTI_HOP / COMPARATIVE / AMBIGUOUS) et alloue un `reasoning_budget`. | `analyze(self, query: str) -> AnalysisResult` |
| **Planner** | Décompose une requête complexe en un graphe acyclique dirigé (DAG) de sous-questions atomiques (`ExecutionPlan`). | `decompose(self, query: str, analysis: AnalysisResult) -> ExecutionPlan` |
| **Critic** | Évalue si le contexte récupéré par le module ACTION est suffisant pour répondre à une sous-question donnée, et produit un feedback actionnable. | `evaluate(self, step: PlanStep, response: RetrievalResponse) -> CriticEvaluation` |

**Composants futurs** (non implémentés à ce jour) : Verifier (Sprint 5), Orchestration LangGraph (Sprint 6).

---

## 3. Contrat d'interface actuel (REASONING → ACTION)

> **Ceci est le contrat tel qu'il existe aujourd'hui dans le code — pas une proposition. Il est verrouillé dans `src/reasoning/contracts/action_interface.py` avec la mention `# CONTRACT v1.0 — NE PAS MODIFIER SANS REVUE`, testé en CI depuis le Sprint 1, et référencé directement dans `Critic.evaluate()` et les tests unitaires.**

### Fichier source : `src/reasoning/contracts/action_interface.py`

```python
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
```

### Flux de données résumé

```
REASONING                         ACTION
─────────────────────────────────────────────────────
PlanStep (step_id, sub_query)
    │
    ▼
RetrievalRequest ──── POST /retrieve ────► Executor / ToolRegistry
    (query_id, sub_query,                     (fusion_search ou équivalent)
     hop_index, top_k, ...)
                                         │
RetrievalResponse ◄──── 200 OK ──────────┘
    (query_id, chunks=[RetrievedChunk...],
     retrieval_score, metadata)
    │
    ▼
Critic.evaluate(step, response) → CriticEvaluation
```

---

## 4. Format TOON (si pertinent pour l'échange)

TOON (Token-Oriented Object Notation) est le format utilisé en **interne** au module REASONING pour les communications entre les composants LLM (Analyzer, Planner, Critic) et la couche Python. Sa syntaxe est définie dans `src/reasoning/shared/toon_utils.py` :

```
Syntaxe TOON v1.0 :
    - Délimiteurs de bloc   : <<< (ouverture) et >>> (fermeture)
    - Séparateur clé/valeur : ::
    - Séparateur de liste   : |
    - Valeur None           : chaîne vide après ::
    - Une paire clé/valeur par ligne

Exemple de bloc valide :
    <<<
    query_type :: MULTI_HOP
    confidence :: 0.92
    detected_entities :: LangChain | LangSmith
    reasoning_budget :: 3
    >>>
```

> **TOON n'est PAS le format attendu sur l'interface HTTP REASONING ↔ ACTION.** C'est un format de communication interne LLM↔REASONING, utilisé uniquement dans les prompts des modèles Ollama locaux. L'interface HTTP entre les deux modules utilise exclusivement du **JSON standard sérialisé depuis/vers les modèles Pydantic** (`RetrievalRequest`, `RetrievalResponse`) décrits en section 3. Ihssane n'a pas besoin d'implémenter TOON côté `astraexec`.

---

## 5. État du transport de communication

**État constaté lors de l'analyse du repo REASONING à ce jour :**

- **Côté REASONING** : il n'existe **pas encore** de client HTTP (`action_client.py` ou équivalent) dans `src/`. La couche de transport entre les deux modules n'est pas implémentée — c'est précisément l'objet du Sprint 6 (Orchestration LangGraph).
- **Côté dépendances** : `httpx` n'est **pas présent** dans `pyproject.toml`. Le choix du client HTTP n'est pas encore acté.
- **Format de transport** : le contrat Pydantic (section 3) est verrouillé et sera utilisé quel que soit le transport choisi — il sera sérialisé en JSON sur le fil.

> **Point à valider ensemble avant Sprint 6** : le protocole exact (REST HTTP, appel en process, etc.) et la bibliothèque client côté REASONING. Ce choix est commun aux deux modules.

---

## 6. Constat sur le module ACTION actuel (à vérifier par Ihssane)

> *Ce constat date d'une analyse ponctuelle du repo `astraexec` — à confirmer par Ihssane si son code a évolué depuis.*

---

- Interface actuelle : `ActionInterface` (dataclass, `app/schemas/action_interface.py`)
  avec `tool: str`, `parameters: dict`, `priority`, `confidence`, `request_id`,
  `metadata` — une interface générique d'invocation d'outil, différente en
  forme du contrat `RetrievalRequest` typé et spécifique-retrieval ci-dessus.
- Transport : serveur FastAPI déjà fonctionnel (`app/api/main.py`), endpoints
  `/execute`, `/search`, `/tools`, `/health`.
- Réponse actuelle de `/execute` : `{status, tool, execution_time, result,
  message}`, où pour `fusion_search` le `result` a la forme
  `{results: [{chunk: {chunk_id, content, source}, final_score, semantic,
  lexical}], profile: {...}}` — imbriquée, différente de la forme plate de
  `RetrievedChunk` ci-dessus.
- Écart fonctionnel identifié : le `request_id` fourni en entrée n'est jamais
  réémis dans la réponse de `Executor.run()` — nécessaire pour corréler les
  réponses lors d'appels multiples/parallèles (multi-hop).

---

## 7. Changements demandés côté ACTION

Voici les adaptations concrètes à envisager côté `astraexec`. La **forme exacte des champs** (noms, types, contraintes) des objets Pydantic en section 3 n'est pas négociable côté REASONING — ce contrat est testé en CI depuis Sprint 1. **La manière dont tu l'implèmes en interne t'appartient entièrement** (tu peux utiliser ton `Executor` et ton `ToolRegistry` existants comme couche interne, et exposer simplement un adaptateur en façade).

### 7.1 — Ajouter un endpoint dédié `POST /retrieve`

Créer un nouvel endpoint FastAPI qui :

- **Accepte** : un corps JSON correspondant à `RetrievalRequest` (section 3)
- **Retourne** : un corps JSON correspondant à `RetrievalResponse` (section 3)
- **Valide automatiquement** avec `response_model=RetrievalResponse`

Exemple de signature FastAPI minimale :

```python
@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    # Appel interne à ton Executor / fusion_search existant
    ...
```

L'endpoint `/execute` existant n'a pas besoin d'être modifié.

### 7.2 — Réémettre `query_id` dans la réponse

Le champ `query_id` de `RetrievalRequest` **doit être réémis tel quel** dans `RetrievalResponse.query_id`. C'est indispensable pour que le module REASONING puisse corréler les réponses lors d'appels parallèles (requêtes multi-hop).

### 7.3 — Aplatir la structure de `chunks`

La réponse actuelle de `fusion_search` retourne des chunks dans une structure imbriquée :

```json
{"results": [{"chunk": {"chunk_id": "...", "content": "...", "source": "..."}, "final_score": 0.87}]}
```

Le contrat attendu côté REASONING est une liste plate de `RetrievedChunk` dans `RetrievalResponse.chunks` :

```json
{
  "query_id": "...",
  "chunks": [
    {"chunk_id": "...", "content": "...", "source": "...", "relevance_score": 0.87}
  ],
  "retrieval_score": null
}
```

L'adaptateur côté `astraexec` doit mapper `final_score` → `relevance_score` et aplatir la structure.

### 7.4 — Figer les versions de dépendances

Pour garantir la reproductibilité de l'intégration, figer les versions dans `requirements.txt` (ou équivalent `uv.lock`) avant la session de test croisé Sprint 6.

---

## 8. Prochaines étapes

> Avant de commencer l'implémentation de `/retrieve`, il serait préférable qu'on fasse une **session de synchronisation courte** (30 min) pour valider ensemble : le protocole de transport exact, la gestion des erreurs HTTP côté REASONING, et l'ordre des tâches Sprint 6 — plutôt que de travailler en isolation et de découvrir des écarts lors du branchement.
