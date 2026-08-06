# RAG-REASON — Feuille de Route du Module de Raisonnement
> **Projet :** Module REASONING d'un agent RAG avancé
> **Rôle :** Architecte / Tech Lead IA
> **Langage :** Python 3.11+ · **Orchestration :** LangGraph · **LLMs :** LiteLLM + Ollama
> **Validation :** Pydantic · **Évaluation :** RAGAS · **Qualité :** pytest + Ruff
> **Gestionnaire de paquets :** uv

---

## Table des Matières

1. [Vision & Périmètre](#1-vision--périmètre)
2. [Architecture Globale](#2-architecture-globale)
3. [Sprint 0 — Initialisation & Infrastructure](#sprint-0--initialisation--infrastructure)
4. [Sprint 1 — Contrats d'Interface JSON (Pydantic)](#sprint-1--contrats-dinterface-json-pydantic)
5. [Sprint 2 — Query Analyzer (Routeur Sémantique)](#sprint-2--query-analyzer-routeur-sémantique)
6. [Sprint 3 — Planner (Plan-and-Solve)](#sprint-3--planner-plan-and-solve)
7. [Sprint 4 — Critic (Self-RAG Feedback Loop)](#sprint-4--critic-self-rag-feedback-loop)
8. [Sprint 5 — Verifier (Groundedness Check)](#sprint-5--verifier-groundedness-check)
9. [Sprint 6 — Orchestration LangGraph](#sprint-6--orchestration-langgraph)
10. [Sprint 7 — Évaluation RAGAS](#sprint-7--évaluation-ragas)
11. [Sprint 8 — Qualité, CI & Documentation Finale](#sprint-8--qualité-ci--documentation-finale)
12. [Tableau de Bord des Dépendances Inter-Sprints](#tableau-de-bord-des-dépendances-inter-sprints)

---

## 1. Vision & Périmètre

### Problème adressé
Le RAG linéaire classique échoue sur les requêtes complexes nécessitant plusieurs sauts de raisonnement (*multi-hop*), la résolution d'ambiguïtés ou la vérification de cohérence. Le module REASONING est le "cerveau" de l'agent : il décide, planifie, critique et valide, sans jamais accéder directement à la base vectorielle.

### Frontières du module REASONING (ce dépôt)
| Responsabilité | Inclus | Exclu |
|---|---|---|
| Analyse de requête | ✅ Query Analyzer | ❌ Embedding / Retrieval |
| Planification | ✅ Planner | ❌ Exécution des recherches |
| Évaluation du contexte | ✅ Critic | ❌ Chunking / Indexation |
| Vérification de réponse | ✅ Verifier | ❌ Ingestion de données |
| Orchestration du flux | ✅ LangGraph Graph | ❌ Interface utilisateur (hors microservice démo) |

> **Nota bene (ajout Sprint 3)** : La contrainte "sans interface utilisateur" est assouplie pour un microservice de démonstration Streamlit (`frontend/app.py`) exploitant Graphviz. Ce microservice opère en **lecture seule** sur le backend (import dynamique de `src/` via `sys.path`, aucune écriture vers le module REASONING) et n'est pas inclus dans le périmètre de test CI.

### Contrat d'interface avec le module ACTION
La communication est **exclusivement via le format TOON** (`<<<...>>>`), validée par des schémas Pydantic partagés. Le module ACTION reçoit un `RetrievalRequest` et retourne un `RetrievalResponse`. Aucun appel direct de fonction entre modules. Le parsing/sérialisation TOON est centralisé dans `src/reasoning/shared/toon_utils.py` (point unique de vérité — cf. Sprint 3).

> **Statut refonte contrat (planifié, non implémenté)** : Une migration `RetrievalRequest` → `PlanExecutionRequest` (pour déléguer la parallélisation au module ACTION) a été discutée mais n'existe pas dans le code à ce jour. `RetrievalRequest` v1.0 reste le schéma actif dans `contracts/action_interface.py`.

---

## 2. Architecture Globale

```
┌─────────────────────────────────────────────────────────┐
│                   MODULE REASONING                      │
│                                                         │
│  User Query                                             │
│      │                                                  │
│      ▼                                                  │
│  ┌───────────────┐                                      │
│  │ Query Analyzer│ ── classifie: simple/multi-hop/ambigu│
│  └───────┬───────┘                                      │
│          │                                              │
│          ▼                                              │
│  ┌───────────────┐                                      │
│  │    Planner    │ ── génère un graphe de sous-requêtes  │
│  └───────┬───────┘                                      │
│          │  RetrievalRequest (TOON/Pydantic)            │
│          ▼                                              │
│  ════════════════════════════════════════               │
│        INTERFACE MODULE ACTION (TOON)                   │
│  ════════════════════════════════════════               │
│          │  RetrievalResponse (TOON/Pydantic)           │
│          ▼                                              │
│  ┌───────────────┐                                      │
│  │    Critic     │ ── évalue la pertinence du contexte  │
│  └───────┬───────┘                                      │
│          │ (boucle de rétroaction si insuffisant)       │
│          ▼                                              │
│  ┌───────────────┐                                      │
│  │   Verifier    │ ── vérifie la fidélité (Groundedness)│
│  └───────┬───────┘                                      │
│          │                                              │
│          ▼                                              │
│   Final Answer + Metadata (Faithfulness Score)          │
└─────────────────────────────────────────────────────────┘
```

---

## Sprint 0 — Initialisation & Infrastructure

> **Objectif :** Environnement de développement reproductible, structure de projet canonique, outillage de qualité.
> **Durée estimée :** 0.5 jour
> **Livrable :** Dépôt initialisé, environnement `uv` fonctionnel, pre-commit actif.

### 0.1 — Initialisation du dépôt
- [x] Créer le dépôt Git local et distant (GitHub/GitLab)
- [x] Rédiger un `.gitignore` adapté (Python, uv, Ollama, secrets)
- [x] Initialiser le projet avec `uv init` et configurer `pyproject.toml`
- [x] Définir la version Python minimale à `3.11` dans `pyproject.toml`

### 0.2 — Structure de répertoires
- [x] Créer l'arborescence canonique du projet :
  ```
  RAG-REASON/
  ├── src/
  │   └── reasoning/
  │       ├── __init__.py
  │       ├── analyzer/        # Query Analyzer
  │       ├── planner/         # Planner
  │       ├── critic/          # Critic
  │       ├── verifier/        # Verifier
  │       ├── graph/           # Orchestration LangGraph
  │       ├── shared/          # Utilitaires partagés (toon_utils.py)
  │       └── contracts/       # Schémas Pydantic partagés
  ├── tests/
  │   ├── unit/
  │   ├── integration/
  │   └── evaluation/          # Scripts RAGAS
  ├── configs/                 # Fichiers de configuration YAML/TOML
  ├── docs/                    # Documentation technique
  ├── frontend/                # Microservice démo Streamlit (lecture seule)
  ├── scripts/                 # Utilitaires (setup Ollama, évaluation HotpotQA, etc.)
  ├── data/                    # Datasets et résultats d'évaluation
  ├── plan.md                  # Ce fichier
  ├── pyproject.toml
  ├── README.md
  └── .env.example
  ```

### 0.3 — Gestion des dépendances avec `uv`
- [x] Ajouter les dépendances principales :
  - `langgraph`, `langchain-core`
  - `litellm`
  - `pydantic>=2.0`
  - `python-dotenv`
  - `pyyaml`
- [x] Ajouter les dépendances d'évaluation :
  - `ragas`
  - `datasets` (HuggingFace)
- [x] Ajouter les dépendances de développement (`dev`) :
  - `pytest`, `pytest-asyncio`, `pytest-cov`
  - `ruff`
  - `mypy`
  - `pre-commit`
- [x] Ajouter les dépendances frontend : `streamlit`, `graphviz`
- [x] Vérifier que `uv lock` génère un `uv.lock` reproductible

### 0.4 — Outillage Qualité
- [x] Configurer `ruff` dans `pyproject.toml` (règles : `E`, `W`, `F`, `I`, `UP`, `B`)
- [x] Configurer `mypy` en mode strict (`strict = true`)
- [x] Créer `.pre-commit-config.yaml` avec les hooks : `ruff`, `mypy`, `trailing-whitespace`
- [x] Installer les hooks : `pre-commit install`
- [x] Valider que `pre-commit run --all-files` s'exécute sans erreur (statut : Skipped — aucun fichier Python suivi par Git, comportement attendu sur base vide)

### 0.5 — Configuration Ollama & LiteLLM
- [x] Vérifier que le service Ollama est opérationnel en local
- [x] Puller le modèle de raisonnement : `ollama pull qwen2.5:7b` (ou `llama3.1:8b`)
- [x] Créer le fichier `.env.example` avec les variables nécessaires :
  ```
  OLLAMA_BASE_URL=http://localhost:11434
  DEFAULT_REASONING_MODEL=ollama/qwen2.5:7b
  DEFAULT_FAST_MODEL=ollama/qwen2.5:3b
  LOG_LEVEL=INFO
  ```
- [x] Tester la connectivité LiteLLM → Ollama avec un script de smoke-test

---

## Sprint 1 — Contrats d'Interface JSON (Pydantic)

> **Objectif :** Définir et verrouiller tous les schémas de données partagés entre les composants internes et avec le module ACTION. C'est le fondement contractuel du projet.
> **Durée estimée :** 1 jour
> **Livrable :** Module `contracts/` complet et testé. Aucun composant ne peut être développé sans ce sprint.

### 1.1 — Contrats d'interface avec le module ACTION
- [x] Définir `RetrievalRequest` (envoyé vers le module ACTION) :
  - Champs : `query_id`, `sub_query`, `hop_index`, `filters`, `top_k`, `metadata`
- [x] Définir `RetrievalResponse` (reçu depuis le module ACTION) :
  - Champs : `query_id`, `chunks` (liste de `RetrievedChunk`), `retrieval_score`, `metadata`
- [x] Définir `RetrievedChunk` :
  - Champs : `chunk_id`, `content`, `source`, `relevance_score`
- [ ] Valider ces schémas avec le développeur du module ACTION (revue conjointe)
- [x] **Geler** ces schémas dans `contracts/action_interface.py` avec un commentaire `# CONTRACT v1.0 — NE PAS MODIFIER SANS REVUE`

### 1.2 — Contrats internes du module REASONING
- [x] Définir `QueryType` (Enum) : `SIMPLE`, `MULTI_HOP`, `AMBIGUOUS`, `COMPARATIVE`
- [x] Définir `AnalysisResult` (sortie du Query Analyzer) :
  - Champs : `query_type`, `confidence`, `detected_entities`, `reasoning_budget`
- [x] Définir `ExecutionPlan` (sortie du Planner) :
  - Champs : `plan_id`, `original_query`, `steps` (liste de `PlanStep`), `dependencies_graph`
- [x] Définir `PlanStep` :
  - Champs : `step_id`, `sub_query`, `depends_on` (liste d'IDs), `status`
- [x] Définir `CriticEvaluation` (sortie du Critic) :
  - Champs : `step_id`, `is_sufficient`, `relevance_score`, `missing_aspects`, `feedback`
- [x] Définir `VerificationResult` (sortie du Verifier) :
  - Champs : `is_grounded`, `faithfulness_score`, `unsupported_claims`, `final_answer`
- [x] Définir `AgentState` (état global du graphe LangGraph) :
  - Union de tous les champs nécessaires à la circulation d'information dans le graphe

### 1.3 — Tests des contrats
- [x] Écrire des tests unitaires `tests/unit/test_contracts.py` couvrant :
  - Sérialisation/désérialisation JSON (`model_dump()`, `model_validate()`)
  - Validation des types stricts (ex: `top_k` doit être `int > 0`)
  - Comportement face aux données invalides (assertions `pytest.raises(ValidationError)`)
- [x] Atteindre une couverture de 100% sur le module `contracts/` — **47/47 tests passés, 100% de couverture sur `contracts/`**
- [x] Vérifier que `mypy` passe sans erreur sur `contracts/` — **"Success: no issues found"** (`mypy src/ tests/ scripts/ --no-error-summary`)

---

## Sprint 2 — Query Analyzer (Routeur Sémantique)

> **Objectif :** Implémenter le composant de classification des requêtes pour rendre le raisonnement adaptatif.
> **Durée estimée :** 1.5 jours
> **Prérequis :** Sprint 1 terminé (`AnalysisResult`, `QueryType` définis)
> **Livrable :** Composant `analyzer/` fonctionnel, testé.

### 2.1 — Design du Query Analyzer
- [x] Rédiger la spécification fonctionnelle dans `docs/analyzer_spec.md` (statut : Validé v1.0) :
  - Règles de classification (patterns multi-hop : "et ensuite", "qui a", "compare")
  - Définition du `reasoning_budget` par type de requête
  - Stratégie de prompt engineering pour la classification
- [x] Choisir la stratégie d'implémentation : **architecture hybride** (LLM-based Niveau 1 + fallback heuristique Python Niveau 2 avec pré-classificateur Niveau 0 pour les cas COMPARATIVE triviaux)

### 2.2 — Implémentation
- [x] Créer `src/reasoning/analyzer/analyzer.py` avec la classe `QueryAnalyzer`
- [x] Implémenter la méthode principale `analyze(query: str) -> AnalysisResult`
- [x] Créer le template de prompt de classification dans `analyzer/prompts.py` (few-shot anglophone, format TOON strict)
- [x] Implémenter la logique de parsing de la réponse LLM via `shared/toon_utils.py` (remplacement du parsing JSON ad-hoc)
- [x] Implémenter le fallback heuristique de secours (marqueurs bilingues FR/EN)
- [x] Créer `src/reasoning/analyzer/__init__.py` exposant proprement l'API publique

### 2.3 — Tests
- [x] Écrire des tests unitaires `tests/unit/test_analyzer.py` avec des mocks LiteLLM :
  - Requête simple → `QueryType.SIMPLE`
  - Requête multi-sauts → `QueryType.MULTI_HOP`
  - Requête ambiguë → `QueryType.AMBIGUOUS`
  - Test du fallback heuristique
- [x] Écrire un test d'intégration `tests/integration/test_analyzer_live.py` avec Ollama réel
- [x] Vérifier la couverture de code : **91% sur `reasoning.analyzer`** (136/137 tests unitaires passés)

> ⚠️ **Écart constaté :** `test_analyzer_default_params` FAIL — le test attend `timeout=15.0` mais `analyzer.py` utilise `timeout=20.0` (ligne 194). La valeur 15.0 correspondait à une optimisation intermédiaire (Solution A du diagnostic) qui a été ajustée à 20.0 dans le code final sans mise à jour du test. **Cette case est cochée car l'implémentation est fonctionnelle, mais le test doit être synchronisé avec la valeur réelle.**

---

## Sprint 3 — Planner (Plan-and-Solve)

> **Objectif :** Décomposer une requête complexe en un plan d'exécution structuré avec gestion des dépendances. Inclut la mise en place de l'utilitaire TOON partagé et d'un microservice de démonstration Streamlit.
> **Durée estimée :** 2 jours
> **Prérequis :** Sprint 1 (`ExecutionPlan`, `PlanStep`), Sprint 2 (`AnalysisResult` comme entrée)
> **Livrable :** Composant `planner/` fonctionnel, graphe de dépendances correct, utilitaire TOON centralisé, dashboard Streamlit opérationnel.

### 3.0 — Centralisation du protocole TOON (transversal)
- [x] Créer `src/reasoning/shared/toon_utils.py` comme point unique de vérité pour la sérialisation TOON v1.0 :
  - `parse_toon_to_dict(raw: str) -> dict[str, Any]` : parse une sortie LLM TOON en dictionnaire Python
  - `dump_dict_to_toon(data: dict[str, Any]) -> str` : sérialise un dictionnaire vers le format TOON
  - `ToonParseError(ValueError)` : exception dédiée, compatible avec les blocs `except` existants
- [x] Créer `src/reasoning/shared/__init__.py`
- [x] Remplacer le parsing JSON ad-hoc de `analyzer.py` par `parse_toon_to_dict` (import depuis `shared.toon_utils`)
- [x] Écrire les tests unitaires `tests/unit/test_toon_utils.py` — **77/77 tests passés, couverture 100%**
- [ ] Migrer `critic.py` vers `toon_utils` (en attente de l'implémentation du Critic — Sprint 4)
- [ ] Migrer `verifier.py` vers `toon_utils` (en attente de l'implémentation du Verifier — Sprint 5)

### 3.1 — Design du Planner
- [x] Rédiger la spécification dans `docs/planner_spec.md` (statut : Validé v1.0) :
  - Format du graphe de dépendances (liste d'adjacence, DAG)
  - Règle : les étapes sans dépendances peuvent être exécutées en parallèle
  - Budget maximal de décomposition (`max_steps` paramétrable)
- [x] Définir la structure du prompt Plan-and-Solve (Few-shot examples inclus, format TOON strict)

### 3.2 — Implémentation
- [x] Créer `src/reasoning/planner/planner.py` avec la classe `Planner`
- [x] Implémenter `decompose(query: str, analysis: AnalysisResult) -> ExecutionPlan`
- [x] Implémenter la logique de parsing du plan généré par le LLM via `shared/toon_utils.py`
- [x] Implémenter la validation du graphe (détection de cycles via algorithme de Kahn, vérification des dépendances)
- [x] Créer `src/reasoning/planner/prompts.py` avec les templates de décomposition (few-shot anglophone)
- [x] Gérer le cas dégénéré : si `QueryType.SIMPLE`, retourner un plan à 1 seule étape (court-circuit Niveau 0, 0 appel LLM)
- [x] Créer `src/reasoning/planner/__init__.py`

### 3.3 — Tests
- [x] Tests unitaires `tests/unit/test_planner.py` avec mocks :
  - Requête simple → plan à 1 étape, pas de dépendances
  - Requête multi-hop à 3 sauts → plan avec dépendances séquentielles
  - Validation : détection de cycle dans le graphe → erreur levée
  - Validation : dépassement de `max_steps` → troncature ou erreur
- [x] Tests d'intégration `tests/integration/test_planner_integration.py` avec Ollama réel

### 3.4 — Dashboard de démonstration (microservice frontend, lecture seule)
- [x] Créer `frontend/app.py` : microservice Streamlit + Graphviz en **lecture seule** sur `src/`
  - Onglet "Démo Interactive" : saisie libre, appel live à `QueryAnalyzer` + `Planner`, rendu DAG via Graphviz
  - Onglet "Évaluation" : lecture dynamique de `data/evaluation/analyzer_results.json` et `planner_results.json` via `@st.cache_data`
  - Onglet "CI/CD" : exécution via `subprocess.Popen` de Ruff, Mypy et Pytest avec streaming stdout en temps réel
- [x] Ajouter `streamlit` et `graphviz` aux dépendances via `uv add`
- [x] Lancement : `uv run streamlit run frontend/app.py`

> **Note :** `_MAX_TOKENS=1024` dans `planner.py` — la réduction à 400 (optimisation KV-cache) est identifiée mais pas encore appliquée.

---

## Sprint 4 — Critic (Self-RAG Feedback Loop)

> **Objectif :** Évaluer la qualité du contexte récupéré et générer un feedback structuré pour déclencher de nouvelles recherches si nécessaire.
> **Durée estimée :** 2 jours
> **Prérequis :** Sprint 1 (`CriticEvaluation`, `RetrievalResponse`), Sprint 3 (`PlanStep` comme contexte)
> **Livrable :** Composant `critic/` avec logique de décision binaire is_sufficient.

### 4.1 — Design du Critic
- [ ] Rédiger la spécification dans `docs/critic_spec.md` :
  - Critères d'évaluation : pertinence, complétude, cohérence, fraîcheur
  - Seuil de `relevance_score` pour `is_sufficient = True` (paramétrable)
  - Stratégie du `feedback` : structuré pour guider la re-requête du Planner
  - Limite maximale de boucles de rétroaction (`max_retries`, anti-boucle infinie)
- [ ] Définir la structure du prompt d'évaluation (Chain-of-Thought recommandé)

### 4.2 — Implémentation
- [ ] Créer `src/reasoning/critic/critic.py` avec la classe `Critic`
- [ ] Implémenter `evaluate(step: PlanStep, response: RetrievalResponse) -> CriticEvaluation`
- [ ] Implémenter la logique de scoring et le parsing TOON via `shared/toon_utils.py`
- [ ] Implémenter le mécanisme de génération du `feedback` (aspects manquants identifiés)
- [ ] Créer `src/reasoning/critic/prompts.py`
- [ ] Implémenter la garde anti-boucle (`max_retries` counter)
- [ ] Créer `src/reasoning/critic/__init__.py`

### 4.3 — Tests
- [ ] Tests unitaires `tests/unit/test_critic.py` :
  - Contexte pertinent → `is_sufficient = True`, `relevance_score >= seuil`
  - Contexte hors-sujet → `is_sufficient = False`, `feedback` non vide
  - Contexte partiel → `is_sufficient = False`, `missing_aspects` correctement remplis
  - Test de la garde anti-boucle : après `max_retries` itérations, forcer la sortie
- [ ] Tests d'intégration avec une paire (PlanStep, RetrievalResponse) réelle mockée

---

## Sprint 5 — Verifier (Groundedness Check)

> **Objectif :** Valider la fidélité de la réponse finale par rapport aux sources récupérées pour prévenir les hallucinations.
> **Durée estimée :** 1.5 jours
> **Prérequis :** Sprint 1 (`VerificationResult`, `RetrievedChunk`)
> **Livrable :** Composant `verifier/` capable d'identifier les affirmations non-fondées.

### 5.1 — Design du Verifier
- [ ] Rédiger la spécification dans `docs/verifier_spec.md` :
  - Définition de "Groundedness" : chaque affirmation de la réponse doit être traçable à un chunk source
  - Stratégie : décomposition de la réponse en claims atomiques, vérification claim par claim
  - Décision finale : `is_grounded` = True si `faithfulness_score >= 0.8` (configurable)
  - Comportement si non-fondé : tronquer, signaler ou demander une reformulation

### 5.2 — Implémentation
- [ ] Créer `src/reasoning/verifier/verifier.py` avec la classe `Verifier`
- [ ] Implémenter `verify(answer: str, sources: list[RetrievedChunk]) -> VerificationResult`
- [ ] Implémenter la décomposition de la réponse en claims atomiques
- [ ] Implémenter la vérification de chaque claim contre les sources via `shared/toon_utils.py`
- [ ] Calculer le `faithfulness_score` : `claims_supported / total_claims`
- [ ] Créer `src/reasoning/verifier/prompts.py`
- [ ] Créer `src/reasoning/verifier/__init__.py`

### 5.3 — Tests
- [ ] Tests unitaires `tests/unit/test_verifier.py` :
  - Réponse 100% fondée → `is_grounded = True`, `faithfulness_score = 1.0`
  - Réponse avec hallucination → `is_grounded = False`, `unsupported_claims` non vide
  - Réponse partiellement fondée → score intermédiaire, décision selon seuil
  - Sources vides → comportement défensif défini (erreur levée ou score = 0)

---

## Sprint 6 — Orchestration LangGraph

> **Objectif :** Assembler tous les composants dans un graphe d'état LangGraph cohérent, gérant les flux conditionnels et les boucles de rétroaction.
> **Durée estimée :** 2.5 jours
> **Prérequis :** Sprints 2, 3, 4, 5 terminés. `AgentState` défini (Sprint 1).
> **Livrable :** Graphe LangGraph complet, exécutable de bout en bout.

### 6.1 — Design du Graphe
- [ ] Rédiger la spécification dans `docs/graph_spec.md`
- [ ] Définir les nœuds : `analyze_query`, `plan`, `retrieve`, `critique`, `generate_answer`, `verify`, `END`
- [ ] Définir les arêtes conditionnelles :
  - `analyze` → `plan` (si MULTI_HOP) ou `retrieve` directement (si SIMPLE)
  - `critique` → `retrieve` (si `is_sufficient = False`) ou `generate_answer` (si True)
  - `verify` → `END` (si `is_grounded = True`) ou `plan` (si False, re-planification)
- [ ] Représenter le graphe sous forme de diagramme Mermaid dans `docs/graph.md`

### 6.2 — Implémentation du Graphe
- [ ] Créer `src/reasoning/graph/state.py` définissant `AgentState` avec `TypedDict`
- [ ] Créer `src/reasoning/graph/nodes.py` : fonctions nœuds (`async def analyze_node(state)`, etc.)
- [ ] Créer `src/reasoning/graph/edges.py` : fonctions de routage conditionnel
- [ ] Créer `src/reasoning/graph/graph.py` : assemblage du `StateGraph` LangGraph
  - Ajouter tous les nœuds
  - Définir les arêtes et arêtes conditionnelles
  - Compiler le graphe (`graph.compile()`)
- [ ] Implémenter le nœud `retrieve` comme client de l'interface ACTION (appel HTTP/JSON mockable)
- [ ] Implémenter le compteur de rétroaction dans `AgentState` (protection anti-boucle globale)
- [ ] Créer `src/reasoning/graph/__init__.py`
- [ ] Créer le point d'entrée `src/reasoning/agent.py` exposant `run_agent(query: str) -> VerificationResult`

### 6.3 — Tests d'Intégration du Graphe
- [ ] Tests `tests/integration/test_graph.py` :
  - Parcours complet SIMPLE : `analyze` → `retrieve` → `critique` → `generate` → `verify` → END
  - Parcours complet MULTI_HOP : plan en 2 étapes, critique échoue au 1er tour, succès au 2e
  - Test de la boucle de re-planification (verify échoue → retour au plan)
  - Test de la garde globale anti-boucle infinie
- [ ] Visualiser le graphe compilé avec `graph.get_graph().draw_mermaid()` et sauvegarder dans `docs/`

---

## Sprint 7 — Évaluation RAGAS

> **Objectif :** Mesurer objectivement la qualité du système sur un dataset de référence avec les métriques RAGAS.
> **Durée estimée :** 2 jours
> **Prérequis :** Sprint 6 terminé (agent exécutable de bout en bout)
> **Livrable :** Pipeline d'évaluation automatisé, rapport de métriques de référence (baseline).

### 7.1 — Préparation du Dataset d'Évaluation
- [ ] Constituer un dataset de 30 à 50 questions dans `tests/evaluation/dataset.json` :
  - 15 requêtes simples
  - 20 requêtes multi-hop
  - 10 requêtes comparatives ou ambiguës
- [ ] Chaque entrée doit contenir : `question`, `ground_truth_answer`, `ground_truth_contexts`
- [ ] Valider le format avec un schéma Pydantic dédié `EvaluationSample`

### 7.2 — Pipeline d'Évaluation RAGAS
- [ ] Créer `tests/evaluation/run_evaluation.py` :
  - Charger le dataset
  - Exécuter l'agent RAG-REASON sur chaque question
  - Collecter : `answer`, `contexts` (chunks utilisés), `question`
  - Construire le dataset RAGAS
- [ ] Configurer les métriques RAGAS à mesurer :
  - `faithfulness` → *cible : > 0.85*
  - `answer_relevancy` → *cible : > 0.80*
  - `context_precision`
  - `context_recall`
- [ ] Configurer RAGAS pour utiliser Ollama comme LLM d'évaluation (via LiteLLM)
- [ ] Générer un rapport `tests/evaluation/reports/baseline_report.json`
- [ ] Créer `tests/evaluation/reports/baseline_report.md` avec tableau de résultats lisible

### 7.3 — Analyse et Seuils de Qualité
- [ ] Définir les seuils d'acceptabilité dans `configs/evaluation_thresholds.toml`
- [ ] Implémenter un script d'assertion `pytest tests/evaluation/test_metrics_thresholds.py`
  - Le test FAIL si une métrique descend sous son seuil minimum
- [ ] Documenter les résultats du baseline dans `docs/evaluation_results.md`
- [ ] Identifier les cas d'échec (failing examples) et créer des tickets d'amélioration

---

## Sprint 8 — Qualité, CI & Documentation Finale

> **Objectif :** Consolider la qualité du code, automatiser la CI, et produire une documentation technique exploitable.
> **Durée estimée :** 1.5 jours
> **Prérequis :** Tous les sprints précédents terminés.
> **Livrable :** Pipeline CI vert, documentation complète, projet prêt pour la revue.

### 8.1 — Couverture de Tests & Qualité Code
- [ ] Atteindre une couverture globale >= 80% (`pytest --cov=reasoning --cov-report=html`) — **état actuel : 93% sur les composants implémentés (S0-S3)**
- [ ] Vérifier que `ruff check .` passe sans warning — **état actuel : 8 erreurs I001 résiduelles dans scripts/ et tests/integration/ (imports non triés)**
- [ ] Vérifier que `mypy src/` passe sans erreur en mode strict — **état actuel : ✅ zéro erreur**
- [ ] Supprimer tout code mort, imports inutilisés, TODOs non résolus
- [ ] Synchroniser `test_analyzer_default_params` avec la valeur réelle de `timeout` dans `analyzer.py`

### 8.2 — Pipeline CI (GitHub Actions / GitLab CI)
- [ ] Créer `.github/workflows/ci.yml` avec les étapes :
  1. `uv sync` — installation des dépendances
  2. `ruff check .` — linting
  3. `mypy src/` — typage
  4. `pytest tests/unit/ tests/integration/ --cov` — tests + couverture
  5. Publication du rapport de couverture en artifact
- [ ] Configurer le pipeline pour se déclencher sur chaque Pull Request vers `main`
- [ ] Ajouter un badge de statut CI dans `README.md`

### 8.3 — Documentation Technique
- [ ] Compléter `README.md` :
  - Description du projet, architecture, prérequis
  - Guide d'installation rapide (`uv sync`, configuration `.env`)
  - Guide d'utilisation (lancer l'agent, lancer le dashboard `uv run streamlit run frontend/app.py`, exécuter l'évaluation)
- [ ] Créer `docs/architecture.md` : description détaillée de chaque composant
- [ ] Créer `docs/interface_contract_v1.md` : documentation formelle des schémas JSON/TOON partagés avec le module ACTION
- [ ] Créer `docs/evaluation_results.md` : résultats RAGAS commentés
- [ ] Générer la documentation de l'API avec `pdoc` ou `mkdocs`

### 8.4 — Revue Finale & Préparation au Hand-off
- [ ] Session de revue de code avec le développeur du module ACTION
- [ ] Vérifier la compatibilité des contrats d'interface dans une session d'intégration end-to-end
- [ ] Créer un tag Git `v1.0.0-reasoning` et rédiger les release notes
- [ ] Archiver les rapports d'évaluation RAGAS dans `docs/`

---

## Tableau de Bord des Dépendances Inter-Sprints

```
Sprint 0 (Infrastructure)
    │
    ▼
Sprint 1 (Contrats Pydantic) ◄──── FONDATION CRITIQUE
    │
    ├──► Sprint 2 (Query Analyzer) ✅
    │         │
    ├──► Sprint 3 (Planner) ✅ ◄─────────────────────┐
    │         │                                       │
    ├──► Sprint 4 (Critic)                            │
    │         │                                       │
    └──► Sprint 5 (Verifier)                          │
              │                                       │
              ▼                                       │
         Sprint 6 (LangGraph) ──────────────────────┘
              │
              ▼
         Sprint 7 (RAGAS Evaluation)
              │
              ▼
         Sprint 8 (CI & Docs)
```

| Sprint | Prérequis | Durée estimée | Risque | Statut |
|--------|-----------|---------------|--------|--------|
| Sprint 0 | — | 0.5 jour | Faible | ✅ Terminé |
| Sprint 1 | S0 | 1 jour | CRITIQUE (fondation) | ✅ Terminé |
| Sprint 2 | S1 | 1.5 jours | Moyen (prompt eng.) | ✅ Terminé (1 test à sync) |
| Sprint 3 | S1, S2 | 2 jours | Moyen (parsing TOON) | ✅ Terminé |
| Sprint 4 | S1, S3 | 2 jours | Moyen (logique boucle) | 🔲 À faire |
| Sprint 5 | S1 | 1.5 jours | Faible | 🔲 À faire |
| Sprint 6 | S2, S3, S4, S5 | 2.5 jours | CRITIQUE (intégration) | 🔲 À faire |
| Sprint 7 | S6 | 2 jours | Moyen (dataset) | 🔲 À faire |
| Sprint 8 | S7 | 1.5 jours | Faible | 🔲 À faire |
| **TOTAL** | | **~14.5 jours** | | |

---

## Règles de Travail & Conventions

### Conventions de Nommage
- **Modules** : `snake_case`
- **Classes** : `PascalCase`
- **Fonctions/méthodes** : `snake_case`
- **Constantes** : `UPPER_SNAKE_CASE`
- **Schémas Pydantic** : `PascalCase` + suffixe sémantique (ex: `RetrievalRequest`, `AnalysisResult`)

### Politique des Branches Git
- `main` : branche protégée, toujours verte (CI obligatoire)
- `develop` : branche d'intégration
- `feature/sprint-N-composant` : branches de fonctionnalités (ex: `feature/sprint2-query-analyzer`)
- **Règle** : aucun `push` direct sur `main`, toutes les modifications passent par PR

### Politique de Commit (Conventional Commits)
```
feat(analyzer): implémente la classification LLM-based
fix(critic): corrige la garde anti-boucle infinie
test(planner): ajoute tests unitaires pour graphe de dépendances
docs(contracts): documente le schéma RetrievalRequest v1.0
chore(ci): configure le pipeline GitHub Actions
```

### Points de Synchronisation avec le Module ACTION
- **Sprint 1 (fin)** : Revue et gel des contrats d'interface JSON/TOON
- **Sprint 6 (milieu)** : Test d'intégration du nœud `retrieve` avec le module ACTION réel
- **Sprint 8 (fin)** : Session de validation end-to-end complète

---

*Document mis à jour le 2026-08-04 — Version 2.0 — RAG-REASON Module REASONING*
*Ce plan est un document vivant. Mettre à jour les cases à cocher au fil de l'avancement.*
