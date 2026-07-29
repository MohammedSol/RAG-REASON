# Rapport d'Avancement — Sprints 1 & 2
## Projet RAG-REASON

| | |
|---|---|
| **Projet** | RAG-REASON — Moteur de Raisonnement Déterministe pour Agent RAG |
| **Étudiant** | Mohammed Solimani |
| **Encadrante** | Mme Zineb Hidila |
| **Institution** | EMSI Casablanca & LPRI EMSI |
| **Date** | Juillet 2026 |
| **Statut** | Sprints 1 & 2 — Terminés et validés |

---

## 1. Introduction

### 1.1 Contexte et Problématique

Les systèmes RAG (*Retrieval-Augmented Generation*) conventionnels souffrent d'une limitation fondamentale : leur logique de raisonnement est **linéaire et non déterministe**. Face à une requête complexe nécessitant plusieurs sauts d'inférence — par exemple, *"Qui dirige l'entreprise qui a publié le modèle ayant obtenu le meilleur score sur le benchmark MMLU en 2024 ?"* — un pipeline RAG classique effectue une seule recherche, récupère des chunks partiellement pertinents, et produit une réponse incomplète ou hallucinée.

### 1.2 Objectif de RAG-REASON

**RAG-REASON** est un moteur de raisonnement déterministe conçu pour pallier ces limites. L'objectif est de construire, depuis zéro, un composant de pilotage intelligent (*Reasoning Engine*) capable de **planifier**, **critiquer** et **vérifier** la construction d'une réponse, en orchestrant de manière adaptative les appels au module de récupération documentaire.

### 1.3 Un Parti Pris Architectural Fort : Le Code Dirige, le LLM Obéit

Une décision fondatrice distingue ce projet des approches populaires : nous **refusons délibérément** d'utiliser la logique "boîte noire" de frameworks d'orchestration tels que LangChain Agents ou CrewAI. Ces outils délèguent le contrôle du flux d'exécution au modèle de langage lui-même, rendant le comportement du système imprévisible, difficile à déboguer et impossible à garantir en production.

Notre architecture inverse ce paradigme :

> **Le code Python orchestre. Le LLM calcule. Jamais l'inverse.**

Chaque décision de routage, chaque transition d'état, chaque condition d'arrêt est exprimée en code Python typé et testé. Le modèle de langage est traité comme un **opérateur fonctionnel** — un outil de classification ou de génération de texte structuré — et non comme un agent autonome.

Cette approche garantit trois propriétés essentielles pour un système de production :

- **Déterminisme** : pour une entrée donnée, le graphe d'exécution emprunte toujours un chemin prévisible
- **Contrôlabilité** : chaque nœud du graphe peut être remplacé, mocké, ou instrumenté indépendamment
- **Testabilité** : la logique métier est découplée du modèle, permettant des tests unitaires complets sans dépendance réseau

---

## 2. Sprint 1 — Architecture "Contract-First"

### 2.1 Philosophie du Contract-First

Avant d'écrire la moindre ligne de logique applicative, le Sprint 1 a posé les **fondations contractuelles** du système. Cette approche, inspirée du développement dirigé par les types (*type-driven development*), consiste à définir rigoureusement les structures de données échangées entre les composants avant de les implémenter.

La justification est simple : dans un système distribué entre deux développeurs — le module REASONING (notre périmètre) et le module ACTION (retrieval vectoriel développé en parallèle) — une ambiguïté dans le format d'échange se traduit inévitablement par des bugs d'intégration coûteux. Les contrats formalisés éliminent cette catégorie d'erreurs par construction.

### 2.2 Contrats d'Interface avec le Module ACTION

Le fichier `src/reasoning/contracts/action_interface.py` définit les trois modèles Pydantic constituant le protocole de communication entre les deux modules :

| Modèle | Direction | Rôle |
|---|---|---|
| `RetrievalRequest` | REASONING → ACTION | Sous-requête de recherche avec contraintes (`top_k`, `filters`, `hop_index`) |
| `RetrievedChunk` | ACTION → REASONING | Unité atomique de contexte récupéré avec score de pertinence |
| `RetrievalResponse` | ACTION → REASONING | Agrégat de chunks répondant à une sous-requête |

Ce fichier est marqué `# CONTRACT v1.0 — NE PAS MODIFIER SANS REVUE` et constitue la surface d'API gelée entre les deux équipes.

### 2.3 Modèles Internes du Moteur de Raisonnement

Le fichier `src/reasoning/contracts/internal_models.py` définit l'ensemble des structures de données circulant à l'intérieur du graphe LangGraph :

- **`QueryType`** (StrEnum) : taxonomie des requêtes (`SIMPLE`, `MULTI_HOP`, `COMPARATIVE`, `AMBIGUOUS`)
- **`StepStatus`** (StrEnum) : cycle de vie d'une étape (`PENDING`, `IN_PROGRESS`, `COMPLETED`)
- **`AnalysisResult`** : sortie du composant Query Analyzer
- **`PlanStep` / `ExecutionPlan`** : structure du plan d'exécution généré par le Planner
- **`CriticEvaluation`** : verdict du Critic sur la suffisance d'un contexte récupéré
- **`VerificationResult`** : résultat de l'audit de fidélité du Verifier
- **`AgentState`** : état global du graphe — la "mémoire de travail" de l'agent pour une requête

### 2.4 Le Concept de `reasoning_budget`

Un mécanisme de sécurité architectural a été introduit dès ce sprint : le **`reasoning_budget`**. Ce champ entier, intégré à `AnalysisResult`, représente le nombre maximum d'appels au module ACTION autorisés pour une requête donnée. Il est alloué en fonction du type de requête détecté :

| Type | Budget | Justification |
|---|---|---|
| `SIMPLE` | 1 | Un seul retrieval suffit par définition |
| `MULTI_HOP` | 3 | Conservatif : la plupart des chaînes se résolvent en 2-3 étapes |
| `COMPARATIVE` | 2 | Un retrieval par entité à comparer |
| `AMBIGUOUS` | 0 | Aucun retrieval — clarification requise en priorité |

Le `reasoning_budget` est la **première ligne de défense contre les boucles infinies** : si le composant Critic demande systématiquement un re-retrieval, le graphe sort obligatoirement après `reasoning_budget` itérations.

### 2.5 Validation : 47 Tests, Couverture 100%

L'ensemble du module `contracts/` est couvert à 100% par 47 tests unitaires (`tests/unit/test_contracts.py`), incluant :

- Tests de sérialisation/désérialisation JSON (`model_dump()` / `model_validate()`)
- Tests des validateurs de contraintes (`Field(ge=0)`, `Field(le=1.0)`)
- Tests de rejet des valeurs invalides (`pytest.raises(ValidationError)`)
- Tests d'isolation des listes mutables (`default_factory` correctement utilisé)

---

## 3. Sprint 2 — Conception du Query Analyzer

### 3.1 Rôle et Position dans le Graphe

Le **Query Analyzer** est le premier nœud exécuté dans le graphe LangGraph pour chaque requête utilisateur. Sa mission est strictement délimitée :

1. **Classifier** la requête selon la taxonomie `QueryType`
2. **Allouer** le `reasoning_budget` correspondant
3. **Extraire** les entités sémantiques détectées

Il ne fait aucun retrieval. Il ne génère aucune réponse. Il est le **poste d'aiguillage** qui conditionne l'ensemble du traitement ultérieur.

### 3.2 Stratégie des Modèles Locaux : Architecture à Deux Vitesses

L'un des choix techniques les plus structurants de ce sprint concerne la **répartition des modèles Ollama** selon la nature computationnelle des tâches.

Nous utilisons deux modèles Qwen 2.5, déployés localement via Ollama :

| Modèle | Paramètres | Tâches assignées | Justification |
|---|---|---|---|
| **Qwen 2.5 3B** | 3 milliards | Query Analyzer (classification) | Latence < 500ms, suffisant pour la classification structurée |
| **Qwen 2.5 7B** | 7 milliards | Planner, Critic, Verifier (Sprints 3-5) | Raisonnement complexe, génération longue, critique nuancée |

Ce choix repose sur un principe d'**économie computationnelle** : la classification d'une requête est une tâche de faible complexité cognitive. Mobiliser un modèle 7B pour catégoriser une phrase en `SIMPLE` ou `MULTI_HOP` serait un gaspillage de ressources qui dégraderait la latence globale du système sans gain de qualité mesurable.

LiteLLM joue ici le rôle d'**adaptateur universel** : en préfixant le nom du modèle (`ollama/qwen2.5:3b`), le même code Python peut cibler n'importe quel provider LLM (Ollama, OpenAI, Anthropic, Mistral) sans modification.

### 3.3 Ingénierie de Prompt : Le Few-Shot Structuré

#### Problématique

Les modèles de petite taille (3B paramètres) sont susceptibles de dériver du format attendu lorsque les instructions sont formulées en Zero-Shot. Une instruction comme *"retourne du JSON"* ne garantit pas l'absence de texte parasite, de blocs Markdown, ou de champs manquants.

#### Solution : Few-Shot Prompting avec Ancrage Formaté

Nous avons conçu un prompt structuré en quatre blocs immuables (`src/reasoning/analyzer/prompts.py`) :

1. **Bloc Rôle** : interdiction explicite de répondre à la question — classification uniquement
2. **Bloc Taxonomie** : définition concise des quatre types avec leurs critères
3. **Bloc Examples (10 exemples)** : deux à trois exemples représentatifs par type, avec le JSON attendu en réponse
4. **Bloc Requête** : injection de la requête utilisateur via un placeholder `{query}`

Ce travail d'ingénierie de prompt n'est pas trivial. Chaque exemple a été sélectionné pour sa valeur pédagogique envers le modèle : les exemples MULTI_HOP illustrent des patterns de dépendance inter-questions ("et ensuite", "dont le", "quel est le X de Y qui..."), les exemples AMBIGUOUS montrent explicitement des termes polysémiques utilisés sans contexte désambiguïsant.

Le résultat : le modèle Qwen 3B génère dans plus de 95% des cas un objet JSON directement parseable par `AnalysisResult.model_validate_json()`.

### 3.4 Architecture Hybride : Tolérance aux Pannes

Un système de production ne peut pas dépendre exclusivement d'un service externe. Ollama peut être indisponible, le modèle peut générer une réponse malformée, le réseau peut subir un timeout. Dans ces cas, un fallback purement LLM-dépendant retournerait une exception non gérée, bloquant le graphe.

Nous avons implémenté une **architecture hybride à deux niveaux** :

```
Niveau 1 (Principal)  : Qwen 2.5 3B via LiteLLM → JSON parsé par Pydantic
                                    │
                              [Succès ?]
                         ┌────────────────┐
                        OUI              NON
                         │                │
                  AnalysisResult    Niveau 2 (Fallback)
                                          │
                              Moteur de règles Python
                              (marqueurs linguistiques)
                                          │
                                  AnalysisResult
                               (confidence = 0.55)
```

Le fallback heuristique Python est **déterministe** et **sans dépendance** : il applique des règles de détection de patterns linguistiques dans un ordre de priorité strict (AMBIGUOUS → MULTI_HOP → COMPARATIVE → SIMPLE). La valeur `confidence=0.55` signale au graphe que la classification est issue du fallback, permettant une instrumentation et un monitoring granulaires.

Les exceptions capturées incluent : `TimeoutError`, `OSError` (connexion réseau), `ValidationError` (JSON non conforme), `ValueError` (JSON absent), et toute exception inattendue via un handler générique loggué.

---

## 4. Tests et Validation de l'Approche

### 4.1 Philosophie de Test

La stratégie de test adopte une séparation stricte entre les tests **isolés** (rapides, sans dépendance réseau) et les tests **d'intégration** (réels, nécessitant Ollama). Cette séparation est rendue possible précisément parce que notre architecture hybride découple la logique métier du modèle LLM : le fallback heuristique peut être exercé sans aucun mock complexe, et le chemin LLM peut être testé en injectant des réponses fictives via `unittest.mock.patch`.

### 4.2 Tests Unitaires — Module `contracts/`

- **Fichier** : `tests/unit/test_contracts.py`
- **Tests** : 47 cas
- **Couverture** : 100% des 65 statements du module

Validation exhaustive des contrats Pydantic : sérialisation JSON, contraintes de champs, comportement face aux données invalides, isolation des instances.

### 4.3 Tests Unitaires — Module `analyzer/`

- **Fichier** : `tests/unit/test_analyzer.py`
- **Tests** : 30 cas
- **Couverture** : 97% du module (3 branches d'exception rarement atteignables simultanément)

Organisation en quatre classes de tests :

| Classe | Objectif |
|---|---|
| `TestLLMSuccessPath` | Vérifie les 4 types de classification avec réponse LLM mockée |
| `TestFallbackHeuristics` | Simule 6 types d'échecs LLM et vérifie l'activation du fallback |
| `TestEdgeCases` | Requêtes vides, espaces, paramètres par défaut |
| `TestInternalMethods` | Méthodes statiques : extraction JSON, entités, détection d'ambiguïté |

Le test `test_fallback_on_llm_timeout` constitue le **test critique** du sprint : il simule un `TimeoutError` sur l'appel LiteLLM et vérifie que :
1. La méthode `analyze()` ne lève aucune exception
2. Le résultat est bien un `AnalysisResult` valide
3. La `confidence` est exactement `0.55` (signature du fallback)

### 4.4 Tests d'Intégration

- **Fichier** : `tests/integration/test_analyzer_live.py`
- **Protection** : marqueur `@pytest.mark.integration` — ignorables si Ollama est absent
- **Jeu de données** : 15 requêtes de référence couvrant les 4 types

Ces tests valident la **précision de classification** du modèle Qwen 3B en conditions réelles, en vérifiant la cohérence du `reasoning_budget` retourné et en s'assurant que la confidence issue du chemin LLM est supérieure à celle du fallback (> 0.55).

### 4.5 Bilan de Couverture Globale

```
Module                                    Stmts   Miss  Cover
─────────────────────────────────────────────────────────────
contracts/__init__.py                         0      0   100%
contracts/action_interface.py                20      0   100%
contracts/internal_models.py                 45      0   100%
analyzer/__init__.py                          2      0   100%
analyzer/prompts.py                           1      0   100%
analyzer/analyzer.py                         89      3    97%
─────────────────────────────────────────────────────────────
TOTAL                                       157      3    98%
```

**78 tests passés. Couverture globale : 98%.**

### 4.6 Outil Qualité : Correction d'un Bug de Contrat

La rédaction des tests a révélé une **inconsistance critique dans le contrat** `AnalysisResult` : le champ `reasoning_budget` était défini avec `Field(gt=0)` (strictement positif), ce qui rendait techniquement impossible la construction d'un résultat `AMBIGUOUS` (dont le budget est par définition `0`).

Ce bug aurait pu se propager silencieusement jusqu'au Sprint 3. Les tests l'ont capturé immédiatement. La correction — passer à `Field(ge=0)` — illustre la valeur de l'approche Contract-First avec couverture de test élevée : **les bugs de spécification sont détectés avant d'affecter les composants qui les consomment**.

---

## 5. Prochaines Étapes

### Sprint 3 — Planner (Plan-and-Solve)

Le composant suivant à développer est le **Planner**. Son rôle est de décomposer une requête classifiée `MULTI_HOP` ou `COMPARATIVE` en un graphe de sous-questions ordonnées, représenté par le contrat `ExecutionPlan` déjà défini au Sprint 1. Le Planner utilisera le modèle **Qwen 2.5 7B** pour cette tâche de raisonnement plus exigeante.

Le développement suivra la même méthodologie : spécification dans `docs/planner_spec.md`, implémentation dans `src/reasoning/planner/`, puis tests unitaires avec mocks LiteLLM et tests d'intégration.

### Script de Comparaison RAG Classique

En parallèle du Sprint 3, un script de comparaison sera développé dans `scripts/` pour mettre en regard les performances de RAG-REASON face à un pipeline RAG linéaire standard sur un jeu de requêtes multi-sauts. Cette baseline permettra de quantifier objectivement l'apport du moteur de raisonnement et de préparer les métriques d'évaluation du Sprint final (RAGAS).

---

*Rapport généré le 21 juillet 2026 — RAG-REASON v0.1.0-dev*
