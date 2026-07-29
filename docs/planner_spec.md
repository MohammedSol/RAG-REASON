# Spécification Technique — Sprint 3 : Le Planner (Plan-and-Solve)
## Projet RAG-REASON

| | |
|---|---|
| **Document** | `docs/planner_spec.md` |
| **Version** | 1.0 |
| **Sprint** | 3 — Planner |
| **Auteur** | Mohammed Solimani |
| **Date** | Juillet 2026 |
| **Statut** | Spécification validée — Prête pour implémentation (3.2) |
| **Précédent** | `docs/analyzer_spec.md` — Query Analyzer (Sprint 2, validé) |

---

## 1. Rôle et Position dans le Graphe

Le **Planner** est le deuxième nœud du graphe de raisonnement. Il reçoit en entrée :

- La **requête originale** de l'utilisateur
- L'objet `AnalysisResult` produit par le Query Analyzer (contenant `query_type`, `reasoning_budget`, et `detected_entities`)

Sa mission est de produire en sortie un objet `ExecutionPlan` : un **graphe acyclique dirigé (DAG)** de sous-requêtes atomiques, chacune correspondant à un appel futur au module ACTION (retrieval).

```
                   ┌─────────────────┐
 query + Analysis  │                 │  ExecutionPlan (DAG)
 ─────────────────►│    PLANNER      │──────────────────────►  Executor
                   │                 │
                   └─────────────────┘
                         │  Appel Qwen 2.5 7B
                         │  (sauf cas SIMPLE)
                         ▼
                   Réponse TOON brute
```

### Invariants

1. Le Planner ne génère **jamais** de réponse finale à l'utilisateur
2. Il ne fait **jamais** d'appel au module ACTION directement
3. Il est **stateless** : chaque appel à `decompose()` est indépendant
4. L'implémentation est en **Python pur** — aucune bibliothèque d'orchestration externe

---

## 2. Cas Dégénéré — Requête SIMPLE (Court-Circuit Python)

### Règle Métier Absolue

> Si `AnalysisResult.query_type == QueryType.SIMPLE`, le Planner **ne doit faire aucun appel LLM**. Il construit directement, en code Python, un `ExecutionPlan` à une seule étape et le retourne immédiatement.

Cette règle garantit une latence minimale pour les requêtes atomiques, qui représentent la majorité des cas d'utilisation en production.

### Comportement Attendu

```python
# Pseudo-code illustratif — pas le code final

def decompose(self, query: str, analysis: AnalysisResult) -> ExecutionPlan:
    if analysis.query_type == QueryType.SIMPLE:
        single_step = PlanStep(
            step_id="step_1",
            sub_query=query,
            depends_on=[],           # aucune dépendance → exécutable immédiatement
            status=StepStatus.PENDING,
        )
        return ExecutionPlan(
            plan_id=_generate_plan_id(query),
            original_query=query,
            steps=[single_step],
            dependencies_graph={"step_1": []},
        )
    # ... suite : appel LLM pour MULTI_HOP et COMPARATIVE
```

### Justification

Les requêtes `SIMPLE` ont un `reasoning_budget=1` par définition. Construire un plan à une étape via le LLM coûterait ~500ms supplémentaires pour un résultat identique. Le court-circuit Python est donc une **optimisation correcte et non un raccourci**.

---

## 3. Gestion du Budget d'Exécution

### Règle du Budget

Le Planner reçoit `AnalysisResult.reasoning_budget` depuis l'Analyzer. Ce budget est une **contrainte dure** :

> **Le nombre total d'étapes dans `ExecutionPlan.steps` ne doit jamais dépasser `reasoning_budget`.**

Cette règle est appliquée à deux niveaux :

1. **Dans le prompt** : le LLM reçoit explicitement le budget comme contrainte dans sa consigne (voir §5)
2. **En post-processing Python** : après parsing de la réponse TOON, si le nombre d'étapes dépasse le budget, les étapes excédentaires (celles à `depends_on` non vides, i.e. les moins prioritaires) sont tronquées avec un log `WARNING`

### Table de Correspondance Budget / Type

| `query_type` | `reasoning_budget` | Comportement Planner |
|---|---|---|
| `SIMPLE` | 1 | Court-circuit Python, 1 étape, 0 appel LLM |
| `MULTI_HOP` | 3 | Appel LLM, 2 à 3 étapes séquentielles ou mixtes |
| `COMPARATIVE` | 2 | Appel LLM, 2 étapes parallèles |
| `AMBIGUOUS` | 0 | **Jamais atteint** — l'Executor intercepte avant le Planner |

---

## 4. Format du Graphe de Dépendances (DAG)

### Représentation : Liste d'Adjacence

Le graphe est encodé dans le champ `dependencies_graph` de `ExecutionPlan`, de type `dict[str, list[str]]`.

Chaque entrée mappe un `step_id` vers la liste des `step_id` dont il dépend (**prérequis**) :

```python
dependencies_graph = {
    "step_1": [],          # aucun prérequis → exécutable immédiatement
    "step_2": [],          # aucun prérequis → exécutable en parallèle avec step_1
    "step_3": ["step_1", "step_2"],  # doit attendre step_1 ET step_2
}
```

### Règle Fondamentale d'Ordonnancement

> **Toute étape dont la liste `depends_on` est vide (ou `None`) est exécutable immédiatement, potentiellement en parallèle avec les autres étapes sans dépendances.**

Formellement : soit `S` l'ensemble des étapes du plan et `D(s)` l'ensemble des prérequis de l'étape `s`. Le **front d'exécution initial** est :

```
F₀ = { s ∈ S | D(s) = ∅ }
```

Après l'exécution complète de toutes les étapes de `Fₙ`, le front suivant est :

```
Fₙ₊₁ = { s ∈ S \ (F₀ ∪ ... ∪ Fₙ) | D(s) ⊆ F₀ ∪ ... ∪ Fₙ }
```

### Contrainte de Cohérence

Le Planner doit vérifier, après parsing, que :

1. Chaque identifiant dans `depends_on` référence bien un `step_id` existant dans le plan
2. Le graphe est **acyclique** (détection par tri topologique Kahn — pas de cycle)
3. Le nombre total d'étapes ≤ `reasoning_budget`

Si une violation est détectée, le Planner bascule sur une **stratégie de secours** : construire un plan séquentiel dégradé (chaque étape dépend de la précédente) en utilisant les `sub_query` déjà parsées.

---

## 5. Schéma TOON Attendu

Le LLM (Qwen 2.5 7B via LiteLLM) doit générer une réponse strictement conforme au schéma TOON suivant. **Aucun autre format n'est accepté.**

### 5.1 Format pour un Plan Multi-Étapes

Le LLM génère **un bloc TOON par étape**, précédé d'un bloc d'en-tête contenant la rationale globale :

```
<<<
plan_rationale :: <explication textuelle de la stratégie de décomposition>
total_steps :: <entier : nombre d'étapes — ne doit pas dépasser le budget>
>>>

<<<
step_id :: step_1
sub_query :: <sous-question atomique à envoyer au retriever>
depends_on ::
>>>

<<<
step_id :: step_2
sub_query :: <sous-question atomique à envoyer au retriever>
depends_on :: step_1
>>>

<<<
step_id :: step_3
sub_query :: <sous-question atomique à envoyer au retriever>
depends_on :: step_1 | step_2
>>>
```

### 5.2 Règles Syntaxiques Strictes

| Champ | Type TOON | Contrainte |
|---|---|---|
| `plan_rationale` | Chaîne libre | Obligatoire dans le bloc d'en-tête |
| `total_steps` | Entier | Obligatoire, ≤ `reasoning_budget` |
| `step_id` | Chaîne | Format `step_N` (N entier croissant à partir de 1) |
| `sub_query` | Chaîne libre | Phrase complète, directement utilisable comme requête de recherche |
| `depends_on` | Vide ou liste de `step_id` séparés par `\|` | Vide si aucune dépendance |

### 5.3 Parsing Python

Le bloc d'en-tête et les blocs d'étapes sont tous parsés par `parse_toon_to_dict()` de `toon_utils.py`. Le Planner itère sur tous les blocs `<<<...>>>` trouvés dans la réponse brute :

```python
# Pseudo-code illustratif

import re

blocks = re.findall(r"<<<(.*?)>>>", raw_response, re.DOTALL)
# blocks[0] → en-tête (plan_rationale, total_steps)
# blocks[1:] → étapes (step_id, sub_query, depends_on)
```

---

## 6. Prompt Plan-and-Solve (Few-Shot)

Ce prompt est défini dans `src/reasoning/planner/prompts.py`.

```
Tu es un planificateur de requêtes pour un moteur de raisonnement RAG.

MISSION UNIQUE : Décomposer la requête complexe en sous-questions atomiques et
                 indépendantes, chacune pouvant être traitée par un moteur de
                 recherche documentaire.
INTERDICTION ABSOLUE : Ne réponds JAMAIS à la question. Décompose-la uniquement.

─── CONTRAINTES ─────────────────────────────────────────────────────────────

1. Le nombre total d'étapes ne doit PAS dépasser le BUDGET fourni.
2. Chaque `sub_query` doit être une phrase complète, autonome et précise.
3. Le champ `depends_on` doit être VIDE si l'étape n'a pas de prérequis.
4. Si deux étapes peuvent être exécutées indépendamment, laisse `depends_on` vide
   pour toutes les deux — elles seront exécutées en parallèle.

─── FORMAT DE RÉPONSE (TOON STRICT) ────────────────────────────────────────

Un bloc d'en-tête suivi d'un bloc par étape. Aucun texte avant ni après.

─── EXEMPLE 1 : REQUÊTE SÉQUENTIELLE ───────────────────────────────────────

Requête : "Qui dirige l'entreprise qui a créé le modèle GPT-4 ?"
Budget   : 2

<<<
plan_rationale :: La requête nécessite d'abord d'identifier l'entreprise créatrice de GPT-4, puis de trouver son dirigeant actuel. Ces deux étapes sont séquentielles car la seconde dépend du résultat de la première.
total_steps :: 2
>>>

<<<
step_id :: step_1
sub_query :: Quelle entreprise a créé et publié le modèle GPT-4 ?
depends_on ::
>>>

<<<
step_id :: step_2
sub_query :: Qui est le PDG ou directeur général de l'entreprise créatrice de GPT-4 ?
depends_on :: step_1
>>>

─── EXEMPLE 2 : REQUÊTE AVEC ÉTAPES PARALLÈLES ─────────────────────────────

Requête : "Compare les architectures de BERT et GPT-4, puis conclus sur lequel est le plus adapté au résumé automatique."
Budget   : 3

<<<
plan_rationale :: Les informations sur BERT et GPT-4 peuvent être récupérées en parallèle car elles sont indépendantes. La comparaison finale dépend des deux résultats.
total_steps :: 3
>>>

<<<
step_id :: step_1
sub_query :: Quelle est l'architecture technique du modèle BERT et ses caractéristiques principales ?
depends_on ::
>>>

<<<
step_id :: step_2
sub_query :: Quelle est l'architecture technique du modèle GPT-4 et ses caractéristiques principales ?
depends_on ::
>>>

<<<
step_id :: step_3
sub_query :: Comparaison entre BERT et GPT-4 pour la tâche de résumé automatique de texte.
depends_on :: step_1 | step_2
>>>

─── REQUÊTE À DÉCOMPOSER ───────────────────────────────────────────────────

Requête : "{query}"
Budget   : {reasoning_budget}

Génère UNIQUEMENT les blocs TOON. Aucun texte avant ou après.
```

---

## 7. Architecture du Composant `planner/`

### 7.1 Fichiers à Créer

```
src/reasoning/planner/
├── __init__.py          # Exporte Planner
├── planner.py           # Classe Planner (logique principale)
└── prompts.py           # PLANNING_PROMPT (template Few-Shot)
```

### 7.2 Interface Publique de `Planner`

```python
# Pseudo-code de l'interface — pas le code final

class Planner:
    def __init__(
        self,
        model: str = DEFAULT_REASONING_MODEL,   # Qwen 2.5 7B
        api_base: str = OLLAMA_BASE_URL,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None: ...

    def decompose(
        self,
        query: str,
        analysis: AnalysisResult,
    ) -> ExecutionPlan: ...

    # Méthodes internes
    def _plan_with_llm(self, query: str, analysis: AnalysisResult) -> ExecutionPlan: ...
    def _parse_toon_plan(self, raw: str, query: str, analysis: AnalysisResult) -> ExecutionPlan: ...
    def _validate_dag(self, plan: ExecutionPlan) -> bool: ...
    def _build_sequential_fallback(self, query: str, analysis: AnalysisResult) -> ExecutionPlan: ...
    @staticmethod
    def _build_simple_plan(query: str) -> ExecutionPlan: ...
```

### 7.3 Flux d'Exécution Détaillé

```
decompose(query, analysis)
        │
        ├─── [SIMPLE] ──────────────────────────────────────────────────────►  _build_simple_plan()
        │                                                                              │
        │                                                                              ▼
        │                                                                        ExecutionPlan (1 étape)
        │
        └─── [MULTI_HOP / COMPARATIVE] ──► _plan_with_llm()
                        │
                        ▼
               LiteLLM → Qwen 7B (temperature=0)
                        │
                        ▼ Réponse TOON brute
               _parse_toon_plan()
                        │
                        ├── parse_toon_to_dict() × N blocs   (toon_utils.py)
                        ├── Instanciation PlanStep × N
                        ├── _validate_dag()
                        │       │
                        │       ├── [OK]  ──────────────────► ExecutionPlan validé
                        │       └── [FAIL] ─────────────────► _build_sequential_fallback()
                        │
                        └── [ToonParseError / ValidationError / Exception]
                                │
                                ▼
                        _build_sequential_fallback()  (log WARNING)
                                │
                                ▼
                        ExecutionPlan dégradé (séquentiel)
```

### 7.4 Modèle LLM Utilisé

Le Planner utilise **Qwen 2.5 7B** (`ollama/qwen2.5:7b`), configuré via la variable d'environnement `DEFAULT_REASONING_MODEL`. Ce choix est délibéré :

- La **décomposition** est une tâche de raisonnement plus complexe que la classification
- Le modèle 7B offre une meilleure capacité à générer des plans cohérents et bien structurés
- La latence plus élevée (~2-4s) est acceptable car le Planner n'est appelé qu'une fois par requête complexe

---

## 8. Stratégie de Fallback Séquentiel

Si le LLM génère un plan invalide (cycle détecté, dépendances manquantes, nombre d'étapes supérieur au budget, réponse non parseable), le Planner instancie un **plan de secours séquentiel** :

```python
# Pseudo-code du fallback — pas le code final

def _build_sequential_fallback(
    self, query: str, analysis: AnalysisResult
) -> ExecutionPlan:
    """Construit un plan séquentiel dégradé à partir des entités détectées."""
    entities = analysis.detected_entities
    budget = analysis.reasoning_budget

    steps = []
    for i, entity in enumerate(entities[:budget], start=1):
        sub_q = f"Informations sur {entity} en lien avec : {query}"
        step = PlanStep(
            step_id=f"step_{i}",
            sub_query=sub_q,
            # Chaque étape dépend de la précédente → plan purement séquentiel
            depends_on=[f"step_{i - 1}"] if i > 1 else [],
        )
        steps.append(step)

    # Si aucune entité, une seule étape avec la requête complète
    if not steps:
        steps = [PlanStep(step_id="step_1", sub_query=query, depends_on=[])]

    return ExecutionPlan(
        plan_id=_generate_plan_id(query),
        original_query=query,
        steps=steps,
        dependencies_graph={s.step_id: s.depends_on for s in steps},
    )
```

Le plan de secours sacrifie le parallélisme (toutes les étapes sont séquentielles) pour garantir la **correction du graphe** et la **continuité du service**.

---

## 9. Validation du DAG — Détection de Cycles

L'algorithme de détection de cycles utilise le **tri topologique de Kahn** :

```python
# Pseudo-code — pas le code final

from collections import deque

def _validate_dag(self, plan: ExecutionPlan) -> bool:
    """Retourne True si le graphe est un DAG valide (sans cycle)."""
    in_degree = {s.step_id: 0 for s in plan.steps}
    adjacency = {s.step_id: [] for s in plan.steps}
    known_ids = set(in_degree.keys())

    for step in plan.steps:
        for dep in step.depends_on:
            if dep not in known_ids:
                return False   # dépendance vers un step_id inexistant
            adjacency[dep].append(step.step_id)
            in_degree[step.step_id] += 1

    queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return visited == len(plan.steps)   # False si cycle détecté
```

---

## 10. Plan de Vérification

### Tests Unitaires (`tests/unit/test_planner.py`)

| Test | Mock | Assertion clé |
|---|---|---|
| `test_simple_query_no_llm_call` | `@patch(completion)` | `completion` non appelé, `len(steps)==1` |
| `test_multi_hop_sequential_plan` | Réponse TOON mockée | `steps[1].depends_on == ["step_1"]` |
| `test_parallel_plan` | Réponse TOON mockée | `steps[0].depends_on == []` ET `steps[1].depends_on == []` |
| `test_budget_constraint_enforced` | Plan à 5 étapes, budget=3 | `len(plan.steps) <= 3` |
| `test_fallback_on_llm_error` | `side_effect=TimeoutError` | `isinstance(result, ExecutionPlan)` |
| `test_fallback_on_cycle_detected` | TOON avec cycle `A→B→A` | `result` est un plan séquentiel valide |
| `test_fallback_on_invalid_toon` | TOON malformé | Aucun crash, `ExecutionPlan` retourné |
| `test_dag_validation_valid` | Plan sans cycle | `_validate_dag()` retourne `True` |
| `test_dag_validation_cycle` | Plan avec cycle | `_validate_dag()` retourne `False` |
| `test_dag_unknown_dependency` | `depends_on` vers step inconnu | `_validate_dag()` retourne `False` |

### Tests d'Intégration (`tests/integration/test_planner_live.py`)

Protégés par `@pytest.mark.integration`. Nécessitent Ollama + Qwen 7B.

Cible : vérifier que pour les 5 requêtes MULTI_HOP de référence, le plan généré contient au moins 2 étapes et que le DAG est acyclique.

---

*Spécification RAG-REASON — Planner v1.0 — Sprint 3*
*À lire en conjonction avec `docs/analyzer_spec.md` et `docs/migration_toon_strategy.md`*
