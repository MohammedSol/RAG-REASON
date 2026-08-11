# Spécification Technique — Sprint 6 : Orchestration LangGraph
## Projet RAG-REASON

| | |
|---|---|
| **Document** | `docs/graph_spec.md` |
| **Version** | 1.0 |
| **Sprint** | 6 — Orchestration LangGraph |
| **Auteur** | Module REASONING (session autonome Sprint 5→7) |
| **Date** | Août 2026 |
| **Statut** | Spécification validée — Prête pour implémentation (6.2) |
| **Précédent** | `docs/verifier_spec.md` — Verifier (Sprint 5, validé) |

---

## 1. Rôle et Portée

Le graphe LangGraph est l'**orchestrateur** du module REASONING : il assemble
`QueryAnalyzer`, `Planner`, `Critic`, `Verifier` (composants **purs**, jugés
figés — cf. règles non négociables de la mission) et un nouveau nœud
`generate_answer`, et pilote le flux conditionnel entre eux, y compris les
boucles de rétroaction. Conformément à la contrainte d'architecture de la
mission :

> **Séparation logique / framework.** La logique de décision (quel nœud
> suivant, quand arrêter la boucle) vit dans une classe Python pure
> `ReasoningPolicy` (`src/reasoning/graph/policy.py`), **sans aucun import
> `langgraph`**. LangGraph n'est qu'un adaptateur d'exécution qui appelle
> `ReasoningPolicy` depuis les fonctions de routage (`edges.py`).
> Vérifiable par `grep langgraph src/reasoning/graph/policy.py` → vide.

---

## 2. Nœuds du Graphe

| Nœud | Composant appelé | Appel LLM ? |
|---|---|---|
| `analyze_query` | `QueryAnalyzer.analyze()` | Oui (ou pré-classificateur regex / fallback) |
| `plan` | `Planner.decompose()` | Non si SIMPLE, oui sinon |
| `retrieve` | `ActionClient.retrieve()` (HTTP vers module ACTION) | Non (pas de LLM, appel réseau) |
| `critique` | `Critic.evaluate()` | Oui |
| `generate_answer` | Appel LLM direct (pas de composant dédié — cf. §6) | Oui |
| `verify` | `Verifier.verify()` | Oui |
| `END` | — | — |
| `clarify` | Aucun appel — construction Python directe (cf. §4.4) | Non |

### 2.1 Nœud `generate_answer` — décision de conception

`generate_answer` n'est **pas** un composant dédié avec sa propre spec
(`plan.md` ne prévoit pas de sprint séparé pour lui). Il est implémenté comme
une fonction de nœud dans `graph/nodes.py` qui appelle directement LiteLLM
avec un prompt de synthèse (contexte = tous les chunks accumulés sur tous les
hops + `original_query`), et retourne le texte brut du LLM comme `answer`.

**Décision actée — pas de parsing TOON pour ce nœud.** La règle non
négociable "Sortie LLM toujours en TOON" s'applique aux **jugements
structurés** dont des champs sont extraits vers un modèle Pydantic pour
piloter le routage (Analyzer, Planner, Critic, Verifier). `generate_answer`
produit le texte final destiné à l'utilisateur — il n'y a aucun champ à en
extraire, la sortie LLM **est** le résultat, exactement comme
`VerificationResult.final_answer` qui est une copie brute de ce texte, jamais
reparsée. TOON n'a donc pas d'objet ici. Cette décision est documentée
explicitement conformément à la procédure d'ambiguïté mineure de la mission.

---

## 3. Le Budget Global Unique — Garde Anti-Boucle (point d'attention critique)

### 3.1 Constat de départ

Trois mécanismes anti-boucle coexistent, non unifiés avant ce sprint :

| Mécanisme | Portée | Défini par | Champ |
|---|---|---|---|
| `reasoning_budget` | Global à la requête | `QueryAnalyzer` | `AgentState.analysis.reasoning_budget` |
| `feedback_loop_count` | Compteur global | Contrat `AgentState` (Sprint 1) | `AgentState.feedback_loop_count` |
| `max_retries` | Local à une `PlanStep` | `Critic` | Non stocké dans le contrat — géré par le graphe |

### 3.2 Décision — Budget global unique

> **`AgentState.feedback_loop_count` est LE compteur global unique. Il est
> incrémenté de 1 à chaque passage par le nœud `critique` OU le nœud
> `verify` (une unité de "raisonnement coûteux" consommée par passage). Le
> plafond est `AgentState.analysis.reasoning_budget`, fixé une fois pour
> toutes par l'Analyzer.**

```
GLOBAL_COUNTER = AgentState.feedback_loop_count   (démarre à 0)
GLOBAL_BUDGET  = AgentState.analysis.reasoning_budget

Invariant : GLOBAL_COUNTER ne décroît jamais, incrémenté de +1 à
            chaque sortie de `critique` et à chaque sortie de `verify`.

Garde prioritaire, vérifiée par ReasoningPolicy AVANT tout appel Critic/Verifier :
    SI GLOBAL_COUNTER >= GLOBAL_BUDGET → forcer la sortie
       (depuis critique  → generate_answer directement, contexte partiel)
       (depuis verify    → END directement, réponse non-garantie fondée)
```

**Justification du choix "incrémenté" (et non "décrémenté depuis le
budget")** : le contrat `AgentState` (figé, Sprint 1, non modifiable) déclare
`feedback_loop_count: int = Field(default=0, ge=0)`. Une sémantique de
décrément depuis le budget nécessiterait d'initialiser ce champ à la valeur
du budget — impossible car le budget n'est connu qu'après l'exécution de
l'Analyzer, alors que `AgentState` est instancié avec `feedback_loop_count`
déjà à sa valeur par défaut `0`. La sémantique incrément-vers-plafond est de
plus **celle déjà actée** dans `analyzer_spec.md §3.2` et `critic_spec.md
§5.2` ("Si `feedback_loop_count >= reasoning_budget` → forcer la sortie").
Ce sprint ne fait qu'unifier officiellement ce mécanisme déjà cohérent avec
le `max_retries` local du Critic — il ne l'invente pas.

### 3.3 Composition avec `max_retries` (compteur local par étape)

`max_retries` (Critic, défaut `2`) borne le nombre de **re-retrievals pour
une même `PlanStep`** — un sous-mécanisme local imbriqué dans le budget
global. Le compteur local `retry_counts: dict[str, int]` n'existe **pas**
dans le contrat `AgentState` figé (question ouverte non tranchée dans
`critic_spec.md §11.2`). Il est donc porté par l'état d'exécution
**spécifique au graphe** (`GraphState`, cf. §5), jamais par le contrat
`AgentState` lui-même.

```
Routage post-critique (ReasoningPolicy.route_after_critique) :
    SI GLOBAL_COUNTER >= GLOBAL_BUDGET
        → generate_answer                              (garde globale, prioritaire)
    SINON SI critic_eval.is_sufficient
        → étape suivante du plan, ou generate_answer si plan épuisé
    SINON SI retry_counts[step_id] >= critic.max_retries
        → étape suivante du plan (contexte partiel accepté), ou generate_answer
          (garde locale : abandon de cette étape, log WARNING)
    SINON
        → retrieve (même step_id, incrémente retry_counts[step_id])
```

### 3.4 Garantie d'arrêt (preuve de terminaison)

Avec `GLOBAL_BUDGET` fini (0, 1, 2 ou 3 selon `QueryType`) et `max_retries`
fini (défaut 2) :

- Le nombre de passages par `critique`/`verify` est borné par `GLOBAL_BUDGET`
  (garde globale prioritaire, vérifiée avant CHAQUE appel Critic/Verifier).
- Pour une étape donnée, le nombre de `retrieve` consécutifs est borné par
  `max_retries` (garde locale).
- `plan` peut être ré-exécuté après un échec de `verify`, mais chaque
  passage par `verify` consomme une unité du budget global — donc le nombre
  de re-planifications est lui aussi borné par `GLOBAL_BUDGET`.

**Conclusion :** le nombre total de nœuds visités par requête est fini et
borné par une fonction de `GLOBAL_BUDGET` et `max_retries`, indépendamment du
comportement du LLM. Un test dédié (`test_budget_exhaustion_forces_exit`,
Sprint 6.3) valide cette garantie en mockant un Critic qui retourne
systématiquement `is_sufficient=False`.

---

## 4. Arêtes Conditionnelles

### 4.1 Après `analyze_query`

```
SI analysis.reasoning_budget == 0        (AMBIGUOUS)
    → clarify → END
SINON
    → plan
```

**Décision — tous les types non-AMBIGUOUS passent par `plan`, y compris
SIMPLE.** `plan.md` mentionne "retrieve directement (si SIMPLE)" comme
simplification de haut niveau ; en pratique le `Planner` déjà construit
court-circuite tout appel LLM pour `QueryType.SIMPLE`
(`planner_spec.md §2` : "aucun appel LLM... latence minimale"). Router
systématiquement par `plan` élimine un embranchement redondant dans le
graphe (un seul contrat d'entrée pour `retrieve` : une `PlanStep`, jamais
une requête brute) sans coût de latence supplémentaire mesurable. Décision
mineure documentée conformément à la procédure de la mission.

### 4.2 Après `plan`

```
→ retrieve (première PlanStep du front d'exécution, cf. planner_spec.md §4)
```

### 4.3 Après `retrieve`

```
→ critique
```

### 4.4 Après `critique`

Cf. §3.3 (logique complète intégrée à `ReasoningPolicy.route_after_critique`).
Sortie possible : `retrieve` (re-boucle), `retrieve` (étape suivante du
plan), ou `generate_answer` (plan épuisé ou budget/retries épuisés).

### 4.5 Après `generate_answer`

```
→ verify
```

### 4.6 Après `verify`

```
SI verify.is_grounded
    → END
SINON SI GLOBAL_COUNTER >= GLOBAL_BUDGET
    → END                          (garde globale — réponse non garantie fondée,
                                     mais présentée avec le VerificationResult tel quel :
                                     is_grounded=False reste visible par l'appelant)
SINON
    → plan                         (re-planification, feedback textuel injecté
                                     dans la query — cf. §4.7)
```

### 4.7 Re-planification après `verify` échoué — mécanisme de feedback

`Planner.decompose(query, analysis)` est un composant **figé** (liste des
composants non modifiables de la mission) : sa signature n'accepte pas de
paramètre de feedback. Pour transmettre les `unsupported_claims` du Verifier
au prochain appel du Planner **sans modifier `planner.py`**, le nœud `plan`
du graphe (pas le Planner lui-même) construit une requête augmentée :

```python
augmented_query = (
    f"{original_query} "
    f"(Additional verification needed on: {', '.join(unsupported_claims)})"
)
plan = planner.decompose(augmented_query, analysis)
```

Le contrat `ExecutionPlan.original_query` reflétera alors la requête
augmentée sur les tours de re-planification — comportement acceptable et
documenté, la requête utilisateur affichée à l'écran restant
`AgentState.original_query` (jamais réécrite).

### 4.8 Diagramme Mermaid

Voir `docs/graph.md`.

---

## 5. État du Graphe — `GraphState` vs contrat `AgentState`

### 5.1 Décision de conception (point structurant, tranché explicitement)

`plan.md §6.2` demande de "Créer `src/reasoning/graph/state.py` définissant
`AgentState` avec `TypedDict`". Or `AgentState` existe déjà comme modèle
**Pydantic figé** dans `contracts/internal_models.py` (Sprint 1, testé en
CI, utilisé ailleurs — ex. Streamlit). Redéfinir un second `AgentState` en
`TypedDict` créerait deux représentations divergentes du même concept —
contraire à la contrainte d'architecture de la mission (contrats figés,
source de vérité unique).

**Décision retenue :** `graph/state.py` définit un **`GraphState(TypedDict)`
distinct**, qui est le schéma d'exécution LangGraph. Il **enveloppe** le
contrat `AgentState` figé (champ `agent_state: AgentState`) et ajoute
uniquement les champs de bookkeeping **propres à l'exécution du graphe**,
absents du contrat car hors de son périmètre (accumulation multi-hop,
compteur local de retries — question ouverte non tranchée dans
`critic_spec.md §11.2`, tranchée ici) :

```python
class GraphState(TypedDict):
    agent_state: AgentState                       # contrat figé, inchangé
    retrieved_chunks: list[RetrievedChunk]         # accumulation tous hops (pour Verifier)
    retry_counts: dict[str, int]                   # compteur LOCAL par step_id (Critic)
    current_step_id: str | None                    # étape PlanStep en cours de traitement
    answer: str | None                             # réponse candidate (generate_answer)
```

Chaque nœud lit/écrit `state["agent_state"]` pour tout ce qui appartient au
contrat (analysis, plan, evaluations, verification, feedback_loop_count) et
les champs additionnels pour la plomberie d'exécution. `run_agent()`
(`agent.py`) extrait `state["agent_state"].verification` en sortie pour
respecter la signature `run_agent(query: str) -> VerificationResult` de
`plan.md §6.2`.

Cette décision est documentée ici conformément à la procédure "ambiguïté
mineure" de la mission : elle ne modifie aucun contrat existant, n'affecte
aucune signature publique déjà testée, et respecte la contrainte
"aucune modification de `contracts/`".

---

## 6. `ReasoningPolicy` — Logique Pure Python

`src/reasoning/graph/policy.py` contient une classe `ReasoningPolicy` sans
aucun import `langgraph`, exposant des méthodes pures (entrée = état actuel,
sortie = décision de routage sous forme de chaîne) :

```python
class ReasoningPolicy:
    def route_after_analysis(self, agent_state: AgentState) -> str: ...
    def route_after_critique(
        self, agent_state: AgentState, evaluation: CriticEvaluation,
        retry_count: int, max_retries: int, has_next_step: bool,
    ) -> str: ...
    def route_after_verification(
        self, agent_state: AgentState, verification: VerificationResult,
    ) -> str: ...
```

`graph/edges.py` contient les fonctions de routage LangGraph (signature
imposée par `add_conditional_edges`), qui ne font qu'extraire les données du
`GraphState` et déléguer à `ReasoningPolicy`. Vérifiable :
`grep langgraph src/reasoning/graph/policy.py` → aucune occurrence.

---

## 7. Client HTTP vers le module ACTION

`src/reasoning/action_client.py` implémente `ActionClient`, un client HTTP
minimal (bibliothèque `httpx`, nouvelle dépendance — cf. rapport final) qui
sérialise `RetrievalRequest` → POST JSON → désérialise `RetrievalResponse`,
conformément au contrat figé `contracts/action_interface.py` et à
`docs/ACTION_INTEGRATION_HANDOFF.md §7.1` (`POST /retrieve`).

Contraintes (cf. règles de la mission) :
- Timeout explicite et configurable (défaut `10.0s`).
- Gestion d'erreur : toute exception réseau (`httpx.RequestError`,
  `httpx.TimeoutException`, `httpx.HTTPStatusError`) est catchée dans le nœud
  `retrieve` et transformée en `RetrievalResponse(chunks=[], ...)` défensif
  (fail-closed — le `Critic` traite alors le cas "aucun chunk" déjà
  spécifié).
- **Double de test injectable** : `ActionClient.retrieve()` est appelée via
  une interface (`Protocol`) que les tests peuvent substituer par un mock
  sans dépendance réseau réelle. Le module ACTION n'étant pas encore
  branché, aucun test n'suppose une API distante joignable (tests
  d'intégration Sprint 6.3 utilisent exclusivement un double en mémoire).

---

## 8. Fichiers à Créer

```
src/reasoning/graph/
├── __init__.py         # Exporte build_graph / le graphe compilé
├── state.py             # GraphState (TypedDict), initialisation
├── policy.py             # ReasoningPolicy — ZÉRO import langgraph
├── nodes.py               # Fonctions de nœuds async (analyze_node, plan_node, ...)
└── edges.py                # Fonctions de routage LangGraph → délèguent à ReasoningPolicy

src/reasoning/action_client.py   # Client HTTP vers le module ACTION
src/reasoning/agent.py            # run_agent(query: str) -> VerificationResult
```

---

## 9. Plan de Vérification (Sprint 6.3)

| Test | Objectif |
|---|---|
| `test_simple_query_full_path` | SIMPLE : analyze → plan(1 step, 0 LLM) → retrieve → critique → generate_answer → verify → END |
| `test_multi_hop_two_steps` | MULTI_HOP 2 étapes, critique échoue étape 1 puis réussit |
| `test_replanning_loop_on_verify_failure` | verify échoue → retour à `plan` avec feedback → 2e tentative |
| `test_budget_exhaustion_forces_exit` | Critic mocké toujours insuffisant → sortie forcée après `reasoning_budget` passages, **jamais de boucle infinie** |
| `test_ambiguous_query_routes_to_clarify` | AMBIGUOUS → `clarify` → END sans aucun appel retrieve |
| `test_action_client_double_injectable` | `retrieve` fonctionne avec un double en mémoire, sans réseau réel |
| `test_policy_has_no_langgraph_import` | `grep langgraph src/reasoning/graph/policy.py` vide (vérifié en test Python via lecture du fichier source) |

---

*Spécification RAG-REASON — Orchestration LangGraph v1.0 — Sprint 6*
*À lire en conjonction avec `docs/verifier_spec.md`, `docs/critic_spec.md` et
`src/reasoning/contracts/internal_models.py`*
