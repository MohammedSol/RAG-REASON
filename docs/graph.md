# Graphe d'Orchestration REASONING — Sprint 6

Diagramme du graphe LangGraph assemblé dans `src/reasoning/graph/graph.py`.
Voir `docs/graph_spec.md` pour la spécification complète (nœuds, arêtes
conditionnelles, budget global unique, `ReasoningPolicy`).

```mermaid
flowchart TD
    START([Entrée : original_query]) --> ANALYZE[analyze_query]

    ANALYZE -->|reasoning_budget == 0<br/>AMBIGUOUS| CLARIFY[clarify]
    CLARIFY --> END_NODE([END])

    ANALYZE -->|reasoning_budget > 0| PLAN[plan]

    PLAN --> RETRIEVE[retrieve]

    RETRIEVE --> CRITIQUE[critique]

    CRITIQUE -->|GLOBAL_COUNTER >= GLOBAL_BUDGET<br/>garde globale| GENERATE[generate_answer]
    CRITIQUE -->|is_sufficient=True| NEXTSTEP{Étape suivante<br/>du plan ?}
    CRITIQUE -->|is_sufficient=False<br/>retry_counts < max_retries| RETRIEVE
    CRITIQUE -->|is_sufficient=False<br/>retry_counts >= max_retries<br/>garde locale| NEXTSTEP

    NEXTSTEP -->|Oui| RETRIEVE
    NEXTSTEP -->|Non, plan épuisé| GENERATE

    GENERATE --> VERIFY[verify]

    VERIFY -->|is_grounded=True| END_NODE
    VERIFY -->|is_grounded=False<br/>GLOBAL_COUNTER >= GLOBAL_BUDGET<br/>garde globale| END_NODE
    VERIFY -->|is_grounded=False<br/>budget restant| PLAN

    style CLARIFY fill:#fff3cd,stroke:#997404
    style GENERATE fill:#d1e7dd,stroke:#0f5132
    style END_NODE fill:#e2e3e5,stroke:#41464b
```

## Légende des gardes anti-boucle

- **Garde globale** (`GLOBAL_COUNTER = AgentState.feedback_loop_count`,
  `GLOBAL_BUDGET = AgentState.analysis.reasoning_budget`) : incrémentée à
  chaque passage par `critique` ou `verify`. Prioritaire sur toute autre
  décision de routage.
- **Garde locale** (`retry_counts[step_id]` vs `Critic.max_retries`) :
  bornée par étape (`PlanStep`), imbriquée dans la garde globale.

La combinaison des deux gardes garantit un nombre fini de nœuds visités par
requête, indépendamment du comportement du LLM (cf. `docs/graph_spec.md §3.4`
pour la preuve de terminaison).
