# AGENTS.md — Conventions de collaboration · RAG-REASON

> Conventions permanentes entre le développeur et l'assistant, lues à chaque
> session. Ce fichier prime sur les habitudes par défaut de l'assistant.
> Il complète `plan.md`, qui reste la **source de vérité** du projet
> (périmètre, sprints, contrat d'interface avec le module ACTION).

---

## Langue

Communiquer **TOUJOURS en français** : réponses, descriptions d'étapes en
cours, résumés d'actions, rapports de fin de mission. Ne jamais basculer en
anglais, même pour de courtes phrases de progression.

Restent en anglais :

- les messages de commit Git (Conventional Commits) ;
- les noms de variables, de fonctions et le code en général ;
- les sorties brutes d'outils (pytest, ruff, mypy, git…), citées telles quelles.

---

## Fichiers protégés — ne jamais modifier sans instruction explicite

- **`src/reasoning/contracts/`** — contrats figés au Sprint 1, testés en CI et
  partagés avec le module ACTION (`astraexec`, développé en binôme). Toute
  modification casserait l'intégration.
- **`src/reasoning/shared/toon_utils.py`** — figé après son extension du
  Sprint 5 (`parse_toon_records`).
- **Tout test existant** — ne jamais modifier un test pour le faire passer.
  Un test qui casse est un **signal à remonter**, pas à corriger
  silencieusement.

En cas de blocage impliquant l'un de ces fichiers : s'arrêter, expliquer
précisément le blocage, proposer des options — et attendre l'arbitrage.

---

## Contraintes architecturales

- **Composants purs et sans état.** Analyzer, Planner, Critic et Verifier
  jugent et retournent un verdict. Ils ne modifient jamais l'état global, ne
  décident jamais du routage, ne comptent jamais leurs propres itérations.
  Seul l'orchestrateur décide du flux.
- **Séparation logique / framework.** `src/reasoning/graph/policy.py` ne doit
  contenir **aucun import de `langgraph`** : la logique de décision reste
  indépendante du framework d'exécution. Un test automatisé verrouille cette
  contrainte.
- **Sortie LLM toujours au format TOON**, parsée via
  `shared/toon_utils.py`. Jamais de parsing JSON ni de regex maison.
- **Repli fail-closed** en cas d'échec LLM ou de parsing : verdict négatif
  prudent, jamais une heuristique de substitution silencieuse.
- **Piège de typage `_infer_value()`** : cette fonction ne convertit pas les
  booléens. `"true"` / `"false"` arrivent en `str`. Toujours comparer
  explicitement — ne jamais supposer un `bool` Python (`"false"` est *truthy*).

---

## Qualité des tests

- **Aucune assertion tautologique.** `assert score >= 0.0` est interdit :
  chaque assertion vérifie une valeur réelle attendue, calculable à la main.
- **Tests unitaires** : LLM systématiquement mocké, aucun appel réseau.
- **Tests d'intégration** : marqueur `integration`, Ollama réel. Si Ollama est
  indisponible, le signaler explicitement — ne jamais les faire échouer
  silencieusement ni les supprimer.
- **Fixtures réalistes** (`chunk_id`, `source` plausibles), jamais `"test1"` :
  elles serviront lors de l'intégration réelle avec le module ACTION.

---

## Avant toute tâche de développement

1. Lire `plan.md` — source de vérité du projet.
2. Lire la spécification du composant concerné dans `docs/`
   (`analyzer_spec.md`, `planner_spec.md`, `critic_spec.md`,
   `verifier_spec.md`, `graph_spec.md`).

---

## Conventions de code (rappel de `plan.md`)

| Élément | Convention |
|---|---|
| Modules | `snake_case` |
| Classes | `PascalCase` |
| Fonctions / méthodes | `snake_case` |
| Constantes | `UPPER_SNAKE_CASE` |
| Schémas Pydantic | `PascalCase` + suffixe sémantique (`RetrievalRequest`, `AnalysisResult`) |

---

## Git

- **Branches** : `main` protégée et toujours verte (CI obligatoire) ;
  `develop` pour l'intégration ; `feature/sprint-N-composant` pour les
  fonctionnalités. Aucun `push` direct sur `main` — tout passe par PR.
- **Commits** : Conventional Commits, en anglais pour le type et le scope,
  corps en français autorisé.

  ```
  feat(analyzer): implémente la classification LLM-based
  fix(critic): corrige la garde anti-boucle infinie
  test(planner): ajoute tests unitaires pour graphe de dépendances
  docs(contracts): documente le schéma RetrievalRequest v1.0
  chore(ci): configure le pipeline GitHub Actions
  ```

- Un commit par sous-tâche cohérente, jamais un commit global en fin de
  mission. Ne rien pousser sur le dépôt distant sans demande explicite.

---

## Portail qualité

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/ tests/
uv run pytest tests/ -q --tb=short
```

Mypy s'exécute sur les **sources ET les tests**, jamais seulement les sources.

---

## Échec de test connu

`test_analyzer_default_params` échoue de façon **préexistante** : `timeout`
vaut `20.0` alors que le test attend `15.0`.

Hors périmètre — **ne pas le corriger**. Mais il doit rester le **SEUL** échec
de la suite, hors tests d'intégration. Tout autre échec est une régression à
signaler immédiatement.
