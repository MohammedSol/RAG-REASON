# Spécification Technique — Sprint 4 : Le Critic (Self-RAG Feedback Loop)
## Projet RAG-REASON

| | |
|---|---|
| **Document** | `docs/critic_spec.md` |
| **Version** | 1.0 |
| **Sprint** | 4 — Critic |
| **Auteur** | Mohammed Solimani |
| **Date** | Août 2026 |
| **Statut** | Spécification validée — Prête pour implémentation (4.2) |
| **Précédent** | `docs/planner_spec.md` — Planner (Sprint 3, validé) |

---

## 1. Rôle et Position dans le Graphe

Le **Critic** est le quatrième nœud du graphe de raisonnement. Il est exécuté **après chaque appel au module ACTION** (retrieval) pour évaluer la qualité du contexte reçu avant de décider si la réponse finale peut être générée ou si un nouveau retrieval est nécessaire.

Il reçoit en entrée :
- Un objet `PlanStep` identifiant la sous-question évaluée (champs : `step_id`, `sub_query`, `depends_on`, `status`)
- Un objet `RetrievalResponse` contenant les chunks récupérés par le module ACTION

Sa mission est de produire en sortie un objet `CriticEvaluation` : un verdict structuré indiquant si le contexte est suffisant et, dans le cas contraire, un feedback actionnable permettant au Planner de formuler une meilleure requête.

```
                    ┌─────────────────┐
 PlanStep           │                 │  CriticEvaluation
 ─────────────────► │    CRITIC       │ ──────────────────►  Routeur conditionnel
 RetrievalResponse  │                 │                        │
                    └─────────────────┘                        ├── is_sufficient = True
                          │  Appel Qwen 2.5 7B                 │      └──► generate_answer
                          ▼                                     └── is_sufficient = False
                    Réponse TOON brute                                 └──► retrieve (re-boucle)
```

### Invariants

1. Le Critic ne génère **jamais** de réponse finale à l'utilisateur
2. Il ne fait **jamais** d'appel direct au module ACTION
3. Il est **stateless** : chaque appel à `evaluate()` est indépendant
4. Il ne modifie **pas** le `PlanStep` reçu en entrée — il produit uniquement un `CriticEvaluation`
5. La décision de relancer un retrieval est prise par le **routeur conditionnel du graphe LangGraph**, pas par le Critic lui-même

---

## 2. Critères d'Évaluation

Le Critic évalue le contexte récupéré (`RetrievalResponse.chunks`) selon quatre critères. Ces critères sont évalués **par le LLM via le prompt Chain-of-Thought** (section 5). Ils ne correspondent pas à des champs séparés dans le contrat — ils sont agrégés en un unique `relevance_score` flottant dans `[0.0, 1.0]`.

### 2.1 Pertinence (*Relevance*)

**Définition :** Dans quelle mesure le contenu des chunks récupérés répond-il directement à la `sub_query` de l'étape évaluée ?

**Évalué par :** Le LLM compare explicitement le champ `sub_query` de `PlanStep` avec le contenu textuel des chunks (`RetrievedChunk.content`).

**Indicateurs d'échec :**
- Les chunks parlent d'un sujet adjacent mais ne répondent pas à la question posée
- La `sub_query` porte sur une entité nommée précise, absente des chunks récupérés
- Le `relevance_score` individuel de chaque chunk (`RetrievedChunk.relevance_score`) est uniformément bas

### 2.2 Complétude (*Completeness*)

**Définition :** Le contexte récupéré couvre-t-il tous les aspects nécessaires pour répondre à la `sub_query` ? Une réponse peut être pertinente mais partielle.

**Évalué par :** Le LLM identifie les aspects implicites ou explicites de la `sub_query` et vérifie s'ils sont tous couverts dans les chunks. Les aspects manquants sont reportés dans `missing_aspects`.

**Indicateurs d'échec :**
- La question porte sur plusieurs attributs (ex : "le nom ET la date de fondation") et un seul est couvert
- Le chunk ne donne qu'une réponse partielle ou indirecte
- Le nombre de chunks (`len(chunks)`) est insuffisant au regard de la complexité de la `sub_query`

### 2.3 Cohérence (*Consistency*)

**Définition :** Les différents chunks récupérés sont-ils cohérents entre eux ? Des informations contradictoires réduisent la fiabilité du contexte, même si chacun pris séparément semble pertinent.

**Évalué par :** Le LLM compare les assertions factuelles entre chunks. Ce critère est évalué implicitement dans le raisonnement Chain-of-Thought sans générer de champ dédié.

**Indicateurs d'échec :**
- Deux chunks affirment des dates, valeurs ou faits contradictoires pour la même entité
- Un chunk est significativement plus récent qu'un autre et les informations divergent

> **Note :** En cas de contradiction détectée, le Critic ne tranche pas — il dégrade le `relevance_score` et formule un `feedback` signalant l'incohérence pour permettre au Planner de cibler des sources plus précises.

### 2.4 Fraîcheur (*Freshness*)

**Définition :** Les informations récupérées sont-elles suffisamment récentes pour le contexte de la question ? Ce critère est **conditionnel** : il n'est pertinent que pour les questions portant sur des états changeants (actualités, versions logicielles, postes, statistiques).

**Évalué par :** Le LLM apprécie la fraîcheur à partir du champ `RetrievedChunk.source` (nom du document, URL, date si présente) et du contenu. Pour les questions factuelles atemporelles (ex : "Qui a inventé le transformeur ?"), ce critère est ignoré et n'affecte pas le score.

**Indicateurs d'échec :**
- La question porte sur une version actuelle ou un état présent, et les chunks font référence à des données obsolètes
- La date de publication du source (`chunk.source`) est explicitement ancienne

### 2.5 Agrégation des critères en `relevance_score`

Le LLM produit **un unique score global** dans son bloc TOON. Ce score synthétise les quatre critères selon l'appréciation Chain-of-Thought décrite en section 5. Il n'y a pas de pondération fixe imposée : le LLM effectue un jugement holistique, ce qui est plus robuste que des formules arithmétiques rigides face à la diversité des questions.

La décision binaire `is_sufficient` est ensuite dérivée automatiquement en code Python en comparant `relevance_score` au seuil paramétrable (voir section 3) — elle n'est **pas** générée par le LLM.

---

## 3. Seuil de Décision (`relevance_score` → `is_sufficient`)

### 3.1 Valeur par défaut proposée

> **Seuil par défaut : `0.70`**

La décision `is_sufficient` est calculée en post-processing Python :

```
is_sufficient = (relevance_score >= seuil)
```

**Justification de la valeur `0.70` :**
- Un score `< 0.50` indique un contexte clairement hors-sujet : le feedback permettra une re-requête ciblée
- Un score entre `0.50` et `0.70` indique un contexte partiel ou ambigu : un deuxième retrieval peut apporter la complétude manquante
- Un score `≥ 0.70` indique un contexte pertinent et suffisamment complet pour générer une réponse raisonnable
- La valeur `0.70` est plus stricte que `0.50` (trop permissif) mais plus souple que `0.85` (trop exigeant pour un modèle local 7B)

### 3.2 Paramètre — pas de valeur hardcodée

Ce seuil doit être **un paramètre configurable**, et non une constante codée en dur dans la logique d'évaluation. La valeur `0.70` est la valeur par défaut recommandée pour l'environnement de développement avec Qwen 2.5 7B en local. En production ou avec un modèle plus puissant, ce seuil devra être recalibré sur le dataset d'évaluation HotpotQA (Sprint 7).

---

## 4. Stratégie du `feedback`

### 4.1 Rôle et destination

Le champ `feedback: str` de `CriticEvaluation` est un message textuel destiné exclusivement au **Planner lors d'une re-décomposition**. Il n'est pas destiné à l'utilisateur final. Lorsque `is_sufficient = True`, ce champ est une chaîne vide (`""`).

Sa fonction est d'informer le Planner de ce qui manque dans le contexte actuel, afin que la `sub_query` reformulée au prochain retrieval soit plus précise et cible spécifiquement les informations absentes.

### 4.2 Format et contenu attendu

Le feedback doit respecter les contraintes suivantes pour être exploitable :

1. **Rédigé en anglais** — le Planner génère des `sub_query` anglophones (conformément au choix fait au Sprint 3)
2. **Actionnable** — il doit permettre de formuler une requête différente et plus ciblée
3. **Factuellement ancré** — il doit nommer les aspects précis manquants plutôt qu'une critique générale
4. **Concis** — une à trois phrases maximum ; une phrase préférable

### 4.3 Exemples comparatifs

#### ✅ Bons feedbacks (actionnables)

| Situation | Feedback |
|---|---|
| La `sub_query` demande la date de fondation d'une entreprise, mais les chunks parlent uniquement de ses produits | `"Retrieved context covers products and services but does not mention the founding date or year of establishment."` |
| La `sub_query` demande le rôle actuel d'une personne, mais les chunks datent de 5 ans | `"Context is outdated (circa 2019). Need a more recent source specifically mentioning the current position of {entity}."` |
| La `sub_query` porte sur deux attributs, un seul est couvert | `"Only the nationality was found. The birth year of {entity} is missing from all retrieved chunks."` |
| Contradiction entre deux chunks sur une date | `"Chunks contradict each other on the release date (2022 vs. 2023). A more authoritative or recent source is needed."` |

#### ❌ Mauvais feedbacks (vagues, non-actionnables)

| Feedback | Problème |
|---|---|
| `"The context is not relevant."` | Trop vague, le Planner ne peut pas identifier ce qu'il faut changer |
| `"Please provide better information."` | Formulation injonctive sans contenu factuel |
| `"This is insufficient."` | Redondant avec `is_sufficient = False`, aucune valeur ajoutée |
| `"I need more context about the topic."` | Trop générique, ne guide pas la reformulation |

### 4.4 Liaison avec `missing_aspects`

Le champ `missing_aspects: list[str]` de `CriticEvaluation` est une version **structurée et atomique** du feedback. Chaque élément est une chaîne courte nommant un aspect précis non couvert. Le `feedback` textuel en est la version rédigée et contextuelle.

**Exemple de cohérence attendue :**
```
missing_aspects = ["founding date", "headquarters location"]
feedback = "Retrieved context covers the product portfolio but is missing the founding date
            and headquarters location of the company."
```

---

## 5. `max_retries` et Garde Anti-Boucle

### 5.1 Définition et valeur par défaut

Le `max_retries` est le nombre maximum de fois que le nœud `retrieve → critique` peut être répété pour une même `PlanStep` avant de forcer la sortie vers `generate_answer`.

> **Valeur par défaut proposée : `max_retries = 2`**

**Justification :**
- Un premier retrieval insuffisant justifie toujours une deuxième tentative avec un feedback plus précis
- Un troisième retrieval échoué sur la même étape indique un problème structurel (question non trouvable dans la base, retriever mal configuré) que le Critic ne peut pas résoudre — continuer est du gaspillage computationnel
- La valeur `2` offre un équilibre entre robustesse et économie de ressources

### 5.2 Articulation avec `reasoning_budget`

Le `reasoning_budget` (défini par l'Analyzer, Sprint 1) et le `max_retries` (paramètre du Critic) sont deux mécanismes **complémentaires mais orthogonaux** dans la garde anti-boucle infinie :

| Mécanisme | Définit par | Contrôle | Compteur dans `AgentState` |
|---|---|---|---|
| `reasoning_budget` | Analyzer (Sprint 1) | Nombre total d'appels au module ACTION par requête | `AgentState.feedback_loop_count` |
| `max_retries` | Critic (Sprint 4) | Nombre de re-retrievals consécutifs pour **une même `PlanStep`** | Local au nœud Critic |

**Règle de priorité :** la garde globale (`feedback_loop_count >= reasoning_budget`) est vérifiée par le **routeur conditionnel du graphe** avant que le Critic ne soit appelé. Si la garde globale est activée, le Critic n'est pas invoqué et le flux passe directement à `generate_answer`.

```
Si AgentState.feedback_loop_count >= AgentState.analysis.reasoning_budget
    → Forcer generate_answer  (garde globale — prioritaire)
Sinon
    → Appeler Critic
        Si retries_pour_cette_step >= max_retries
            → Forcer generate_answer avec contexte partiel  (garde locale)
        Sinon si is_sufficient
            → Avancer à l'étape suivante ou generate_answer
        Sinon
            → Re-déclencher retrieve avec le feedback
```

**Garantie d'arrêt :** avec `reasoning_budget = 3` (MULTI_HOP) et `max_retries = 2`, le cas le plus défavorable est 3 × 2 = 6 appels au module ACTION au maximum par requête. Cette borne est déterministe et finie.

---

## 6. Structure du Prompt Chain-of-Thought

### 6.1 Technique retenue : Chain-of-Thought structuré

Le Critic utilise le **Chain-of-Thought (CoT) Prompting** avec une sortie TOON stricte. Cette approche est privilégiée sur le Few-Shot simple pour deux raisons :

1. **Transparence du raisonnement** : le LLM doit justifier son score étape par étape avant de le produire, ce qui réduit les scores arbitraires et améliore la cohérence entre `relevance_score`, `missing_aspects` et `feedback`
2. **Équilibre précision/latence** : le modèle 7B génère un raisonnement de qualité supérieure sur des tâches d'évaluation qualitative quand il est guidé vers une démarche structurée

### 6.2 Template du Prompt

Ce prompt sera défini dans `src/reasoning/critic/prompts.py`.

```
You are a context quality evaluator for a RAG reasoning engine.

UNIQUE MISSION: Evaluate whether the retrieved context is sufficient to answer
                the given sub-question. Do NOT answer the question.

─── EVALUATION CRITERIA ─────────────────────────────────────────────────────

Evaluate the context on four criteria:
1. RELEVANCE      : Does the context directly address the sub-question?
2. COMPLETENESS   : Does the context cover ALL aspects of the sub-question?
3. CONSISTENCY    : Are the retrieved chunks free of contradictions?
4. FRESHNESS      : Is the information recent enough? (Only relevant for
                    time-sensitive questions. Ignore for timeless facts.)

─── CHAIN-OF-THOUGHT PROCESS ────────────────────────────────────────────────

Before producing your output, reason through the following steps:
  Step 1: State the key information required to answer the sub-question.
  Step 2: Check each retrieved chunk against these requirements.
  Step 3: Identify any missing, contradictory, or outdated information.
  Step 4: Assign a global relevance_score between 0.0 and 1.0.
  Step 5: List the missing aspects as short atomic strings (or leave empty).
  Step 6: Write an actionable feedback sentence if the score is below threshold.

─── OUTPUT FORMAT (STRICT TOON) ─────────────────────────────────────────────

Return ONLY this TOON block. No text before or after.

<<<
relevance_score :: <float between 0.0 and 1.0>
missing_aspects :: <aspect_1 | aspect_2 | ...>    (empty if none)
feedback        :: <one to three sentences in English, empty if sufficient>
>>>

─── INPUTS ──────────────────────────────────────────────────────────────────

Sub-question to evaluate:
  {sub_query}

Retrieved context ({n_chunks} chunks):
{chunks_content}

─── YOUR EVALUATION ─────────────────────────────────────────────────────────

Think step by step, then produce the TOON block.
```

### 6.3 Schéma TOON Attendu en Sortie

Le LLM doit générer un unique bloc TOON contenant exactement trois champs. Aucun autre champ n'est attendu.

| Champ TOON | Type TOON | Contrainte |
|---|---|---|
| `relevance_score` | Flottant (`float`) | Obligatoire, dans `[0.0, 1.0]` |
| `missing_aspects` | Vide ou liste séparée par `\|` | Vide si le contexte est suffisant |
| `feedback` | Chaîne libre | Vide si le contexte est suffisant, sinon 1-3 phrases en anglais |

**Champ absent du TOON :** `is_sufficient` n'est **pas** généré par le LLM. Il est calculé en Python après parsing, par comparaison de `relevance_score` au seuil paramétrable (section 3). Ceci élimine le risque que le LLM retourne un `is_sufficient = True` avec un score de `0.3`.

**Champ renseigné en Python :** `step_id` est connu avant l'appel LLM depuis l'objet `PlanStep.step_id` reçu en entrée — il n'est pas demandé au LLM.

### 6.4 Exemple d'Entrée/Sortie Complet

**Entrée :**
```
Sub-question to evaluate:
  What year was the company that created GPT-4 founded?

Retrieved context (2 chunks):
  [Chunk 1 — openai_products.txt]: OpenAI has developed several large language
  models including GPT-3, GPT-4, and the o1 series. These models are available
  via the OpenAI API for developers.

  [Chunk 2 — ai_landscape_2024.txt]: OpenAI is a leading AI research company.
  GPT-4 was released in March 2023 and represents a significant milestone in
  generative AI capabilities.
```

**Sortie LLM attendue :**
```
<<<
relevance_score :: 0.35
missing_aspects :: founding year of OpenAI
feedback        :: Retrieved chunks confirm that OpenAI created GPT-4 but do not
                   mention the company's founding date or year of establishment.
                   A source specifically covering OpenAI's history or founding is needed.
>>>
```

**Post-processing Python (seuil `0.70`) :**
```
is_sufficient = (0.35 >= 0.70) → False
step_id       = "step_2"        (repris depuis PlanStep.step_id)
```

**Objet `CriticEvaluation` instancié :**
```
CriticEvaluation(
    step_id         = "step_2",
    is_sufficient   = False,
    relevance_score = 0.35,
    missing_aspects = ["founding year of OpenAI"],
    feedback        = "Retrieved chunks confirm that OpenAI created GPT-4 but do not
                       mention the company's founding date or year of establishment.
                       A source specifically covering OpenAI's history or founding is needed."
)
```

---

## 7. Architecture du Composant `critic/`

### 7.1 Fichiers à Créer

```
src/reasoning/critic/
├── __init__.py          # Exporte Critic
├── critic.py            # Classe Critic (logique principale)
└── prompts.py           # EVALUATION_PROMPT (template CoT)
```

### 7.2 Interface Publique de `Critic`

```python
# Pseudo-code de l'interface — pas le code final

class Critic:
    def __init__(
        self,
        model: str = DEFAULT_REASONING_MODEL,   # Qwen 2.5 7B
        api_base: str = OLLAMA_BASE_URL,
        temperature: float = 0.0,
        sufficiency_threshold: float = 0.70,    # Paramétrable, pas hardcodé
        max_retries: int = 2,                   # Paramétrable, pas hardcodé
    ) -> None: ...

    def evaluate(
        self,
        step: PlanStep,
        response: RetrievalResponse,
    ) -> CriticEvaluation: ...
```

### 7.3 Flux d'Exécution Détaillé

```
evaluate(step, response)
        │
        ├── Formater le prompt CoT avec step.sub_query + chunks formatés
        │
        ├── Appel LiteLLM → Qwen 7B (temperature=0)
        │
        ▼ Réponse TOON brute
        parse_toon_to_dict()    (shared/toon_utils.py)
        │
        ├── [Succès]
        │       ├── Extraire relevance_score, missing_aspects, feedback
        │       ├── Calculer is_sufficient = (relevance_score >= threshold)
        │       └── Instancier CriticEvaluation(step_id=step.step_id, ...)
        │
        └── [ToonParseError / ValidationError / Timeout]
                │
                ▼
        Fallback défensif :
        CriticEvaluation(
            step_id=step.step_id,
            is_sufficient=False,
            relevance_score=0.0,
            missing_aspects=["parsing_failed"],
            feedback="LLM evaluation failed — forcing re-retrieval."
        )
        + log WARNING
```

### 7.4 Modèle LLM Utilisé

Le Critic utilise **Qwen 2.5 7B** (`ollama/qwen2.5:7b`), le même modèle que le Planner, configuré via `DEFAULT_REASONING_MODEL`. L'évaluation de qualité contextuelle est une tâche de compréhension fine qui nécessite un modèle plus capable que le 3B réservé au Query Analyzer.

| Paramètre | Valeur |
|---|---|
| Variable d'environnement | `DEFAULT_REASONING_MODEL` |
| Valeur par défaut | `ollama/qwen2.5:7b` |
| Température | `0.0` (déterministe) |
| `max_tokens` | `256` (le bloc TOON de sortie est court : 3 lignes) |

---

## 8. Gestion des Erreurs

| Scénario | Comportement |
|---|---|
| LLM retourne un bloc TOON valide | Parsing via `parse_toon_to_dict()`, instanciation `CriticEvaluation` |
| `relevance_score` hors `[0.0, 1.0]` | `ValidationError` Pydantic interceptée → fallback défensif |
| Bloc TOON malformé ou absent | `ToonParseError` interceptée → fallback défensif (score=0.0, is_sufficient=False) |
| Timeout Ollama / erreur réseau | Exception LiteLLM interceptée → fallback défensif + log WARNING |
| `RetrievalResponse.chunks` est vide | Cas spécial : `relevance_score=0.0`, `is_sufficient=False`, feedback = `"No chunks were retrieved for this step."` sans appel LLM |

---

## 9. Métriques de Qualité

Les métriques suivantes seront mesurées lors du Sprint 7 (évaluation RAGAS) pour valider la performance du Critic :

| Métrique | Cible | Méthode de mesure |
|---|---|---|
| **Précision de décision** (`is_sufficient`) | ≥ 80% | Comparaison avec labels ground-truth sur dataset HotpotQA |
| **Taux de faux positifs** (`is_sufficient=True` alors que contexte insuffisant) | ≤ 10% | Mesure sur le sous-ensemble de questions MULTI_HOP à 3 étapes |
| **Taux de boucles forcées** (garde `max_retries` déclenchée) | ≤ 15% | `count(retries_reached_max) / total_steps_evaluated` |
| **Latence P95** | ≤ 4s | Mesure sur le modèle Qwen 7B local |
| **Taux de fallback** (ToonParseError) | ≤ 5% | `count(parsing_failed) / total_evaluations` |

---

## 10. Plan de Vérification

### Tests Unitaires (`tests/unit/test_critic.py`)

| Test | Mock | Assertion clé |
|---|---|---|
| `test_sufficient_context` | TOON avec `relevance_score=0.85` | `is_sufficient=True`, `feedback==""` |
| `test_insufficient_context` | TOON avec `relevance_score=0.35` | `is_sufficient=False`, `feedback != ""` |
| `test_partial_context_at_threshold` | TOON avec `relevance_score=0.70` | `is_sufficient=True` (score ≥ seuil) |
| `test_missing_aspects_populated` | TOON avec `missing_aspects` non vide | `len(evaluation.missing_aspects) > 0` |
| `test_step_id_propagated` | `PlanStep(step_id="step_2", ...)` | `evaluation.step_id == "step_2"` |
| `test_empty_chunks_no_llm_call` | `RetrievalResponse(chunks=[])` | `completion` non appelé, `is_sufficient=False` |
| `test_fallback_on_toon_parse_error` | `side_effect=ToonParseError` | `isinstance(result, CriticEvaluation)`, `relevance_score=0.0` |
| `test_fallback_on_timeout` | `side_effect=TimeoutError` | `is_sufficient=False`, aucun crash |
| `test_custom_threshold` | `Critic(sufficiency_threshold=0.85)` | `is_sufficient=False` pour score=0.80 |
| `test_max_retries_parameter` | `Critic(max_retries=1)` | Attribut accessible et paramétrable |

### Tests d'Intégration (`tests/integration/test_critic_live.py`)

Protégés par `@pytest.mark.integration`. Nécessitent Ollama + Qwen 7B.

Cible : vérifier que pour 5 paires (`PlanStep`, `RetrievalResponse`) de référence, le Critic produit un `CriticEvaluation` valide avec un `relevance_score` dans `[0.0, 1.0]` et un `feedback` non vide quand `is_sufficient=False`.

---

## 11. Questions Ouvertes

Les points suivants dépassent le périmètre de la tâche 4.1 et nécessitent une décision à une étape ultérieure avant ou pendant l'implémentation (4.2) :

1. **Format des chunks dans le prompt** : Le template de prompt (section 6.2) utilise `{chunks_content}` comme placeholder. La mise en forme exacte des chunks (numéroté, avec source ? avec score de pertinence ? longueur maximale par chunk ?) n'est pas spécifiée ici. Elle doit être définie lors de 4.2 en tenant compte des limites de contexte du modèle 7B.

2. **Persistance du compteur `max_retries` par étape** : Le `max_retries` est un compteur **local par `PlanStep`**, distinct de `AgentState.feedback_loop_count` (qui est global). Ce compteur local n'a pas de champ dédié dans `AgentState` tel que défini dans `internal_models.py`. La question est : faut-il ajouter un champ `retry_counts: dict[str, int]` à `AgentState`, ou gérer ce compteur différemment (ex: dans une variable d'état temporaire du nœud LangGraph) ? **Cette décision implique une modification du contrat `AgentState` et doit faire l'objet d'une revue d'architecture avant 4.2.**

3. **Cas AMBIGUOUS** : Si le flux parvient au Critic avec un `PlanStep` issu d'une requête `AMBIGUOUS` (cas normalement impossible car `reasoning_budget=0` coupe le flux avant), le comportement n'est pas spécifié. Un comportement défensif explicite devra être documenté lors de 4.2.

4. **Localisation du seuil** : La section 3.2 indique que `sufficiency_threshold` doit être paramétrable mais ne tranche pas entre un paramètre constructeur, une constante dans `.env`, ou un fichier `configs/thresholds.toml`. Ce choix est délégué au Sprint 4.2 / Sprint 8.

---

*Spécification RAG-REASON — Critic v1.0 — Sprint 4*
*À lire en conjonction avec `docs/planner_spec.md` et `src/reasoning/contracts/internal_models.py`*
