# Spécification Technique — Query Analyzer
## Sprint 2.1 · Module REASONING · RAG-REASON

> **Statut :** Validé — Ne pas modifier sans revue d'architecture
> **Auteur :** Équipe REASONING
> **Version :** 1.0
> **Contrat Pydantic de référence :** `AnalysisResult` (`src/reasoning/contracts/internal_models.py`)

---

## 1. Rôle et Responsabilités

Le **Query Analyzer** est le point d'entrée du graphe LangGraph. Il est le premier nœud exécuté pour chaque requête utilisateur. Sa mission est double :

1. **Classifier** la requête entrante selon une taxonomie sémantique prédéfinie (`QueryType`)
2. **Allouer le budget computationnel** (`reasoning_budget`) pour contraindre le nombre maximum d'étapes de raisonnement dans le graphe

Le Query Analyzer est le **gardien de l'efficacité** du système : il empêche d'engager des ressources LLM coûteuses sur des requêtes simples, et garantit que les requêtes complexes reçoivent le traitement qu'elles méritent.

**Ce composant ne fait pas de retrieval.** Il ne communique pas avec le module ACTION. Il consomme uniquement la `original_query` de l'`AgentState` et enrichit l'état avec un objet `AnalysisResult`.

---

## 2. Règles de Classification

### 2.1 Taxonomie des types de requêtes (`QueryType`)

La classification repose sur quatre catégories mutuellement exclusives. La priorité de détection est hiérarchique : `AMBIGUOUS` > `MULTI_HOP` > `COMPARATIVE` > `SIMPLE`.

---

#### `SIMPLE`
**Définition :** Requête dont la réponse peut être trouvée dans un seul document ou chunk, sans nécessiter de chaîner plusieurs faits ou de résoudre une ambiguïté. La réponse est directe et ne dépend pas d'informations intermédiaires.

**Critères de détection :**
- Un seul sujet, un seul prédicat, une seule entité cible
- Pas de connecteurs logiques de séquence ou de comparaison
- La réponse attendue est factuelle et atomique

**Patterns linguistiques caractéristiques :**
| Pattern | Exemple |
|---|---|
| "Qu'est-ce que X ?" | "Qu'est-ce que la photosynthèse ?" |
| "Qui est X ?" | "Qui est Ada Lovelace ?" |
| "Quelle est la définition de X ?" | "Quelle est la définition du RAG ?" |
| "Quand a eu lieu X ?" | "Quand a eu lieu la Révolution française ?" |
| "Où se trouve X ?" | "Où se trouve le Mont-Blanc ?" |
| Question directe sans subordination | "Quel est le rôle d'un Transformer ?" |

**`reasoning_budget` alloué : `1`**

---

#### `MULTI_HOP`
**Définition :** Requête nécessitant la résolution de plusieurs sous-questions intermédiaires et indépendantes dont les réponses doivent être chaînées pour produire la réponse finale. La réponse à chaque sous-question devient un input pour la suivante.

**Critères de détection :**
- Présence de connecteurs de séquence ou de causalité
- Plusieurs entités dont les relations doivent être inférées
- La réponse finale dépend de la résolution préalable d'au moins une question intermédiaire

**Patterns linguistiques caractéristiques :**
| Pattern | Exemple |
|---|---|
| "X, et ensuite Y ?" | "Qui a fondé OpenAI, et ensuite quel poste occupe-t-il aujourd'hui ?" |
| "Qui a [verbe] [entité], et quel est [attribut] ?" | "Qui a inventé le transformeur, et dans quelle entreprise travaille-t-il ?" |
| "Quel est [A] de [B] qui a [C] ?" | "Quel est le PDG de l'entreprise qui a créé GPT-4 ?" |
| "Depuis que X, comment Y ?" | "Depuis que l'attention a été introduite, comment a évolué la taille des modèles ?" |
| Présence de pronoms référentiels ("il", "elle", "ce dernier") | "LangChain a lancé LangGraph. Comment ce dernier gère-t-il les états ?" |
| Questions imbriquées avec "dont", "lequel", "duquel" | "Cite l'auteur de l'article dont le modèle a obtenu le meilleur score BLEU." |

**`reasoning_budget` alloué : `3`**

---

#### `AMBIGUOUS`
**Définition :** Requête dont l'intention ou le référent principal est indéterminable sans contexte supplémentaire. Le terme clé possède plusieurs significations plausibles, ou la portée de la question est trop large pour permettre une réponse précise.

**Critères de détection :**
- Terme polysémique sans contexte désambiguïsant (ex: "banque", "Python", "réseau")
- Absence de sujet clairement identifiable
- Question trop générique dont la réponse exhaustive serait impossible
- Requête nécessitant une clarification de l'utilisateur avant tout traitement

**Patterns linguistiques caractéristiques :**
| Pattern | Exemple |
|---|---|
| Terme polysémique isolé | "Comment fonctionne Python ?" (langage ? serpent ?) |
| Question trop large | "Explique-moi l'IA." |
| Référent ambigu | "Quel est son rôle ?" (sans antécédent) |
| Négation implicite floue | "Ce n'est pas ce que je voulais dire par apprentissage." |
| Acronyme non défini dans le contexte | "Quel est le taux de FPR optimal ?" |

**`reasoning_budget` alloué : `0`**

> ⚠️ **Cas particulier :** Un `reasoning_budget` de `0` signale au graphe LangGraph qu'aucun retrieval ne doit être lancé. Le nœud de routage conditionnel redirigera vers une réponse de clarification ou une demande de reformulation à l'utilisateur.

---

#### `COMPARATIVE`
**Définition :** Requête demandant explicitement une mise en parallèle de deux ou plusieurs entités, concepts, méthodes ou périodes selon un ou plusieurs critères d'évaluation.

**Critères de détection :**
- Verbes comparatifs : "compare", "distingue", "différencie", "contraste"
- Conjonctions d'opposition : "versus", "vs.", "par rapport à", "comparé à", "plutôt que"
- Structures duales explicites : "X et Y : quelles différences ?"
- Questions de classement relatif : "lequel est meilleur / plus rapide / plus efficace ?"

**Patterns linguistiques caractéristiques :**
| Pattern | Exemple |
|---|---|
| "Compare X et Y" | "Compare BERT et GPT-4." |
| "Quelle est la différence entre X et Y ?" | "Quelle est la différence entre RAG et Fine-Tuning ?" |
| "X versus Y" | "LangGraph versus AutoGen : avantages et inconvénients ?" |
| "Quel est le plus [adjectif] : X ou Y ?" | "Quel est le plus efficace : BM25 ou un retriever dense ?" |
| "Avantages et inconvénients de X par rapport à Y" | "Avantages de la quantification GGUF par rapport au GPTQ." |

**`reasoning_budget` alloué : `2`**

---

### 2.2 Tableau récapitulatif

| `QueryType` | `reasoning_budget` | Retrieval ? | Description courte |
|---|---|---|---|
| `SIMPLE` | `1` | ✅ 1 appel | Fact unique, pas de chaîne |
| `MULTI_HOP` | `3` | ✅ N appels | Chaîne de sous-questions |
| `COMPARATIVE` | `2` | ✅ 2 appels (un par entité) | Mise en parallèle |
| `AMBIGUOUS` | `0` | ❌ Aucun | Clarification requise |

---

## 3. Définition du `reasoning_budget`

### 3.1 Définition formelle

Le `reasoning_budget` est un entier positif ou nul (`int ≥ 0`) qui représente le **nombre maximum d'appels au module ACTION** (de retrieval) que le graphe LangGraph est autorisé à effectuer pour une requête donnée.

Il est matérialisé dans le contrat Pydantic :
```
AnalysisResult.reasoning_budget: int = Field(gt=0)
```
> **Note :** La contrainte `gt=0` du contrat est levée pour `AMBIGUOUS` (budget = 0) via une logique de pre-validation dans l'Analyzer. Le champ `AgentState.feedback_loop_count` est le compteur d'exécution réel ; `reasoning_budget` est le plafond théorique.

### 3.2 Rôle dans l'architecture anti-boucle infinie

Le `reasoning_budget` est la **première ligne de défense** contre les boucles infinies dans le graphe LangGraph. La logique de sortie dans l'arête conditionnelle post-Critic est :

```
Si feedback_loop_count >= reasoning_budget → forcer la sortie vers generate_answer
Sinon → autoriser une nouvelle itération de retrieval
```

Sans ce mécanisme, un Critic systématiquement insatisfait pourrait déclencher une boucle de retrieval infinie, épuisant les ressources computationnelles.

### 3.3 Valeurs par type et justification

| `QueryType` | `reasoning_budget` | Justification |
|---|---|---|
| `SIMPLE` | **1** | Une seule recherche suffit. Tout re-try signale un échec du retriever, pas de la question. |
| `MULTI_HOP` | **3** | Conservatif : la plupart des questions multi-sauts se résolvent en 2-3 étapes. Valeur modifiable via `.env`. |
| `COMPARATIVE` | **2** | Un retrieval par entité à comparer. Un 3ème retrieval de synthèse peut être justifié mais reste exceptionnel. |
| `AMBIGUOUS` | **0** | Aucun retrieval autorisé. La requête est redirigée vers une demande de clarification. |

---

## 4. Stratégie de Prompt Engineering

### 4.1 Technique retenue : Few-Shot Prompting structuré

Le Query Analyzer utilise exclusivement la technique du **Few-Shot Prompting** avec des exemples représentatifs de chaque type de requête. Cette approche est privilégiée sur le Zero-Shot pour trois raisons :

1. **Réduction de l'ambiguïté** : les exemples ancrent le LLM dans notre taxonomie spécifique, évitant les interprétations libres
2. **Stabilité du format de sortie** : les exemples montrent le JSON attendu, réduisant les erreurs de parsing
3. **Coût computationnel contrôlé** : le prompt est compact et reproductible, sans chaîne de pensée superflue

### 4.2 Structure du prompt

Le template de prompt suit une structure en 4 blocs immuables :

```
[BLOC 1 — Rôle et contraintes]
Tu es un classificateur de requêtes pour un moteur de raisonnement RAG.
Ta seule tâche est de classer la requête entrante et de retourner un JSON valide.
Tu ne dois JAMAIS répondre à la question. Tu dois UNIQUEMENT la classifier.

[BLOC 2 — Taxonomie]
Voici les types possibles et leurs critères :
- SIMPLE : ...
- MULTI_HOP : ...
- COMPARATIVE : ...
- AMBIGUOUS : ...

[BLOC 3 — Exemples Few-Shot]
Exemple 1 :
  Requête : "Qu'est-ce que la rétropropagation ?"
  JSON : {"query_type": "SIMPLE", "confidence": 0.97, "detected_entities": ["rétropropagation"], "reasoning_budget": 1}

Exemple 2 :
  Requête : "Qui a créé LangChain, et quelle est sa relation avec LangSmith ?"
  JSON : {"query_type": "MULTI_HOP", "confidence": 0.91, "detected_entities": ["LangChain", "LangSmith"], "reasoning_budget": 3}

Exemple 3 :
  Requête : "Compare le fine-tuning et le RAG pour une application médicale."
  JSON : {"query_type": "COMPARATIVE", "confidence": 0.88, "detected_entities": ["fine-tuning", "RAG"], "reasoning_budget": 2}

Exemple 4 :
  Requête : "Explique-moi les réseaux."
  JSON : {"query_type": "AMBIGUOUS", "confidence": 0.79, "detected_entities": ["réseaux"], "reasoning_budget": 0}

[BLOC 4 — Requête utilisateur]
Requête à classifier : "{query}"
Retourne UNIQUEMENT le JSON. Aucun texte avant ou après.
```

### 4.3 Contrat de sortie JSON (liaison avec Pydantic)

La sortie attendue du LLM doit être **directement parseable** par `AnalysisResult.model_validate_json()`. Le LLM doit produire exactement :

```json
{
  "query_type": "SIMPLE | MULTI_HOP | COMPARATIVE | AMBIGUOUS",
  "confidence": 0.0,
  "detected_entities": ["entité_1", "entité_2"],
  "reasoning_budget": 1
}
```

**Règles de parsing :**
- Si le LLM entoure le JSON d'un bloc Markdown (` ```json ... ``` `), le parser doit l'extraire via regex avant validation
- Si le JSON est invalide ou si les champs sont absents → déclencher le **fallback heuristique** (section 5.2)
- Si `query_type` n'est pas un membre valide de `QueryType` → déclencher le fallback heuristique

---

## 5. Stratégie d'Implémentation

### 5.1 Décision d'architecture validée : Approche Hybride

**La stratégie d'implémentation retenue est une approche hybride à deux niveaux :**

```
Niveau 1 (Principal)  : Moteur LLM via LiteLLM → Ollama (modèle DEFAULT_FAST_MODEL)
                                    │
                          [Succès : JSON valide ?]
                         /                        \
                       OUI                        NON
                        │                          │
              Retourner AnalysisResult    Niveau 2 (Fallback)
                                                   │
                                        Moteur de règles heuristiques Python
                                                   │
                                        Retourner AnalysisResult
```

**Justification du choix hybride :**
- Le LLM seul peut échouer : timeout Ollama, réponse mal formatée, modèle surchargé
- Les règles seules sont trop rigides : sensibles à la langue, aux fautes d'orthographe, aux formulations inédites
- L'hybride garantit un **taux de disponibilité de 100%** du composant, avec une dégradation gracieuse de la précision

### 5.2 Moteur de règles heuristiques Python (Fallback)

Le fallback est un module Python pur, sans dépendance LLM, basé sur la détection de patterns dans la requête normalisée.

#### Règles de détection (par ordre de priorité)

**Détection AMBIGUOUS (priorité 1) :**
```
SI longueur_tokens(query) < 4 → AMBIGUOUS
SI aucune entité nommée détectable → AMBIGUOUS
SI query contient un terme polysémique de la liste noire
   ET absence de contexte désambiguïsant → AMBIGUOUS
```

Termes polysémiques de la liste noire (exemples) :
`["python", "réseau", "banque", "agent", "modèle", "cloud", "java", "framework"]`

**Détection MULTI_HOP (priorité 2) :**
```
SI query contient l'un des marqueurs suivants → MULTI_HOP
```
Marqueurs linguistiques :
```python
MULTI_HOP_MARKERS = [
    "et ensuite", "puis", "après quoi", "à la suite de",
    "qui a ensuite", "dont le", "duquel", "de laquelle",
    "ce dernier", "cette dernière", "lequel", "laquelle",
    "quel est le [X] de [Y] qui", "depuis que",
]
```

**Détection COMPARATIVE (priorité 3) :**
```
SI query contient l'un des marqueurs suivants → COMPARATIVE
```
Marqueurs linguistiques :
```python
COMPARATIVE_MARKERS = [
    "compare", "versus", " vs ", "vs.", "différence entre",
    "par rapport à", "comparé à", "plutôt que", "avantages et inconvénients",
    "lequel est meilleur", "lequel est plus",
    "quelle est la distinction", "distingue",
]
```

**Détection SIMPLE (priorité 4, par défaut) :**
```
SI aucun marqueur MULTI_HOP ou COMPARATIVE détecté
   ET requête non AMBIGUOUS
→ SIMPLE
```

#### Valeurs par défaut du fallback

Quand le fallback s'applique, la `confidence` est systématiquement réduite à `0.55` pour signaler l'incertitude de la classification heuristique. Le `reasoning_budget` suit la table de la section 2.2.

```python
FALLBACK_CONFIDENCE = 0.55  # Signal de dégradation gracieuse
```

### 5.3 Modèle LLM utilisé

| Paramètre | Valeur |
|---|---|
| Variable d'environnement | `DEFAULT_FAST_MODEL` |
| Valeur par défaut | `ollama/qwen2.5:3b` |
| Justification | Le modèle 3B est suffisant pour la classification. Économise le 7B pour les tâches de génération (Planner, Critic, Verifier). |
| Température | `0.0` (déterministe — sortie JSON stable) |
| `max_tokens` | `256` (une réponse JSON ne nécessite pas plus) |

---

## 6. Interface du Composant

### 6.1 Entrée

```
original_query: str  ← lu depuis AgentState.original_query
```

### 6.2 Sortie

```
AnalysisResult
├── query_type: QueryType
├── confidence: float          # [0.0, 1.0]
├── detected_entities: list[str]
└── reasoning_budget: int      # ≥ 0
```

### 6.3 Effets de bord sur l'AgentState

```
AVANT  : AgentState.analysis = None
APRÈS  : AgentState.analysis = AnalysisResult(...)
```

Le nœud `analyze_query` du graphe LangGraph retournera un dictionnaire partiel :
```python
{"analysis": AnalysisResult(...)}
```
LangGraph fusionnera cet update avec l'état existant.

---

## 7. Gestion des Erreurs

| Scénario | Comportement |
|---|---|
| LLM renvoie un JSON valide | Utiliser `AnalysisResult.model_validate_json()` |
| LLM renvoie un JSON dans un bloc Markdown | Extraire avec regex `r'\{.*?\}'` (flags=DOTALL), puis valider |
| LLM renvoie un texte non-JSON | Déclencher le fallback heuristique |
| Timeout Ollama / erreur réseau | Déclencher le fallback heuristique + logger WARNING |
| Fallback heuristique échoue (cas extrême) | Retourner `QueryType.SIMPLE` avec `confidence=0.3` comme valeur de sécurité absolue |

---

## 8. Métriques de Qualité

Les métriques suivantes seront mesurées lors du Sprint 7 (évaluation RAGAS) pour valider la performance du Query Analyzer :

| Métrique | Cible | Méthode de mesure |
|---|---|---|
| **Précision de classification** | ≥ 85% | Comparaison avec labels ground-truth sur dataset d'évaluation |
| **Taux de fallback** | ≤ 5% | `count(confidence == 0.55) / total_queries` |
| **Latence P95** | ≤ 2s | Mesure du temps de réponse sur le modèle 3B local |
| **Taux d'erreur** | ≤ 1% | `count(exceptions) / total_queries` |
