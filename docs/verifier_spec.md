# Spécification Technique — Sprint 5 : Le Verifier (Groundedness Check)
## Projet RAG-REASON

| | |
|---|---|
| **Document** | `docs/verifier_spec.md` |
| **Version** | 1.0 |
| **Sprint** | 5 — Verifier |
| **Auteur** | Mohammed Solimani |
| **Date** | Août 2026 |
| **Statut** | Spécification validée — Prête pour implémentation (5.2) |
| **Précédent** | `docs/critic_spec.md` — Critic (Sprint 4, validé) |

---

## 1. Rôle et Position dans le Graphe

Le **Verifier** est le cinquième nœud du graphe de raisonnement. Il est exécuté **après la génération de la réponse candidate** (nœud `generate_answer`), avant que la réponse ne soit présentée à l'utilisateur. Son rôle est de détecter les hallucinations en vérifiant que chaque affirmation factuelle de la réponse est traçable dans les sources récupérées par le module ACTION.

Il reçoit en entrée :
- `answer: str` — la réponse candidate générée par le nœud `generate_answer`
- `sources: list[RetrievedChunk]` — l'ensemble des chunks récupérés au cours de tous les retrievals du plan d'exécution (tous les hops confondus)

Sa mission est de produire en sortie un objet `VerificationResult` : un verdict de fidélité indiquant si la réponse est entièrement fondée sur les sources, le score de fidélité calculé, et la liste des affirmations non traçables.

```
                    ┌─────────────────┐
 answer: str        │                 │  VerificationResult
 ─────────────────► │   VERIFIER      │ ──────────────────►  Routeur conditionnel
 sources:           │                 │                        │
 list[RetrievedChunk│                 │                   ├── is_grounded = True
                    └─────────────────┘                   │      └──► retourner final_answer
                          │  Appel Qwen 2.5 7B            └── is_grounded = False
                          ▼                                       └──► re-génération ou
                    Réponse TOON brute                                  avertissement (Sprint 6)
```

### Invariants

1. Le Verifier ne **modifie jamais** le texte de la réponse — `final_answer` est toujours identique à `answer` reçu en entrée
2. Il ne fait **jamais** d'appel direct au module ACTION
3. Il est **stateless** : chaque appel à `verify()` est indépendant
4. Il ne **génère** pas de réponse alternative — il juge uniquement
5. La décision de re-générer ou d'insérer un avertissement dans la réponse présentée à l'utilisateur est prise par le **routeur conditionnel du graphe LangGraph**, pas par le Verifier lui-même

---

## 2. Définition de "Groundedness"

### 2.1 Critère formel

Une réponse est dite **"grounded"** (fondée sur les sources) si et seulement si **chaque affirmation factuelle** qu'elle contient est **traçable** à au moins un chunk de la liste `sources` fournie en entrée.

On appelle **"claim"** (affirmation) toute proposition factuelle de la réponse susceptible d'être vraie ou fausse : une date, un nom propre, une relation causale, un chiffre, une attribution d'un fait à une entité. Les transitions rhétoriques, introductions et formules de politesse ne constituent pas des claims.

**Exemples de claims :**
- `"OpenAI was founded in December 2015"` — claim factuel sur une date
- `"GPT-4 is a large multimodal model"` — claim factuel sur une propriété
- `"Sam Altman is the CEO of OpenAI"` — claim factuel sur une relation

**Exemples de non-claims :**
- `"Here is a summary of the available information."` — formule rhétorique
- `"Based on the provided context..."` — transition narrative
- `"This is a complex topic."` — opinion générale non vérifiable

### 2.2 Condition de traçabilité

Un claim est **supporté** si un chunk de `sources` contient une information explicite ou fortement implicite corroborant ce claim. Le Verifier n'exige pas une citation mot-à-mot, mais une correspondance sémantique suffisante.

Un claim est **non supporté** si :
- Aucun chunk ne mentionne l'information
- Les chunks contredisent explicitement le claim
- Le claim est une inférence au-delà de ce que les sources permettent de déduire raisonnablement

---

## 3. Choix d'Architecture — Méthode d'Alignement Claim↔Source

### 3.1 Deux approches évaluées

Avant de définir l'architecture retenue, deux approches ont été analysées pour le problème d'alignement claim↔source :

#### Approche A : LLM-as-a-judge (retenue)

Le LLM (Qwen 2.5 7B via LiteLLM/Ollama) reçoit dans un seul prompt la réponse candidate et l'ensemble des sources, et produit un bloc TOON structuré listant chaque claim avec son verdict et le chunk source associé.

**Arguments en faveur :**
- **Cohérence architecturale** : même stack LiteLLM + Ollama + TOON que l'Analyzer, le Planner et le Critic — aucune nouvelle dépendance, aucun nouveau service à déployer
- **Robustesse sémantique** : capable de valider des claims impliquant un raisonnement implicite ou une reformulation (ex : "the company was founded in the mid-2010s" validé par un chunk qui dit "founded in December 2015")
- **Coût de développement** : la structure de classe est identique au Critic — 1.5 jours estimés (contre 3+ pour une pipeline NLI)
- **Latence acceptable** : un seul appel LLM en mode déterministe (`temperature=0`), compatible avec la cible P95 ≤ 6s

**Contraintes :**
- Dépend de la qualité du modèle local 7B ; peut varier selon la version d'Ollama
- Limite de contexte : si `answer` est très long et `sources` nombreuses, le prompt peut dépasser la fenêtre de contexte — mitigation via troncature des chunks (voir section 8)

#### Approche B : NLI léger (écartée pour cette itération)

Un modèle de classification pré-entraîné dédié à l'inférence de langage naturel (type DeBERTa-large ou RoBERTa-large entraîné sur MNLI/SNLI) serait utilisé pour scorer chaque paire (claim, chunk source) avec une probabilité d'entailment/contradiction/neutral.

> **Clarification importante** : l'approche NLI désigne ici un modèle de classification à tête softmax entraîné spécifiquement sur des datasets d'inférence (MNLI, SNLI, MultiNLI) — pas une heuristique lexicale type BM25 ou comparaison de n-grammes. La sortie est un vecteur de probabilités sur trois classes (entailment, contradiction, neutral).

**Arguments en faveur :**
- Latence très faible (inférence CPU ou GPU léger, < 100ms par paire)
- Déterministe et reproductible indépendamment d'Ollama

**Raisons de l'écarter pour cette itération :**
- **Nouvelle dépendance lourde** : ajout de `transformers` + `torch` dans `pyproject.toml`, téléchargement d'un modèle HuggingFace (400MB–1GB), gestion du runtime Python séparé
- **Robustesse moindre sur du raisonnement complexe** : les claims multi-hop ou implicites sont systématiquement classés "neutral" par un NLI 3-classes, là où un LLM avec contexte complet les valide correctement
- **Granularité insuffisante** : le modèle NLI compare une paire (claim, chunk) isolée, sans accès au contexte global de la réponse

**Piste d'optimisation future documentée explicitement :** architecture en **cascade** — le modèle NLI sert de pré-filtre rapide (faible latence, basse précision) pour identifier les claims trivialement supportés (entailment fort > 0.95) et les exclure du prompt LLM, réservant l'appel LLM aux seuls claims ambigus (neutral ou entailment faible). Cette architecture n'est pas implémentée en Sprint 5 mais documentée comme cible de Sprint 8 / optimisation de production.

### 3.2 Décision

> **Approche retenue : LLM-as-a-judge (Approche A).**

La cohérence avec l'architecture existante, l'absence de nouvelle dépendance, et la robustesse sémantique sur les cas complexes (multi-hop, reformulations) justifient ce choix pour l'itération Sprint 5. L'approche NLI en cascade est documentée comme piste d'optimisation explicite pour une itération future.

---

## 4. Stratégie de Décomposition — Un Seul Appel LLM

### 4.1 Principe

Le Verifier effectue la **décomposition en claims ET leur vérification dans un seul appel LLM structuré**. Le LLM identifie lui-même les claims contenus dans `answer`, les confronte aux chunks `sources`, et produit un bloc TOON unique listant le verdict de chaque claim.

**Justification :** une architecture en deux appels séparés (appel 1 = extraire les claims, appel 2 = vérifier chaque claim) doublerait la latence et le coût par invocation. Le modèle 7B est capable de réaliser les deux tâches en un seul raisonnement structuré, comme démontré par le pattern Chain-of-Thought du Critic (Sprint 4). La maîtrise de la latence est une contrainte opérationnelle du projet (cible P95 ≤ 6s, section 9).

### 4.2 Structure de la Sortie TOON

Le LLM produit un bloc TOON multi-claims. Chaque claim est représenté par un groupe de trois champs préfixés par son indice :

```
<<<
total_claims    :: <entier strictement positif>
claim_1_text    :: <texte du claim extrait de answer>
claim_1_verdict :: supported | unsupported
claim_1_source  :: <chunk_id du chunk source> | none
claim_2_text    :: <texte du claim>
claim_2_verdict :: supported | unsupported
claim_2_source  :: <chunk_id> | none
...
>>>
```

Ce format est entièrement compatible avec `parse_toon_to_dict()` de `toon_utils.py` : chaque clé est unique et le séparateur `|` est utilisé pour les listes uniquement sur `claim_N_source` quand plusieurs chunks supportent un claim.

---

## 5. Calcul du `faithfulness_score`

### 5.1 Formule

```
faithfulness_score = claims_supported / total_claims
```

où :
- `claims_supported` = nombre de claims dont `verdict == "supported"`
- `total_claims` = nombre total de claims identifiés par le LLM

### 5.2 Cas limite : `total_claims = 0`

Si la réponse candidate ne contient **aucun claim vérifiable** (formule purement rhétorique, réponse vide, ou LLM qui n'identifie aucune affirmation factuelle), la division par zéro est interdite.

**Comportement défini :**

> Si `total_claims == 0` : `faithfulness_score = 1.0`, `is_grounded = True`, `unsupported_claims = []`.

**Justification :** une réponse sans affirmation factuelle ne peut pas être factuellement incorrecte. Ce comportement est cohérent avec la définition de Groundedness (section 2.1) : s'il n'y a aucun claim à vérifier, la condition de traçabilité est trivialement satisfaite. L'orchestrateur LangGraph peut néanmoins logguer ce cas comme une réponse de qualité insuffisante (sans affirmations concrètes).

### 5.3 Cas limite : `sources = []` (sources vides)

Si la liste `sources` est vide en entrée, le Verifier **ne fait aucun appel LLM** et retourne immédiatement un verdict défensif :

```
is_grounded       = False
faithfulness_score = 0.0
unsupported_claims = ["no sources available for verification"]
final_answer      = answer   (identique à l'entrée, jamais modifié)
```

**Justification :** sans sources, aucun claim ne peut être vérifié — forcer `is_grounded=False` sans appel LLM coûteux est le comportement défensif correct, identique au traitement de `chunks=[]` par le Critic (Sprint 4).

---

## 6. Seuil de Décision (`faithfulness_score` → `is_grounded`)

### 6.1 Valeur par défaut

> **Seuil par défaut : `0.80`**

La décision `is_grounded` est calculée en post-processing Python :

```
is_grounded = (faithfulness_score >= seuil)
```

**Justification de la valeur `0.80` :**
- Un score `< 0.50` indique que la majorité des affirmations n'est pas traçable — hallucination significative
- Un score entre `0.50` et `0.80` indique une réponse partiellement fondée : certains claims sont vérifiés, d'autres sont inventés ou sur-généralisés
- Un score `≥ 0.80` signifie que 4 affirmations sur 5 au minimum sont traçables — standard minimal acceptable pour présenter une réponse à l'utilisateur
- La valeur `0.80` est plus stricte que le seuil du Critic (`0.70`) car la vérification de fidélité porte sur la réponse finale (qualité visible par l'utilisateur), contrairement à l'évaluation du contexte intermédiaire (qualité interne)

### 6.2 Paramètre — pas de valeur hardcodée

Ce seuil doit être un **paramètre configurable** du constructeur de `Verifier`, non une constante codée en dur. La valeur `0.80` est la valeur par défaut recommandée pour l'environnement de développement avec Qwen 2.5 7B. En production ou avec un modèle plus puissant, ce seuil sera recalibré sur le dataset d'évaluation (Sprint 7). Le mécanisme est identique à `sufficiency_threshold` du Critic.

---

## 7. Comportement de `final_answer` — Composant Purement Déclaratif

### 7.1 Règle absolue

> **`final_answer` est TOUJOURS retourné strictement identique à `answer` reçu en entrée.**

Le Verifier ne tronque rien, ne reformule rien, n'insère aucun avertissement dans le texte de la réponse. Dans tous les cas (`is_grounded=True` ou `is_grounded=False`), le champ `final_answer` de `VerificationResult` est une copie exacte de `answer`.

### 7.2 Justification — Séparation des Responsabilités

Ce principe de **composant purement déclaratif** est une décision d'architecture actée, cohérente avec le principe de séparation des responsabilités appliqué au Critic :

- **Le Verifier juge** : il produit un verdict (`is_grounded`, `faithfulness_score`, `unsupported_claims`)
- **L'orchestrateur décide** : la décision de re-générer une réponse (si `is_grounded=False` et budget encore disponible), ou d'insérer un avertissement dans la réponse présentée à l'utilisateur (si le budget global est épuisé), est une responsabilité du routeur conditionnel du graphe LangGraph (Sprint 6)

**Analogie directe avec le Critic :** le Critic ne re-formule pas la `sub_query` — il produit un `CriticEvaluation.feedback`, et c'est le Planner (via l'orchestrateur) qui reformule. Le Verifier ne modifie pas `answer` — il produit un `VerificationResult`, et c'est l'orchestrateur qui décide quoi faire.

**Avantages de ce principe :**
1. **Testabilité indépendante** : le Verifier peut être testé unitairement avec n'importe quelle paire (`answer`, `sources`) sans dépendance à l'orchestrateur
2. **Prévisibilité** : un composant sans état qui ne modifie pas son entrée est déterministe et facile à déboguer
3. **Réutilisabilité** : le même Verifier peut être utilisé dans différents contextes d'orchestration (LangGraph, test unitaire, démo Streamlit) sans adaptation

> **Note sur la docstring de `VerificationResult`** : le champ `final_answer` est actuellement décrit dans `internal_models.py` comme `"Réponse finale validée (ou tronquée si non-fondée)."` La mention "tronquée" est une ambiguïté de la docstring existante. **Le comportement acté pour le Sprint 5 est : jamais tronquée.** Cette docstring sera mise à jour lors du Sprint 5.2 pour refléter le comportement réel.

---

## 8. Structure du Prompt d'Évaluation

### 8.1 Technique retenue : Chain-of-Thought structuré

Le Verifier utilise le même pattern Chain-of-Thought avec sortie TOON stricte que le Critic. Le LLM est guidé vers une démarche explicite : extraire les claims, les confronter aux sources, rendre un verdict par claim, avant de synthétiser.

### 8.2 Format de Sortie TOON Retenu — Option C (enregistrements séparés par `---`)

**Décision actée.** Le LLM produit un **unique bloc TOON `<<<...>>>`** contenant plusieurs enregistrements de claims, chacun séparé par une ligne `---`. Chaque enregistrement est un groupe de trois champs identiques (`claim_text`, `is_supported`, `source_chunk_id`).

```
<<<
claim_text      :: Rabat est la capitale du Maroc
is_supported    :: true
source_chunk_id :: c1
---
claim_text      :: La ville compte 500000 habitants
is_supported    :: false
source_chunk_id ::
>>>
```

**Justification du choix de l'Option C :**
- **Clés uniformes, non préfixées** : contrairement à l'ancien format `claim_N_text`/`claim_N_verdict` (Option B rejetée), les clés sont identiques pour chaque enregistrement — le LLM ne génère jamais de `claim_7_text` quand il en attendait `claim_6_text`. Moins sujet aux erreurs d'indexation.
- **Lisibilité** : le séparateur `---` est visuellement clair et n'interfère pas avec les valeurs des champs.
- **Extensible** : ajouter un champ à tous les enregistrements ne change pas le parsing.
- **Compatibilité minimale** : le séparateur `---` n'est pas un délimiteur réservé de TOON v1.0 (`::`, `|`, `<<<`, `>>>`), il peut être traité comme une ligne spéciale sans ambiguïté.

**Cas zéro claim (réponse sans affirmation vérifiable) :**
```
<<<
>>>
```
Le bloc TOON est valide mais vide — `parse_toon_records()` retourne une liste vide, ce qui déclenche le comportement `total_claims=0` documenté en section 5.2.

### 8.3 Extension de `toon_utils.py` — `parse_toon_records()`

Ce format multi-enregistrements nécessite une **nouvelle fonction publique** dans `src/reasoning/shared/toon_utils.py` :

```python
# Pseudo-code — pas le code final
def parse_toon_records(raw: str) -> list[dict[str, Any]]:
    """Parse un bloc TOON contenant plusieurs enregistrements séparés par '---'.

    Extrait d'abord le bloc <<<...>>> via _extract_toon_block(), puis découpe
    le contenu en enregistrements sur les lignes '---' et applique le parsing
    clé::valeur de _infer_value() à chacun.

    Returns:
        Liste de dicts, un par enregistrement. Liste vide si le bloc est vide.

    Raises:
        ToonParseError: Si le bloc <<<...>>> est absent ou malformé.
    """
```

**Règle de périmètre stricte :**
- `parse_toon_records()` est **ajoutée** à `toon_utils.py` sans modifier les fonctions existantes
- `parse_toon_to_dict()`, `_extract_toon_block()`, `dump_dict_to_toon()` restent **inchangées**
- L'Analyzer, le Planner et le Critic continuent d'appeler `parse_toon_to_dict()` sans modification
- Seul `verifier.py` appelle `parse_toon_records()`

### 8.4 Piège de Typage — `is_supported` est une Chaîne, pas un Booléen

> **Avertissement d'implémentation obligatoire à lire avant d'écrire `verifier.py`.**

La fonction `_infer_value()` de `toon_utils.py` applique une inférence de type basée sur des patterns numériques (`int`, `float`) et sur le séparateur `|` (listes). Il n'existe **pas de règle d'inférence booléenne native** dans TOON v1.0.

**Conséquence directe :** après parsing, `is_supported` est toujours une **chaîne Python** (`str`), jamais un `bool` :

```python
# Ce que parse_toon_records() retourne — exemple
{
    "claim_text":      "OpenAI was founded in 2015",
    "is_supported":    "true",    # str, PAS bool True
    "source_chunk_id": "c1",
}
```

**Règle impérative dans `verifier.py` :** comparer la chaîne explicitement, ne jamais supposer un `bool` Python :

```python
# Correct
if record["is_supported"].strip().lower() == "true":
    claims_supported += 1

# INCORRECT — ne jamais écrire
if record["is_supported"]:  # "false" est truthy en Python !
    claims_supported += 1
```

Le champ `source_chunk_id` avec valeur vide (`source_chunk_id ::`) est parsé comme `None` par `_infer_value()` (valeur None = chaîne vide après `::`) — ce comportement est conforme et attendu.

### 8.5 Template du Prompt

Ce prompt sera défini dans `src/reasoning/verifier/prompts.py`.

```
You are a faithfulness evaluator for a RAG reasoning engine.

UNIQUE MISSION: Verify that every factual claim in the answer is traceable
                to the provided source chunks. Do NOT re-answer the question.

─── DEFINITIONS ─────────────────────────────────────────────────────────────

A CLAIM is any factual proposition in the answer that can be true or false:
a date, a name, a causal relationship, a number, an attribution.
Rhetorical transitions ("Based on the context...", "Here is a summary..."),
hedging phrases, and purely subjective opinions are NOT claims.

A claim is SUPPORTED if at least one source chunk contains explicit or strongly
implied information corroborating it. A claim is UNSUPPORTED if no chunk
mentions it, if chunks contradict it, or if it requires an inference beyond
what the sources reasonably allow.

─── CHAIN-OF-THOUGHT PROCESS ────────────────────────────────────────────────

Before producing your output, reason through the following steps:
  Step 1: Read the answer and identify each distinct factual claim.
  Step 2: For each claim, search the source chunks for supporting evidence.
  Step 3: Mark each claim as true (supported) or false (unsupported).
  Step 4: Note the chunk_id of the supporting chunk, or leave empty if none.
  Step 5: Produce the TOON block below.

─── OUTPUT FORMAT (STRICT TOON) ─────────────────────────────────────────────

Return ONLY this TOON block. No text before or after.
Separate each claim record with a line containing only "---".

<<<
claim_text      :: <exact or paraphrased claim from the answer>
is_supported    :: true | false
source_chunk_id :: <chunk_id of supporting chunk, or empty>
---
claim_text      :: <next claim>
is_supported    :: true | false
source_chunk_id :: <chunk_id or empty>
... (repeat for each claim)
>>>

If there are NO verifiable factual claims in the answer, output an empty block:
<<<
>>>

─── INPUTS ──────────────────────────────────────────────────────────────────

Answer to verify:
  {answer}

Source chunks ({n_chunks} chunks):
{chunks_content}

─── YOUR EVALUATION ─────────────────────────────────────────────────────────

Think step by step, then produce the TOON block.
```

### 8.6 Format des Chunks dans le Prompt

Chaque chunk est présenté avec son `chunk_id` (repris tel quel dans la sortie TOON), sa `source`, et son `content` tronqué à `_MAX_CHUNK_CHARS` caractères (valeur proposée : 600) pour rester dans la fenêtre de contexte du modèle 7B :

```
  [chunk_id=openai-history-001 | source=openai_history.pdf]: OpenAI was
  founded in December 2015 by Elon Musk, Sam Altman, Greg Brockman...
  [truncated]
```

Le `chunk_id` est explicitement affiché pour que le LLM puisse le référencer dans le champ `source_chunk_id`.

### 8.7 Schéma TOON Attendu en Sortie

| Champ TOON | Type TOON | Contrainte |
|---|---|---|
| `claim_text` | Chaîne libre | Obligatoire par enregistrement |
| `is_supported` | `"true"` ou `"false"` (chaîne) | Obligatoire par enregistrement — **jamais converti en `bool` Python par le parser** |
| `source_chunk_id` | Chaîne (chunk_id) ou vide | Obligatoire par enregistrement ; vide si claim non supporté |

**Champs calculés en Python, absents du TOON :**
- `faithfulness_score` — calculé comme `claims_supported / total_claims` après parsing
- `is_grounded` — dérivé de `faithfulness_score >= seuil`
- `unsupported_claims` — liste des `claim_text` dont `is_supported == "false"`
- `final_answer` — copie directe de `answer` (jamais généré par le LLM)

### 8.8 Exemple d'Entrée/Sortie Complet

**Entrée :**
```
Answer to verify:
  OpenAI was founded in December 2015 by Sam Altman and Elon Musk. GPT-4 was
  released in March 2023. The model supports text and image inputs.

Source chunks (2 chunks):
  [chunk_id=openai-history-001 | source=openai_history.pdf]:
  OpenAI was founded in December 2015 by Elon Musk, Sam Altman, Greg Brockman,
  Ilya Sutskever, Wojciech Zaremba, and John Schulman.

  [chunk_id=gpt4-report-002 | source=gpt4_technical_report.pdf]:
  GPT-4 was released by OpenAI in March 2023. It is a large multimodal model
  capable of processing both text and image inputs.
```

**Sortie LLM attendue (Option C) :**
```
<<<
claim_text      :: OpenAI was founded in December 2015 by Sam Altman and Elon Musk
is_supported    :: true
source_chunk_id :: openai-history-001
---
claim_text      :: GPT-4 was released in March 2023
is_supported    :: true
source_chunk_id :: gpt4-report-002
---
claim_text      :: The model supports text and image inputs
is_supported    :: true
source_chunk_id :: gpt4-report-002
>>>
```

**Post-processing Python (seuil `0.80`) :**
```python
records            = parse_toon_records(raw_toon)  # list de 3 dicts
total_claims       = 3
claims_supported   = sum(1 for r in records if r["is_supported"].strip().lower() == "true")
                   = 3
faithfulness_score = 3 / 3 = 1.0
is_grounded        = (1.0 >= 0.80)  → True
unsupported_claims = []
final_answer       = "OpenAI was founded in December 2015..."  (identique à l'entrée)
```

**Objet `VerificationResult` instancié :**
```
VerificationResult(
    is_grounded        = True,
    faithfulness_score = 1.0,
    unsupported_claims = [],
    final_answer       = "OpenAI was founded in December 2015 by Sam Altman and Elon
                          Musk. GPT-4 was released in March 2023. The model supports
                          text and image inputs."
)
```

---

## 9. Architecture du Composant `verifier/`

### 9.1 Fichiers à Créer

```
src/reasoning/verifier/
├── __init__.py          # Exporte Verifier
├── verifier.py          # Classe Verifier (logique principale)
└── prompts.py           # VERIFICATION_PROMPT (template CoT)
```

### 9.2 Interface Publique de `Verifier`

```python
# Pseudo-code de l'interface — pas le code final

class Verifier:
    def __init__(
        self,
        model: str = DEFAULT_REASONING_MODEL,   # Qwen 2.5 7B
        api_base: str = OLLAMA_BASE_URL,
        temperature: float = 0.0,
        faithfulness_threshold: float = 0.80,   # Paramétrable, pas hardcodé
    ) -> None: ...

    def verify(
        self,
        answer: str,
        sources: list[RetrievedChunk],
    ) -> VerificationResult: ...
```

### 9.3 Flux d'Exécution Détaillé

```
verify(answer, sources)
        │
        ├── Cas spécial : sources == [] → retour immédiat défensif (pas d'appel LLM)
        │       └── VerificationResult(is_grounded=False, faithfulness_score=0.0,
        │                              unsupported_claims=["no sources available"],
        │                              final_answer=answer)
        │
        ├── Formater le prompt CoT avec answer + chunks formatés
        │
        ├── Appel LiteLLM → Qwen 7B (temperature=0)
        │
        ▼ Réponse TOON brute
        parse_toon_to_dict()    (shared/toon_utils.py)
        │
        ├── [Succès]
        │       ├── Parser total_claims, claim_N_text/verdict/source
        │       ├── Cas spécial : total_claims == 0
        │       │       └── faithfulness_score=1.0, is_grounded=True, unsupported_claims=[]
        │       ├── Calculer faithfulness_score = claims_supported / total_claims
        │       ├── Calculer is_grounded = (faithfulness_score >= threshold)
        │       ├── Construire unsupported_claims = [claim_N_text pour tout N où verdict=unsupported]
        │       └── Instancier VerificationResult(final_answer=answer, ...)
        │
        └── [ToonParseError / ValidationError / Timeout]
                │
                ▼
        Fallback défensif :
        VerificationResult(
            is_grounded       = False,
            faithfulness_score = 0.0,
            unsupported_claims = ["verification_failed"],
            final_answer      = answer
        )
        + log WARNING
```

### 9.4 Modèle LLM Utilisé

Le Verifier utilise **Qwen 2.5 7B** (`ollama/qwen2.5:7b`), le même modèle que le Planner et le Critic, configuré via `DEFAULT_REASONING_MODEL`. La vérification de fidélité requiert une compréhension sémantique fine comparable à celle du Critic.

| Paramètre | Valeur |
|---|---|
| Variable d'environnement | `DEFAULT_REASONING_MODEL` |
| Valeur par défaut | `ollama/qwen2.5:7b` |
| Température | `0.0` (déterministe) |
| `max_tokens` | `512` (le bloc TOON de sortie peut être long si nombreux claims) |

---

## 10. Gestion des Erreurs

| Scénario | Comportement |
|---|---|
| LLM retourne un bloc TOON valide | Parsing via `parse_toon_to_dict()`, calcul Python, instanciation `VerificationResult` |
| `faithfulness_score` hors `[0.0, 1.0]` | `ValidationError` Pydantic interceptée → fallback défensif |
| Bloc TOON malformé ou absent | `ToonParseError` interceptée → fallback défensif (score=0.0, is_grounded=False) |
| Timeout Ollama / erreur réseau | Exception LiteLLM interceptée → fallback défensif + log WARNING |
| `sources` est vide (`[]`) | Cas spécial : retour immédiat défensif sans appel LLM (section 5.3) |
| `answer` est vide (`""`) | Cas spécial : `total_claims=0`, `faithfulness_score=1.0`, `is_grounded=True` |
| `total_claims == 0` dans la réponse TOON | Score `1.0`, `is_grounded=True`, `unsupported_claims=[]` (section 5.2) |

---

## 11. Métriques de Qualité

Les métriques suivantes seront mesurées lors du Sprint 7 (évaluation RAGAS) pour valider la performance du Verifier :

| Métrique | Cible | Méthode de mesure |
|---|---|---|
| **Précision de détection** (`is_grounded`) | ≥ 85% | Comparaison avec labels ground-truth sur dataset HotpotQA |
| **Taux de faux négatifs** (`is_grounded=False` pour réponse réellement correcte) | ≤ 10% | Mesure sur le sous-ensemble de requêtes SIMPLE à réponse validée |
| **Taux de fallback** (ToonParseError) | ≤ 5% | `count(verification_failed) / total_verifications` |
| **Latence P95** | ≤ 6s | Mesure sur le modèle Qwen 7B local avec 3 claims moyens |
| **RAGAS Faithfulness** | ≥ 0.75 | Score RAGAS officiel sur le dataset de référence Sprint 7 |

---

## 12. Plan de Vérification

### Tests Unitaires (`tests/unit/test_verifier.py`)

| Test | Mock | Assertion clé |
|---|---|---|
| `test_fully_grounded_answer` | TOON avec 3 claims, 3 `supported` | `is_grounded=True`, `faithfulness_score=1.0` |
| `test_partially_grounded_answer` | TOON avec 3 claims, 1 `unsupported` | `is_grounded=False`, `faithfulness_score ≈ 0.67` |
| `test_fully_ungrounded_answer` | TOON avec 2 claims, 0 `supported` | `is_grounded=False`, `faithfulness_score=0.0` |
| `test_no_claims_answer` | TOON avec `total_claims=0` | `is_grounded=True`, `faithfulness_score=1.0` |
| `test_empty_sources_no_llm_call` | `sources=[]` | `completion` non appelé, `is_grounded=False` |
| `test_final_answer_never_modified` | Tout résultat LLM | `result.final_answer == answer` (identité stricte) |
| `test_unsupported_claims_populated` | TOON avec verdict `unsupported` | `len(result.unsupported_claims) > 0` |
| `test_fallback_on_toon_parse_error` | `side_effect=ToonParseError` | `isinstance(result, VerificationResult)`, `faithfulness_score=0.0` |
| `test_fallback_on_timeout` | `side_effect=TimeoutError` | `is_grounded=False`, aucun crash |
| `test_custom_threshold` | `Verifier(faithfulness_threshold=0.90)` | `is_grounded=False` pour score=0.85 |

### Tests d'Intégration (`tests/integration/test_verifier_live.py`)

Protégés par `@pytest.mark.integration`. Nécessitent Ollama + Qwen 7B.

Cible : vérifier que pour 3 paires (`answer`, `sources`) de référence (une réponse correcte, une partiellement hallucinnée, une entièrement hallucinnée), le Verifier produit un `VerificationResult` valide avec `faithfulness_score` dans `[0.0, 1.0]` et `final_answer` identique à `answer`.

---

## 13. Questions Ouvertes

Les points suivants dépassent le périmètre de la tâche 5.1 et nécessitent une décision lors de l'implémentation (5.2) :

1. **Nombre maximum de chunks dans le prompt** : si `sources` contient de nombreux chunks (multi-hop avec 3 étapes × `top_k=5` = 15 chunks), le prompt peut dépasser la fenêtre de contexte du modèle 7B. Une stratégie de sélection (top-K chunks par score, ou troncature à un budget fixe de tokens) doit être définie lors de 5.2.

2. **Niveau de granularité des claims** : le LLM peut sur-décomposer ("OpenAI was founded" et "in December 2015" comme deux claims séparés) ou sous-décomposer. Une instruction de calibrage dans le prompt peut être nécessaire — à ajuster empiriquement lors de 5.2.

3. **Localisation du seuil `faithfulness_threshold`** : même question ouverte qu'au Critic — paramètre constructeur, `.env`, ou `configs/thresholds.toml`. Décision déléguée au Sprint 5.2 / Sprint 8, en cohérence avec le choix retenu pour `sufficiency_threshold`.

---

*Spécification RAG-REASON — Verifier v1.0 — Sprint 5*
*À lire en conjonction avec `docs/critic_spec.md` et `src/reasoning/contracts/internal_models.py`*
