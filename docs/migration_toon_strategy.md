# Stratégie de Migration : JSON → TOON
## Projet RAG-REASON — Document Technique v1.0

| | |
|---|---|
| **Auteur** | Mohammed Solimani |
| **Date** | Juillet 2026 |
| **Décision d'origine** | Directive Mme Zineb Hidila — migration format TOON |
| **Scope** | Module REASONING (Sprint 2 existant) + interface avec module ACTION (Ihssane) |
| **Statut** | Plan de migration — À exécuter avant Sprint 3 |

---

## Analyse d'Impact Préliminaire

Avant de détailler le plan, il est essentiel de cartographier précisément ce que la migration affecte dans notre base de code existante.

### Ce qui change

| Composant | Localisation | Nature du changement |
|---|---|---|
| **Prompt Few-Shot** | `src/reasoning/analyzer/prompts.py` | Remplacement du schéma JSON et des 10 exemples par des équivalents TOON |
| **Extracteur de réponse LLM** | `analyzer.py` → `_extract_json()` | Renommer en `_extract_toon()`, adapter la regex et la logique d'extraction |
| **Pipeline de parsing** | `analyzer.py` → `_classify_with_llm()` | Remplacer `model_validate_json()` par `parse_toon_to_dict()` + `model_validate()` |
| **Interface ACTION** | `contracts/action_interface.py` | Les modèles Pydantic restent, la sérialisation sur le fil passe en TOON |
| **Mocks de test** | `tests/unit/test_analyzer.py` | Toutes les réponses LLM mockées passent de JSON à TOON |

### Ce qui NE change PAS

> [!IMPORTANT]
> Les modèles Pydantic (`AnalysisResult`, `RetrievalRequest`, `RetrievalResponse`, etc.) **ne sont pas modifiés**. Pydantic reste le validateur interne unique. TOON est exclusivement un **format de sérialisation sur le fil** (LLM output + communication inter-modules), jamais un format de stockage interne.

La couverture de tests actuelle (98%) et l'architecture hybride (LLM + fallback heuristique) sont **intégralement préservées**.

---

## Phase 1 : Définition du Standard TOON

### 1.1 Pourquoi TOON ?

Le format TOON (*Token-Oriented Object Notation*) est conçu pour améliorer la fiabilité de la génération de données structurées par les LLMs, en particulier les modèles de petite taille (3B paramètres). Ses avantages sur JSON dans ce contexte :

- **Moins de tokens ambigus** : les accolades `{`, `}`, les guillemets doubles imbriqués et les virgules de JSON sont des sources fréquentes d'erreurs pour les petits modèles
- **Parsing robuste** : les délimiteurs ligne-par-ligne rendent l'extraction plus résistante au "texte parasite" que le LLM insère souvent autour de sa réponse
- **Lisibilité directe** : un humain lisant les logs peut interpréter une réponse TOON plus rapidement qu'un bloc JSON compact

### 1.2 Syntaxe TOON — Spécification v1.0

Nous adoptons la convention suivante pour RAG-REASON. Cette convention **doit être validée avec Ihssane** avant toute implémentation (voir Phase 2).

#### Règles de base

| Règle | Description | Exemple |
|---|---|---|
| **Délimiteur de bloc** | Un bloc TOON est encadré par `<<<` (ouverture) et `>>>` (fermeture) | `<<<` … `>>>` |
| **Séparateur clé/valeur** | Le signe `::` sépare une clé de sa valeur | `query_type :: SIMPLE` |
| **Séparateur de liste** | Les éléments d'une liste sont séparés par `\|` | `detected_entities :: LangChain \| LangSmith` |
| **Une clé par ligne** | Chaque paire clé/valeur occupe une ligne unique | — |
| **Casse des clés** | Minuscules snake_case, identiques aux noms de champs Pydantic | `reasoning_budget :: 3` |
| **Pas de types explicites** | Les types sont inférés au parsing (float si `.`, int sinon, liste si `\|`) | — |

#### Exemple complet — AnalysisResult au format TOON

```
<<<
query_type :: MULTI_HOP
confidence :: 0.92
detected_entities :: LangChain | LangSmith
reasoning_budget :: 3
>>>
```

Équivalent JSON (avant migration) :
```json
{
  "query_type": "MULTI_HOP",
  "confidence": 0.92,
  "detected_entities": ["LangChain", "LangSmith"],
  "reasoning_budget": 3
}
```

#### Exemple — RetrievalRequest au format TOON (interface avec Ihssane)

```
<<<
query_id :: qid-001
sub_query :: Qui a fondé OpenAI ?
hop_index :: 0
top_k :: 5
filters ::
metadata ::
>>>
```

> [!NOTE]
> Les champs optionnels (`filters`, `metadata`) à valeur `None` sont représentés par une valeur vide après `::`. Le parser doit les convertir en `None` Python.

### 1.3 Règles de Robustesse du Parser

Pour garantir que le parser est tolérant aux variantes de formatage du LLM :

1. **Ignorer les lignes vides** à l'intérieur du bloc `<<< ... >>>`
2. **Trim whitespace** sur les clés et les valeurs avant parsing
3. **Casse insensible** sur les valeurs d'Enum (`SIMPLE` = `simple` = `Simple`)
4. **Tolérance au bloc manquant** : si `<<<` est absent mais que les lignes `clé :: valeur` sont présentes, tenter le parsing en mode dégradé
5. **Encodage UTF-8** : les entités peuvent contenir des caractères accentués (`rétropropagation`)

### 1.4 Décision de Gouvernance

Ce standard doit être formalisé dans un fichier `docs/toon_spec.md` versionné dans le dépôt partagé entre les deux modules. Toute modification de syntaxe doit faire l'objet d'un accord bilatéral Mohammed ↔ Ihssane et d'une montée de version (`v1.1`, `v2.0`).

---

## Phase 2 : L'Utilitaire Partagé `toon_utils.py`

### 2.1 Rôle et Positionnement

`toon_utils.py` est le **seul point de vérité** pour la sérialisation/désérialisation TOON dans le projet. Ce fichier doit être :

- **Partagé** entre le module REASONING (Mohammed) et le module ACTION (Ihssane), idéalement dans un package commun ou copié dans les deux dépôts avec versioning explicite
- **Sans dépendance LLM** : uniquement de la manipulation de chaînes Python standard
- **Agnostique au schéma** : il opère sur des `dict[str, Any]` bruts, sans connaissance des modèles Pydantic spécifiques

### 2.2 Spécification des Fonctions

#### `parse_toon_to_dict(raw: str) -> dict[str, Any]`

**Rôle** : Convertit une chaîne TOON brute (réponse LLM) en dictionnaire Python.

**Algorithme attendu :**
```python
# Pseudo-code illustratif — pas le code final

def parse_toon_to_dict(raw: str) -> dict[str, Any]:
    # 1. Extraire le contenu entre <<< et >>>
    #    Si absent : tenter le parsing ligne-par-ligne en mode dégradé
    content = _extract_toon_block(raw)

    result: dict[str, Any] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or "::" not in line:
            continue

        key, _, value = line.partition("::")
        key = key.strip()
        value = value.strip()

        # Conversion de type
        if "|" in value:
            result[key] = [v.strip() for v in value.split("|") if v.strip()]
        elif value == "":
            result[key] = None
        elif _is_float(value):
            result[key] = float(value)
        elif _is_int(value):
            result[key] = int(value)
        else:
            result[key] = value

    return result
```

**Gestion des erreurs** : lever une `ToonParseError` (exception custom) si le contenu est vide ou si aucune paire clé/valeur n'est détectable.

#### `dump_dict_to_toon(data: dict[str, Any]) -> str`

**Rôle** : Convertit un dictionnaire Python (ou le `.model_dump()` d'un modèle Pydantic) en chaîne TOON.

**Algorithme attendu :**
```python
# Pseudo-code illustratif — pas le code final

def dump_dict_to_toon(data: dict[str, Any]) -> str:
    lines = ["<<<"]
    for key, value in data.items():
        if isinstance(value, list):
            serialized = " | ".join(str(v) for v in value) if value else ""
        elif value is None:
            serialized = ""
        else:
            serialized = str(value)
        lines.append(f"{key} :: {serialized}")
    lines.append(">>>")
    return "\n".join(lines)
```

#### Intégration avec Pydantic — Le Flux Complet

Pydantic reste l'arbitre de validation finale. Le flux de parsing devient :

```
Réponse LLM (str TOON brut)
        │
        ▼
parse_toon_to_dict(raw)          ← toon_utils.py
        │
        ▼
dict[str, Any]
        │
        ▼
AnalysisResult.model_validate(d) ← Pydantic (validation des types et contraintes)
        │
        ▼
AnalysisResult (objet Python valide)
```

> [!IMPORTANT]
> La `ToonParseError` et la `ValidationError` de Pydantic sont deux exceptions distinctes à capturer séparément dans le bloc `try/except` de `analyzer.py`. La première indique un problème de format TOON, la seconde un problème de cohérence des données.

### 2.3 Convention de Coordination avec Ihssane

Pour l'interface `RetrievalRequest` / `RetrievalResponse`, les deux modules doivent s'accorder sur :

1. La version exacte de `toon_utils.py` utilisée (hash de fichier ou tag Git)
2. La représentation des champs `None` (valeur vide après `::`)
3. La représentation des listes vides (`detected_entities ::` vs `detected_entities :: []`)
4. Le comportement en cas de champ inconnu (ignorer silencieusement vs lever une erreur)

**Recommandation** : organiser une session de revue de 30 minutes avec Ihssane pour valider la spécification TOON avant de commencer l'implémentation. Documenter les décisions dans `docs/toon_spec.md`.

---

## Phase 3 : Refactoring du Sprint 2

Les modifications à effectuer dans le Query Analyzer existant sont **localisées et chirurgicales**. L'architecture hybride, les 4 niveaux de classification et la logique de fallback ne sont pas touchés.

### 3.1 Étape 3.1 — Mise à jour du Prompt Few-Shot (`prompts.py`)

**Changements requis :**
- Remplacer le bloc `─── SCHÉMA JSON ATTENDU ───` par `─── SCHÉMA TOON ATTENDU ───`
- Remplacer les 10 exemples `JSON : {{...}}` par leurs équivalents TOON
- Mettre à jour l'instruction finale

**Avant :**
```
─── SCHÉMA JSON ATTENDU ────────────────────────────────────────────────────

{{
  "query_type": "SIMPLE | MULTI_HOP | COMPARATIVE | AMBIGUOUS",
  "confidence": <float entre 0.0 et 1.0>,
  "detected_entities": ["entité_1", "entité_2"],
  "reasoning_budget": <entier : SIMPLE=1, MULTI_HOP=3, COMPARATIVE=2, AMBIGUOUS=0>
}}

─── EXEMPLES ────────────────────────────────────────────────────────────────

Requête : "Qu'est-ce que la rétropropagation ?"
JSON    : {{"query_type": "SIMPLE", "confidence": 0.97, ...}}
```

**Après :**
```
─── SCHÉMA TOON ATTENDU ────────────────────────────────────────────────────

<<<
query_type :: SIMPLE | MULTI_HOP | COMPARATIVE | AMBIGUOUS
confidence :: <float entre 0.0 et 1.0>
detected_entities :: entité_1 | entité_2
reasoning_budget :: <entier : SIMPLE=1, MULTI_HOP=3, COMPARATIVE=2, AMBIGUOUS=0>
>>>

─── EXEMPLES ────────────────────────────────────────────────────────────────

Requête : "Qu'est-ce que la rétropropagation ?"
TOON    :
<<<
query_type :: SIMPLE
confidence :: 0.97
detected_entities :: rétropropagation
reasoning_budget :: 1
>>>
```

> [!NOTE]
> La lisibilité des exemples multi-lignes est un avantage direct de TOON : chaque champ est immédiatement identifiable, ce qui renforce l'apprentissage par le modèle lors du Few-Shot.

### 3.2 Étape 3.2 — Refactoring de `_extract_json()` → `_extract_toon()`

**Dans `analyzer.py`**, la méthode statique `_extract_json` est renommée `_extract_toon` et sa logique est adaptée.

**Avant :**
```python
@staticmethod
def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("{"):
        return raw
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError(f"Aucun objet JSON trouvé dans la réponse LLM : {raw!r}")
```

**Après :**
```python
@staticmethod
def _extract_toon(raw: str) -> str:
    """Extrait le bloc TOON (<<<...>>>) d'une réponse LLM brute.

    Tente d'abord l'extraction via les délimiteurs officiels,
    puis en mode dégradé sur les lignes clé :: valeur.
    """
    raw = raw.strip()

    # Cas idéal : délimiteurs <<< >>> présents
    match = re.search(r"<<<(.*?)>>>", raw, re.DOTALL)
    if match:
        return "<<<" + match.group(1) + ">>>"

    # Mode dégradé : lignes "clé :: valeur" sans délimiteurs
    lines = [l for l in raw.splitlines() if "::" in l]
    if lines:
        return "<<<\n" + "\n".join(lines) + "\n>>>"

    raise ValueError(f"Aucun bloc TOON trouvé dans la réponse LLM : {raw!r}")
```

### 3.3 Étape 3.3 — Mise à jour de `_classify_with_llm()`

**Avant :**
```python
json_str = self._extract_json(raw_content)
result = AnalysisResult.model_validate_json(json_str)
```

**Après :**
```python
from reasoning.shared.toon_utils import parse_toon_to_dict, ToonParseError

toon_str = self._extract_toon(raw_content)
data = parse_toon_to_dict(toon_str)      # dict[str, Any]
result = AnalysisResult.model_validate(data)  # Pydantic valide
```

### 3.4 Étape 3.4 — Ajout de `ToonParseError` au bloc `except`

Le bloc de gestion des erreurs dans `analyze()` doit capturer la nouvelle exception :

**Avant :**
```python
except (ValidationError, ValueError, json.JSONDecodeError) as exc:
    logger.warning("Réponse LLM non parseable (%s: %s) — fallback.", ...)
```

**Après :**
```python
except (ValidationError, ValueError, ToonParseError) as exc:
    logger.warning("Réponse LLM non parseable TOON (%s: %s) — fallback.", ...)
```

L'import `json` devient inutile et peut être retiré du module.

### 3.5 Résumé des Fichiers Modifiés

| Fichier | Type de modification | Lignes impactées (estimé) |
|---|---|---|
| `src/reasoning/analyzer/prompts.py` | Réécriture du template | 100% du contenu |
| `src/reasoning/analyzer/analyzer.py` | Refactoring chirurgical | ~10 lignes |
| `src/reasoning/shared/toon_utils.py` | **Nouveau fichier** | ~60 lignes |
| `src/reasoning/contracts/action_interface.py` | Aucun changement | — |
| `src/reasoning/contracts/internal_models.py` | Aucun changement | — |

---

## Phase 4 : Stratégie de Mise à Jour des Tests

### 4.1 Philosophie de Migration des Tests

Les tests existants ont une structure saine grâce au découplage strict entre logique métier et appels LLM. La migration des 30 tests unitaires de `test_analyzer.py` se résume principalement à **remplacer les payloads mockés** : au lieu de simuler une réponse JSON, on simule une réponse TOON.

> [!TIP]
> Aucun test de `test_contracts.py` (47 tests) n't est impacté : les modèles Pydantic ne changent pas. La couverture 100% de ce module est acquise et stable.

### 4.2 Mise à Jour du Helper `_make_llm_response()`

La fonction helper qui construit les mocks est le **seul point central à modifier** pour mettre à jour la majorité des tests.

**Avant :**
```python
def _make_llm_response(json_payload: dict[str, Any]) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(json_payload)
    return mock_response
```

**Après :**
```python
from reasoning.shared.toon_utils import dump_dict_to_toon

def _make_llm_response(payload: dict[str, Any]) -> MagicMock:
    """Simule une réponse LLM au format TOON."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = dump_dict_to_toon(payload)
    return mock_response
```

Avec cette modification, **tous les tests utilisant `_make_llm_response()`** sont automatiquement migrés en TOON sans autre modification.

### 4.3 Tests Spécifiques au Format TOON

En plus de la migration des tests existants, ajouter les nouveaux cas de test suivants dans `TestLLMSuccessPath` et `TestInternalMethods` :

| Test à ajouter | Objectif |
|---|---|
| `test_llm_response_with_toon_delimiters` | Réponse avec `<<<...>>>` bien formée |
| `test_llm_response_toon_in_markdown_block` | Bloc TOON enveloppé dans ` ```toon ... ``` ` |
| `test_llm_response_toon_degraded_no_delimiters` | Lignes `clé :: valeur` sans délimiteurs |
| `test_extract_toon_raises_on_no_toon` | `ValueError` si aucun contenu TOON détectable |
| `test_parse_toon_list_with_pipe` | `detected_entities :: A \| B` → `["A", "B"]` |
| `test_parse_toon_empty_list` | `detected_entities ::` → `None` ou `[]` (selon convention retenue) |
| `test_parse_toon_none_value` | `filters ::` → `None` |
| `test_dump_dict_to_toon_roundtrip` | `parse(dump(d)) == d` pour tous les types de champs |

### 4.4 Mise à Jour des Tests d'Intégration

`test_analyzer_live.py` n'a **aucun mock** à modifier. Cependant, si le modèle Ollama retourne du texte TOON malformé (probable lors des premières exécutions avec le nouveau prompt), la classe `TestAnalyzerLive` doit inclure un test de robustesse :

```python
def test_fallback_is_not_activated_with_new_toon_prompt(
    self, live_analyzer: QueryAnalyzer
) -> None:
    """Vérifie que le nouveau prompt TOON génère bien des réponses parseable
    (confidence > 0.55 signifie que le chemin LLM a réussi)."""
    result = live_analyzer.analyze("Qu'est-ce que la rétropropagation ?")
    assert result.confidence > 0.55, (
        "Le fallback a été activé — le prompt TOON doit être revu."
    )
```

### 4.5 Commande de Validation Post-Migration

Après l'implémentation de toutes les phases, exécuter la suite complète avec la cible de couverture maintenue :

```bash
# Régression complète — cible ≥ 98% maintenue
uv run pytest tests/unit/ -q --tb=short \
    --cov=src/reasoning/analyzer \
    --cov=src/reasoning/shared \
    --cov-report=term-missing \
    -m "not integration"

# Validation TOON uniquement
uv run pytest tests/unit/test_analyzer.py -v -k "toon"

# Test d'intégration (Ollama requis)
uv run pytest tests/integration/ -v -m integration
```

---

## Phase 5 : Checklist de Préparation au Sprint 3

Avant d'entamer le développement du Planner, valider que **chacun des points suivants** est coché.

### ✅ Checklist Technique

- [ ] **`docs/toon_spec.md` créé et validé** avec Ihssane (syntaxe, cas limites, versioning)
- [ ] **`src/reasoning/shared/toon_utils.py` implémenté** avec `parse_toon_to_dict`, `dump_dict_to_toon`, `ToonParseError`
- [ ] **Tests unitaires de `toon_utils.py`** écrits dans `tests/unit/test_toon_utils.py` (couverture 100%)
- [ ] **`prompts.py` reécrit** avec les 10 exemples Few-Shot en format TOON
- [ ] **`analyzer.py` refactorisé** : `_extract_toon()`, pipeline TOON → dict → Pydantic
- [ ] **`test_analyzer.py` migré** : helper `_make_llm_response()` mis à jour, nouveaux cas TOON
- [ ] **Suite de tests complète exécutée** : 0 régression, couverture ≥ 98%
- [ ] **Commit de migration créé** : `git commit -m "refactor: migrate JSON → TOON format (Sprint 2 → Sprint 3 boundary)"`
- [ ] **Pre-commit hooks passent** : ruff ✅, mypy ✅, trailing-whitespace ✅

### ✅ Checklist de Coordination avec Ihssane

- [ ] **Version de `toon_utils.py` synchronisée** entre les deux dépôts (même hash)
- [ ] **Test de sérialisation croisée validé** : un `RetrievalRequest` sérialisé par REASONING est correctement parsé par ACTION
- [ ] **Convention sur les champs `None`** documentée et testée bilatéralement
- [ ] **Convention sur les listes vides** documentée (`::` vs `:: []`)
- [ ] **Contrat d'interface `action_interface.py` v1.1 taggé** si la migration TOON génère un changement de comportement observable

### ✅ Checklist Sprint 3 — Environnement Prêt

- [ ] **`docs/planner_spec.md` à rédiger** (équivalent de `docs/analyzer_spec.md` pour le Planner)
- [ ] **Modèle Qwen 2.5 7B disponible** : `ollama pull qwen2.5:7b` exécuté et vérifié
- [ ] **`AgentState.plan`** correctement initialisé à `None` dans tous les tests existants
- [ ] **Contrats `ExecutionPlan` et `PlanStep`** relus et validés (déjà définis au Sprint 1)
- [ ] **Variable d'environnement `DEFAULT_REASONING_MODEL`** configurée dans `.env` (≠ `DEFAULT_FAST_MODEL` du Query Analyzer)

---

## Risques et Mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Le modèle Qwen 3B génère du TOON malformé | Moyenne | Élevé | Le fallback heuristique absorbe la panne ; itérer sur les exemples Few-Shot |
| Divergence de `toon_utils.py` entre les deux modules | Haute | Critique | Fichier versionné dans un dépôt partagé ou copie avec checksum vérifié au démarrage |
| Régression de couverture sous 98% | Faible | Moyen | Exécuter la suite avant chaque commit ; le hook pre-commit bloque si les tests échouent |
| Latence accrue due au parsing TOON multi-lignes | Très faible | Négligeable | TOON est parsé en O(n) sur les lignes — négligeable face à la latence LLM |
| Incompatibilité de la convention liste vide | Moyenne | Faible | Définir la convention explicitement dans `toon_spec.md` et tester bilatéralement |

---

*Document technique RAG-REASON — Migration JSON → TOON — v1.0*
*À versionner dans `docs/` et à partager avec Ihssane avant implémentation.*
