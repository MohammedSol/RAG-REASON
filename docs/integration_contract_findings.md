# Écarts de contrat REASONING × ACTION — constats du Sprint I3

**Date des mesures** : 18 août 2026
**Branche** : `integration/action-module`
**Dépôt fournisseur** : `astraexec-integration` (fork du module ACTION d'Ihssane)

Toutes les valeurs de ce document ont été **mesurées**, jamais reprises d'une
documentation antérieure. Les points de mesure sont donnés en `fichier:ligne`.

Ce sprint **constate et documente**. Il ne corrige rien : aucun fichier de
`app/` (fournisseur) ni de `src/reasoning/` (hors tests) n'a été modifié.

---

## 0. Les deux copies du contrat

Le contrat Pydantic existe en deux exemplaires, un par dépôt :

| Rôle | Fichier |
|---|---|
| Consommateur | `src/reasoning/contracts/action_interface.py` |
| Fournisseur | `app/schemas/retrieval_contract.py` |

**Constat : les déclarations de champs sont identiques.** Un `diff` restreint
aux lignes de déclaration ne produit aucune sortie ; seuls les en-têtes de
docstring diffèrent.

```
RetrievedChunk    : chunk_id: str · content: str · source: str · relevance_score: float
RetrievalRequest  : query_id: str · sub_query: str · hop_index: int
                    filters: dict[str, Any] | None = None
                    top_k: int = Field(gt=0)
                    metadata: dict[str, Any] | None = None
RetrievalResponse : query_id: str · chunks: list[RetrievedChunk]
                    retrieval_score: float | None = None
                    metadata: dict[str, Any] | None = None
```

Rien ne garantit que cela le reste : deux dépôts, deux personnes. C'est la
raison d'être des fixtures partagées de `tests/contracts/fixtures/`, chargées
à l'identique des deux côtés.

---

## 1. Plafond de résultats — **aucun écart en pratique**

### 1.1 Valeur réelle du plafond côté fournisseur

`RetrievalAdapter.ENGINE_MAX_RESULTS = 5` — `app/integration/retrieval_adapter.py:68`.

**Cette constante est purement déclarative.** Un `grep` sur tout le dépôt
fournisseur montre qu'elle n'apparaît que dans des commentaires et une
docstring : **elle n'est référencée par aucun code exécutable.** Le plafond
effectif vient d'ailleurs :

```python
# app/retrieval/fusion_search.py:244
def search(self, query: str, top_k: int = 5, ...):
    ...
    return results[:top_k]          # ligne 316

# app/api/main.py:62 — FusionTool.execute ne passe PAS de top_k
results = self.search_engine.search(query, profile=profile)
```

Le moteur retourne donc **au plus 5 résultats**, quel que soit le `top_k`
demandé. La valeur 5 est le défaut de la signature de `search`, jamais
surchargé.

### 1.2 Comportement quand `top_k` dépasse le plafond

**Troncature silencieuse.** Aucune erreur, aucun avertissement, aucun
signalement dans la réponse. Mesuré sur l'API réelle avec
`retrieval_request_top_k_high.json` (`top_k = 20`) :

```json
"metadata": { "hop_index": 0, "n_chunks": 5, "top_k": 20 }
```

Le champ `metadata.top_k` réémet **20**, la valeur demandée, tandis que
`n_chunks` vaut **5**. Le consommateur reçoit donc moins que demandé, et rien
dans la réponse ne le lui signale autrement que par l'écart entre ces deux
champs — qu'il faut penser à comparer.

L'ordre des opérations dans l'adaptateur mérite d'être noté :
`_apply_filters` s'applique **avant** la troncature à `top_k`
(`retrieval_adapter.py:137` puis `:140`). Le filtre porte donc sur les 5
résultats déjà rendus par le moteur, pas sur l'ensemble du corpus.

### 1.3 Valeur de `top_k` réellement produite côté REASONING

**Rectification d'une prémisse de la mission** : le Planner ne produit
aucun `top_k`. Le contrat `PlanStep`
(`src/reasoning/contracts/internal_models.py:83`) déclare exactement quatre
champs — `step_id`, `sub_query`, `depends_on`, `status` — et **aucun d'eux
n'est `top_k`**. L'arbitrage ne porte donc pas sur le Planner.

La valeur vient du nœud `retrieve` :

```python
# src/reasoning/graph/nodes.py:53
_DEFAULT_TOP_K: int = 5

# src/reasoning/graph/nodes.py:236
def make_retrieve_node(client: RetrievalClient, top_k: int = _DEFAULT_TOP_K) -> NodeFn:

# src/reasoning/graph/nodes.py:263
request = RetrievalRequest(query_id=plan.plan_id, sub_query=step.sub_query,
                           hop_index=hop_index, top_k=top_k)
```

C'est une **constante de module**, pas une variable d'environnement : elle
n'est pas configurable sans modification du code. `build_graph` n'expose pas
non plus ce paramètre — `make_retrieve_node(retrieval_client)` est appelé sans
`top_k` (`src/reasoning/graph/graph.py:156`).

Valeur mesurée à l'exécution du vrai nœud, avec un client espion :
**`top_k == 5`**.

### 1.4 Verdict d'écart

| | Valeur mesurée |
|---|---|
| Plafond du moteur ACTION | **5** |
| `top_k` émis par REASONING | **5** |

> ### **AUCUN ÉCART EN PRATIQUE. Aucune correction nécessaire.**
>
> `5 ≤ 5` : le module REASONING ne demande jamais plus que ce que le moteur
> peut fournir. La décision d'arbitrage prise en amont (Option A — plafonner
> côté REASONING) **n'a pas d'objet aujourd'hui**. Ce n'est pas un point
> ouvert.

Ce n'est toutefois pas une coïncidence à laisser tacite : les deux valeurs
sont deux constantes indépendantes, dans deux dépôts, qu'aucun mécanisme ne
lie. Si l'une bouge sans l'autre, l'écart apparaît silencieusement.

Le test `test_emitted_top_k_versus_engine_ceiling`
(`tests/contracts/test_contract_reasoning.py`) fige la comparaison et échoue
si le `top_k` émis vient à dépasser 5. C'est le garde-fou.

**Si l'écart devait apparaître**, le point de modification côté REASONING est
`src/reasoning/graph/nodes.py:53` (`_DEFAULT_TOP_K`). Le mécanisme cohérent
avec le reste du projet serait `os.getenv("RETRIEVAL_TOP_K", "5")` après
`load_dotenv()`, comme `OLLAMA_BASE_URL` (`nodes.py:47`) et
`USE_REAL_ACTION` (`graph.py:80`), avec l'entrée correspondante dans
`.env.example`. **Cette mise en œuvre relève d'une mission séparée.**

### 1.5 Impact sur le Critic

Seuil réel, mesuré dans le code et non supposé :

```python
# src/reasoning/critic/critic.py:90
sufficiency_threshold: float = 0.70
# src/reasoning/critic/critic.py:265
is_sufficient = score >= self.sufficiency_threshold
```

Conforme à `docs/critic_spec.md §3` (« Seuil par défaut : `0.70` »).

**Le seuil porte sur `relevance_score`, pas sur le nombre de chunks.** Aucun
seuil quantitatif sur `len(chunks)` n'existe, ni dans le code ni dans la
spécification. `critic_spec.md §2.2` (COMPLETENESS) mentionne bien « le nombre
de chunks est insuffisant au regard de la complexité de la `sub_query` », mais
comme un critère **apprécié par le LLM**, agrégé dans le `relevance_score`
global — pas comme une règle en dur.

**Conclusion : le passage de 5 chunks demandés à 5 chunks reçus ne change
rien.** L'impact d'une réduction ne se poserait que si un écart apparaissait,
et il serait alors indirect : moins de passages fournis au LLM du Critic
peuvent lui faire juger la couverture incomplète et abaisser son
`relevance_score` sous 0,70, déclenchant une boucle de re-retrieval. Cet effet
est plausible mais **non mesuré** — aucune campagne d'évaluation n'a fait
varier le nombre de chunks. À ne pas présenter comme un fait tant qu'il n'est
pas mesuré.

---

## 2. Filtres

### 2.1 Ce que le fournisseur reconnaît réellement

**Une seule clé : `source`.** `app/integration/retrieval_adapter.py:163-180`.

```python
source_filter = filters.get("source")
if source_filter is None:
    return chunks
if isinstance(source_filter, (list, tuple, set)):
    allowed = set(source_filter)
    return [c for c in chunks if c.source in allowed]
return [c for c in chunks if c.source == source_filter]
```

- Accepte une **chaîne** (égalité stricte) ou une **liste/tuple/ensemble**
  (appartenance). Vérifié par test des deux formes.
- La comparaison est une **égalité exacte** sur le nom de fichier :
  ni casse ignorée, ni correspondance partielle, ni glob.
- **Toute autre clé est silencieusement ignorée** — ni honorée, ni rejetée.
- Des filtres composés uniquement de clés inconnues ne filtrent rien du tout :
  `filters.get("source")` vaut `None`, la fonction retourne la liste intacte.

Mesuré sur l'API réelle avec `retrieval_request_with_filters.json`, qui porte
volontairement une clé non reconnue (`published_after`) :

```json
"filters": { "source": ["Ed_Wood.txt", "Ed_Wood_(film).txt"],
             "published_after": "2020-01-01" }
→ 2 chunks, sources : ["Ed_Wood_(film).txt", "Ed_Wood.txt"]
```

`source` a bien filtré ; `published_after` n'a rien retiré de plus.

### 2.2 Ce que le consommateur envoie réellement

**Rien.** Le nœud `retrieve` construit sa `RetrievalRequest` sans `filters`
ni `metadata` (`src/reasoning/graph/nodes.py:259-264`) : les deux champs
restent à `None`, leur valeur par défaut.

### 2.3 Écart

**Écart latent, sans conséquence actuelle.** La surface de filtrage offerte
par le fournisseur est plus étroite que ce que le contrat laisse croire — le
type `dict[str, Any]` de `filters` n'annonce aucune restriction — mais le
consommateur ne s'en sert pas.

Le risque est **le silence**, pas l'étroitesse : le jour où REASONING
enverrait `{"published_after": ...}` ou `{"lang": "en"}`, il croirait filtrer
et recevrait des résultats non filtrés, sans le moindre signal. Le test
`test_unknown_filter_keys_are_silently_ignored` fixe ce comportement par
écrit, pour qu'il soit constaté plutôt que découvert.

---

## 3. Erreurs — correspondance codes HTTP ↔ exceptions typées

### 3.1 Codes retournés par le fournisseur, mesurés

| Situation | Code | Corps |
|---|---|---|
| Requête valide | **200** | `RetrievalResponse` |
| `RetrievalError` (échec moteur) | **400** | `{"detail": "...", "query_id": "..."}` |
| Corps non conforme au schéma | **422** | `{"detail": [ {type, loc, msg, input, ctx}, … ]}` |

Le **400** vient du gestionnaire explicite `app/api/main.py:99-105`, qui
ajoute le `query_id` au corps quand il est disponible — utile à la
corrélation multi-hop.

Le **422** est le comportement **par défaut de FastAPI**, non écrit par le
fournisseur. Son corps est une liste d'erreurs Pydantic et **ne contient pas
de `query_id`** : une requête rejetée à la validation ne peut donc pas être
corrélée par le corps de la réponse.

Corps réellement observés :

```
top_k = 0        → 422  {"detail":[{"type":"greater_than","loc":["body","top_k"],
                         "msg":"Input should be greater than 0","input":0,"ctx":{"gt":0}}]}
sub_query absent → 422  {"detail":[{"type":"missing","loc":["body","sub_query"], …}]}
hop_index="zero" → 422  {"detail":[{"type":"int_parsing", …}]}
sub_query = ""   → 400  {"detail":"Le parametre 'query' est obligatoire.","query_id":"q1"}
```

Le cas `sub_query = ""` est notable : le contrat l'accepte (`str` sans
contrainte de longueur), mais le moteur le refuse. La validation de non-vacuité
n'existe donc **d'aucun côté du contrat** — elle est appliquée par l'outil, en
bout de chaîne, et remonte en 400.

### 3.2 Correspondance avec les exceptions du consommateur

| Réponse du fournisseur | Exception levée par `ActionClient` |
|---|---|
| 400 | `ActionHTTPError(status_code=400)` |
| 422 | `ActionHTTPError(status_code=422)` |
| 500, 503 | `ActionHTTPError(status_code=…)` |
| Connexion refusée / timeout | `ActionUnavailableError` |
| Corps non-JSON ou non conforme | `ActionProtocolError` |
| `query_id` non corrélé | `ActionProtocolError` |

Toutes dérivent de `ActionClientError`. Vérifié en conditions réelles :
un 422 forcé sur l'API en marche produit bien
`ActionHTTPError status_code=422`, corps préservé.

### 3.3 Le 422 est aujourd'hui inatteignable

`RetrievalRequest.model_validate` s'exécute **côté REASONING, avant tout appel
réseau**. Un `top_k = 0` lève une `ValidationError` locale ; aucune requête
n'est émise. Mesuré : les trois cas de corps invalide échouent localement,
sans atteindre le réseau.

**Le module ACTION ne peut donc renvoyer 422 que si les deux copies du contrat
ont divergé** — exactement la panne que les fixtures partagées existent pour
détecter, et qu'elles détecteraient plus tôt, par un test unitaire nommant le
champ fautif.

---

## 4. Autres écarts constatés

### 4.1 `relevance_score` n'est pas comparable d'une requête à l'autre

**C'est le constat le plus lourd de conséquences de ce sprint.**

`FusionSearch.search` applique une **normalisation min-max des scores à
l'intérieur du lot de résultats d'une même requête**
(`app/retrieval/fusion_search.py:288-295`) :

```python
sem_norm = self.normalize_scores(sem_scores)
lex_norm = self.normalize_scores(lex_scores)
```

Le meilleur candidat d'une requête reçoit donc toujours une valeur normalisée
maximale sur sa meilleure composante, **indépendamment de sa pertinence
absolue**.

Deux mesures qui l'établissent :

**(a) Même chunk, deux scores.** Le chunk `2557` (`Scott_Derrickson.txt`) :

| Requête | `relevance_score` |
|---|---|
| « What nationality is Scott Derrickson? » | **0,96000** |
| « Which American film directors are known for horror films? » | **0,73791** |

**(b) Requête sans réponse dans le corpus.** Sous-requête volontairement
absurde — « What is the airspeed velocity of an unladen swallow on Neptune? » :

```
0.96000  MS_Excelsior_Neptune.txt
0.68984  What_Are_Little_Girls_Made_Of.txt
0.62153  What_Are_Little_Boys_Made_Of.txt
retrieval_score global : 0.6516
```

Le meilleur chunk obtient **0,96** alors qu'aucun document du corpus ne
répond à la question. Le score mesure « le meilleur de ce lot », pas
« pertinent ».

**Conséquence sur le Critic.** `docs/critic_spec.md §2.1` liste parmi les
signaux d'insuffisance : « Le `relevance_score` individuel de chaque chunk
(`RetrievedChunk.relevance_score`) est uniformément bas ». Ce signal est
**structurellement inopérant** avec ce moteur : le meilleur chunk est toujours
haut, y compris hors-sujet. Le Critic ne doit pas s'y fier, et son prompt ne
doit pas y inviter le LLM.

Il en va de même pour `retrieval_score`, moyenne des pertinences
(`retrieval_adapter.py:182-188`) : 0,6516 sur une requête sans réponse contre
0,694 sur une requête pertinente — les deux valeurs ne se séparent pas.

**Ceci n'est un défaut d'aucun des deux modules pris isolément.** Le moteur
est cohérent avec lui-même ; le contrat ne promet nulle part que
`relevance_score` soit une mesure absolue — sa docstring dit « Score de
pertinence calculé par le module ACTION (0.0 – 1.0) ». L'écart est dans
l'**interprétation** qu'en fait `critic_spec.md`. À traiter comme une
correction de spécification, pas comme un correctif de code, et à arbitrer
avec Ihssane puisque cela touche à la sémantique de son moteur.

### 4.2 Le nœud `retrieve` journalise les pannes du Sprint I2 comme « inattendues »

`src/reasoning/graph/nodes.py:266-290` intercepte, dans l'ordre :
`(httpx.HTTPError, OSError, TimeoutError)` en `WARNING`, `ValidationError` en
`WARNING`, puis un `except Exception` final en **`ERROR`**, libellé « erreur
inattendue ».

Depuis le Sprint I2, `ActionClient` ne lève plus d'exceptions `httpx` : il
lève des `ActionClientError`, qui ne sont sous-classes d'aucun des types
listés. Elles tombent donc toutes dans la branche finale.

Mesuré, avec un client simulant une panne :

```
[ERROR] retrieve[step_1] : erreur inattendue (ActionUnavailableError: …) — réponse vide défensive.
```

**Le repli fail-closed fonctionne** — la réponse vide est bien produite, le
`query_id` conservé, le graphe poursuit. L'écart est d'**observabilité** : une
indisponibilité banale et anticipée du module ACTION est journalisée en
`ERROR` comme un incident imprévu. En exploitation, cela noie les vraies
anomalies.

`graph/` est hors périmètre de ce sprint. Correctif suggéré pour une mission
ultérieure : ajouter `ActionClientError` à la première clause `except`, en
`WARNING`.

### 4.3 Le `chunk_id` n'est pas stable dans le temps

Les `chunk_id` observés (`2557`, `987`, `986`) sont les **indices de position**
des chunks dans l'index construit par `DocumentManager`. Ils ne dérivent pas du
contenu. Toute reconstruction de l'index après ajout, retrait ou renommage d'un
document du corpus **réattribue ces identifiants**.

Sans conséquence dans le flux actuel — le `chunk_id` n'est utilisé qu'à
l'intérieur d'une même exécution, notamment par le Verifier pour
`source_chunk_id`. À ne pas persister entre deux constructions d'index, ni
utiliser comme clé de cache.

---

## 5. Récapitulatif

| # | Point | Écart ? | Suite |
|---|---|---|---|
| 1 | Plafond de résultats (5 vs 5) | **Non** | Aucune. Garde-fou en place par test |
| 2 | Filtres — seul `source` reconnu | Latent | Aucune ; documenté |
| 3 | Codes HTTP ↔ exceptions typées | Non | Aucune ; 422 inatteignable sauf divergence |
| 4.1 | `relevance_score` non comparable | **Oui** | Corriger `critic_spec.md §2.1` ; à arbitrer avec Ihssane |
| 4.2 | Journalisation `ERROR` sur panne banale | Mineur | Corriger `nodes.py` en mission séparée |
| 4.3 | `chunk_id` non stable entre index | Latent | Ne pas persister |

Les points 1, 2, 3 et 4.3 ne demandent aucune action. Les points **4.1** et
**4.2** appellent une décision : le premier de vous deux, le second d'une
mission de correction distincte.
