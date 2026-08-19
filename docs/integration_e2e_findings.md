# Intégration de bout en bout — constats du Sprint I4

**Date des mesures** : 18 août 2026
**Branche** : `integration/action-module`
**Configuration** : Ollama local (`qwen2.5:3b` Analyzer, `qwen2.5:7b`
Planner/Critic/Verifier) · module ACTION `astraexec-integration` sur
`localhost:8000` · corpus HotpotQA distractor, 1966 articles, 3239 chunks.

Ce sprint **constate**. Aucun défaut relevé ici n'a été corrigé : les
correctifs relèveront de missions séparées, après arbitrage.

Les traces complètes sont reproductibles par
`tests/integration/test_end_to_end.py`, qui fige chacun de ces constats sous
forme d'assertion — chaque test échouera le jour où le comportement changera,
ce qui signalera que le constat est levé.

---

## Vue d'ensemble des quatre exécutions

| Palier | Question | Classe | Budget | Chemin | Total |
|---|---|---|---|---|---|
| **1** | `5a75e05c` — Brown State Fishing Lake | SIMPLE | 1 | 6 nœuds, 1 retrieval | 256 s |
| **2/3** | `5a8c7595` — Corliss Archer | MULTI_HOP | 3 | 10 nœuds, **3 retrievals** | 423 s |
| **4.2** | question inventée (Zorblatt) | SIMPLE | 1 | 6 nœuds, 1 retrieval | 185 s |
| **4.1** | `5a8c7595`, module ACTION arrêté | MULTI_HOP | 3 | 10 nœuds, 3 retrievals à vide | 45 s |

Les quatre exécutions atteignent `verify` puis `END`. **Aucune boucle infinie,
aucune exception propagée, aucune hallucination.** Le squelette du pipeline
fonctionne. Ce sont ses mécanismes de rattrapage qui posent problème.

---

## §1 — La boucle de rétroaction est inatteignable pour les requêtes SIMPLE

**Symptôme.** Palier 1 : le Critic rejette explicitement le contexte —
`is_sufficient=False`, `relevance_score=0.0`, `missing_aspects=['population']`
— et pourtant l'exécution part directement en `generate_answer`. Aucune
relance.

**Cause.** `ReasoningPolicy.route_after_critique`
(`src/reasoning/graph/policy.py:106`) évalue la garde globale **avant** le
verdict du Critic :

```python
if feedback_loop_count >= reasoning_budget:
    return CritiqueDecision(route=ROUTE_GENERATE_ANSWER, advance_step=True)
if is_sufficient:
    ...
```

Le nœud `critique` incrémente `feedback_loop_count` **avant** d'appeler la
politique (`nodes.py:346`). À la première critique d'une requête SIMPLE, le
compteur vaut donc déjà 1 et le budget vaut 1 : `1 >= 1`. La garde tranche,
quel que soit le verdict.

**Portée.** Toute requête classée SIMPLE. Le verdict du Critic y est
structurellement sans effet — l'appel LLM du Critic (81 s au palier 1) est
consommé pour un résultat qui ne peut rien changer.

Ce n'est pas un écart à la spécification : `graph_spec.md §3` pose bien que
la garde globale est prioritaire. C'est le **choix de `reasoning_budget = 1`
pour SIMPLE** qui rend la boucle inopérante. La question est de savoir si
c'était l'intention.

---

## §2 — ~~CONSTAT MAJEUR~~ : la relance rapportait exactement les mêmes chunks

> ### ✅ CORRIGÉ — Lot A, 19 août 2026
>
> Le nœud `retrieve` enrichit désormais la sous-requête d'une **relance** avec
> les `missing_aspects` du dernier verdict du Critic (`nodes.py`,
> `enrich_sub_query`). La première tentative d'une étape reste inchangée.
>
> **Mesure après correction**, même question, même corpus :
>
> | Passage | Sous-requête envoyée | Chunks nouveaux vs passage 1 |
> |---|---|---|
> | 1 (1re tentative) | `Who portrayed Corliss Archer in the film Kiss and Tell?` | — |
> | 2 (relance 1) | `… Kiss and Tell? Corliss Archer's portrayal in Kiss and Tell` | **1 sur 5** |
> | 3 (relance 2) | `… Kiss and Tell? Corliss_Archer_portrayer` | 0 sur 5 |
>
> Le mécanisme est **opérant**. Deux réserves mesurées, qui n'invalident pas la
> correction mais en bornent la portée :
>
> 1. **Le chunk gagné est moins bon que celui qu'il évince.** Le passage 2
>    gagne `Kiss_(Carly_Rae_Jepsen_album).txt` et perd `Janet_Waldo.txt` —
>    lequel est, lui, pertinent. Sur cette question, les `missing_aspects`
>    du Critic ne contiennent quasiment aucun terme absent de la requête
>    d'origine : le mécanisme fonctionne, mais il n'a ici aucun signal utile
>    à propager.
> 2. **Le Critic émet parfois des aspects à underscores** —
>    `Corliss_Archer_portrayer`, `Corliss_Archer_portrayer_in_Kiss_and_Tell` —
>    qui ne correspondent à aucun terme du corpus et sont donc inertes pour un
>    index lexical. C'est ce qui explique le passage 3 identique au passage 1.
>    Relève du prompt du Critic, hors périmètre du Lot A.
>
> La stratégie d'enrichissement retenue (expansion de requête par rétroaction
> de pertinence, `missing_aspects` verbatim + quelques mots de contenu du
> `feedback`, plafonnée à 300 caractères) est documentée en tête de `nodes.py`,
> avec la mesure comparative des quatre variantes testées.
>
> Le test `test_relaunch_returns_different_chunks` fige le nouveau
> comportement.

Le constat d'origine, tel qu'établi au Sprint I4, est conservé ci-dessous :
c'est lui qui justifie la correction.

C'est le constat central de ce sprint, et celui que le palier 3 devait établir.

**Symptôme.** Palier 2 : trois retrievals successifs sur `step_1`, et
**15 chunks accumulés = 5 chunks identiques × 3**. Même `chunk_id`, même
`source`, même `relevance_score`, dans le même ordre :

```
0.92271  A_Kiss_for_Corliss.txt                (chunk 202)
0.90085  Kiss_and_Tell_(1945_film).txt         (chunk 1675)
0.60179  Meet_Corliss_Archer_(TV_series).txt   (chunk 1996)
0.54412  Meet_Corliss_Archer.txt               (chunk 1995)
0.41707  Janet_Waldo.txt                       (chunk 1518)
   ← puis exactement la même liste, deux fois de plus
```

Les trois évaluations du Critic sont elles aussi rigoureusement identiques,
au caractère près :

```
#1 #2 #3  is_sufficient=False  relevance_score=0.6
          missing_aspects : ["Corliss Archer's portrayal in Kiss and Tell"]
          feedback        : 'The context does not mention who portrayed
                             Corliss Archer in "Kiss and Tell". More
                             information is needed.'
```

**Cause — deux mécanismes manquants, tous deux vérifiés dans le code.**

1. **La sous-requête n'est jamais réécrite.** `sub_query=step.sub_query`
   (`src/reasoning/graph/nodes.py:261`) est la **seule** occurrence de
   `sub_query` dans tout `nodes.py`. Le nœud `retrieve` renvoie donc la
   requête d'origine, inchangée, à chaque tentative.

2. **Le feedback du Critic n'est lu par personne.** Un `grep` sur
   `src/reasoning/` ne trouve aucune lecture de `CriticEvaluation.feedback`
   en dehors de sa production. Le champ est rempli avec soin par le LLM —
   `critic_spec.md §4` lui consacre une section entière sur la qualité
   attendue du message — puis **jeté**.

Le moteur `fusion_search` étant déterministe, une requête identique produit
un résultat identique. La boucle re-consomme donc le budget et le temps LLM
sans aucune chance d'améliorer quoi que ce soit.

**Coût mesuré.** Palier 2 : deux relances inutiles, soit **2 × (2,6 s de
retrieval + 12,7 s de critique) ≈ 31 s** consommés pour un résultat
identique — et, surtout, **tout le budget de la requête** (voir §4).

**Ce n'est pas une régression** : le mécanisme n'a jamais existé. Les tests
d'intégration du Sprint 6.3 validaient la boucle avec un
`FakeRetrievalClient` retournant une réponse fixe — un double qui, par
construction, ne pouvait pas révéler que la relance était vaine.

### §2 bis — Le retrieval avait pourtant trouvé l'entité pont

Détail relevé dans le contenu des chunks du palier 2, et qui nuance le
diagnostic. Le premier chunk retourné, `A_Kiss_for_Corliss.txt`, contient :

> « A Kiss for Corliss is a 1949 American comedy film […] **It stars Shirley
> Temple** in her final starring role as well as her final film appearance.
> **It is a sequel to the 1945 film "Kiss and Tell"** […] »

L'entité pont recherchée par `step_1` — la femme ayant interprété Corliss
Archer dans *Kiss and Tell* — **est donc présente dans le contexte récupéré
dès la première passe**. Le rapprochement demande une inférence en deux temps
(ce film de 1949 est une suite de *Kiss and Tell* ; Shirley Temple y tient le
rôle principal), mais l'information y est.

Le Critic a néanmoins rejeté, trois fois, avec :

> « The context does not mention who portrayed Corliss Archer in "Kiss and
> Tell". »

L'affirmation est **littéralement exacte** — aucun chunk ne l'énonce
directement — et **fonctionnellement fausse** : l'information est déductible.
`relevance_score = 0.6`, sous le seuil de 0,70, à 0,10 près.

Cela déplace en partie la charge du §2 : même une relance capable de reformuler
la sous-requête n'aurait pas aidé ici, puisque le bon contexte était déjà là.
Le blocage vient d'un Critic qui exige une correspondance littérale là où le
corpus n'offre qu'une chaîne d'inférence — comportement attendu d'un modèle 7B,
et cohérent avec `critic_spec.md §2.1` qui demande au LLM de vérifier que les
chunks « répondent directement » à la sous-requête.

Ce cas unique ne permet pas de généraliser. Il justifie en revanche de ne pas
attribuer au seul §2 l'échec du palier 2.

---

## §3 — La seconde sous-requête ne porte pas l'entité résolue

**Symptôme.** Palier 2, plan généré pour « What government position was held
by the woman who portrayed Corliss Archer in the film Kiss and Tell? » :

```
[step_1] depends_on=[]          sub_query : Who portrayed Corliss Archer in the film Kiss and Tell?
[step_2] depends_on=['step_1']  sub_query : What government position did the identified woman hold?
```

La décomposition est **correcte et pertinente** — c'est exactement le bon
découpage bridge. Mais `step_2` désigne sa cible par une périphrase, « the
identified woman ». Envoyée telle quelle à un moteur lexical, cette requête ne
peut rien retrouver d'utile.

**Cause.** Le Planner produit **toutes** les sous-requêtes en une seule passe,
avant toute exécution. Il ne peut pas connaître le résultat de `step_1`. Et
aucun mécanisme de substitution n'existe entre `critique` et `retrieve` : le
nœud `retrieve` lit `step.sub_query` sans transformation.

`depends_on` et `dependencies_graph` sont donc renseignés correctement, mais
**ne servent qu'à ordonner** les étapes. La dépendance de *données* — la
sortie de l'étape 1 alimentant la requête de l'étape 2 — n'est implémentée
nulle part.

**Conséquence.** Le multi-hop est décomposé mais pas *chaîné*. Sur ce corpus,
le second saut ne peut aboutir.

---

## §4 — ~~Les relances affament les étapes suivantes du plan~~

> ### ✅ CORRIGÉ — Lot A, 19 août 2026
>
> `ReasoningPolicy.route_after_critique` fait désormais primer la garde locale
> `max_retries` sur la garde globale **dans le seul cas où elle fait progresser
> le plan** :
>
> ```python
> if not is_sufficient and retry_count >= max_retries and has_next_step:
>     return CritiqueDecision(route=ROUTE_RETRIEVE, advance_step=True)
> ```
>
> **Mesure après correction**, même question :
>
> | | Avant | Après |
> |---|---|---|
> | Étapes exécutées | `['step_1']` | **`['step_1', 'step_2']`** |
> | Étapes jamais tentées | **`['step_2']`** | `[]` |
> | Passages de retrieval | 3 | 4 |
>
> ### La borne du travail total a changé — décision actée
>
> Cette correction a une conséquence assumée : **`reasoning_budget` n'est plus
> la borne supérieure du travail total.** La nouvelle borne est
>
>     len(plan.steps) × (1 + max_retries)
>
> Les deux exigences initiales — « qu'une étape ne puisse pas affamer les
> suivantes » et « le budget global reste la borne absolue » — sont
> **incompatibles** dès qu'un plan de 2 étapes reçoit un budget de 3 avec
> `max_retries = 2` : le pire cas réclame 6 critiques. Le premier a été
> privilégié, la nouvelle borne validée en arbitrage.
>
> **La terminaison reste garantie** : la branche d'avancement porte
> `advance_step=True`, qui retire une étape de `pending_step_ids`. Elle ne peut
> donc se déclencher qu'un nombre de fois borné par la longueur du plan, finie
> et fixée par le Planner. Vérifié par 100 combinaisons paramétrées de
> (`n_steps` × `budget` × `max_retries`) dans
> `tests/unit/test_feedback_loop.py`.
>
> Trois tests d'intégration ont vu leurs assertions mises à jour en
> conséquence — `test_loop_is_bounded_by_plan_length_and_retries`,
> `test_budget_exhaustion_exits_cleanly`,
> `test_action_module_down_degrades_cleanly` —, la borne étant calculée en un
> seul endroit (`_loop_upper_bound`) à partir de `Critic().max_retries` réel,
> pour qu'elle ne puisse pas dériver silencieusement.

Le constat d'origine, tel qu'établi au Sprint I4 :

**Symptôme.** Palier 2 : le plan comporte 2 étapes. Seul `step_1` a été
exécuté — trois fois. **`step_2` n'a jamais été tenté.** Les trois évaluations
du Critic portent toutes sur `step_1`.

**Cause.** `reasoning_budget` est un compteur **unique** couvrant à la fois le
nombre d'étapes du plan et le nombre de relances. Avec un budget de 3 :

| Critique | `feedback_loop_count` | Décision |
|---|---|---|
| #1 sur `step_1` | 1 < 3, insuffisant, `retry_count`=0 < 2 | relance `step_1` |
| #2 sur `step_1` | 2 < 3, insuffisant, `retry_count`=1 < 2 | relance `step_1` |
| #3 sur `step_1` | **3 ≥ 3** → garde globale | `generate_answer` |

Le budget est intégralement consommé par la première étape. La garde locale
`max_retries = 2` (`critic.py:92`) n'a jamais eu l'occasion de s'appliquer :
la garde globale l'a devancée.

Combiné au §2 — les relances étant sans effet — le résultat est le pire des
deux mondes : **le budget est dépensé en relances stériles, et le plan
multi-hop n'aboutit jamais**.

**Note de lecture.** `retry_counts` apparaît vide (`{}`) dans l'état final.
Ce n'est pas un défaut : la dernière décision porte `advance_step=True`, qui
retire l'entrée (`nodes.py:377`). Le compteur a bien fonctionné pendant la
boucle.

---

## §5 — `is_grounded` ne distingue pas « pas de réponse » de « réponse fondée »

**Symptôme.** Deux situations où le système refuse légitimement de répondre,
avec deux verdicts opposés du Verifier :

| Situation | Chunks | `is_grounded` | `faithfulness_score` | `unsupported_claims` |
|---|---|---|---|---|
| **4.2** — question inventée, chunks hors-sujet récupérés | 5 | **True** | 1.0 | `[]` |
| **4.1** — module ACTION arrêté, aucun chunk | 0 | **False** | 0.0 | `['no sources available for verification']` |

Dans les deux cas la réponse produite est un refus explicite (« The provided
context does not contain… »).

**Ce n'est pas un défaut du Verifier.** Sa mission est la *fidélité aux
sources*, et un refus de répondre **est** fidèle aux sources : il n'avance
aucune affirmation qu'elles ne soutiennent pas. `is_grounded=True` est donc
sémantiquement correct au palier 4.2.

**C'est un écart d'usage à signaler.** `is_grounded` ne peut pas servir à
détecter « le système n'a pas trouvé la réponse ». Un consommateur en aval —
tableau de bord, métrique d'évaluation, logique de reprise — qui l'utiliserait
ainsi conclurait à tort que le palier 4.2 s'est bien passé.

Le signal utilisable pour « pas de réponse » est le verdict du Critic
(`is_sufficient=False` sur toutes les étapes), pas celui du Verifier.

---

## §6 — L'Analyzer classe SIMPLE des questions bridge

**Symptôme.** Palier 1 : `5a75e05c` — « Brown State Fishing Lake is in a
country that has a population of how many inhabitants? » — est de type
`bridge` dans HotpotQA (deux articles gold : `Brown_State_Fishing_Lake.txt`
et `Brown_County,_Kansas.txt`). L'Analyzer la classe **SIMPLE avec une
confiance de 0,95**.

**Conséquence en cascade.** Budget 1 → plan à une seule étape → un seul
retrieval avec la question complète comme sous-requête → le second article
gold n'est jamais cherché (couverture gold **1/2**) → le Critic rejette à
juste titre → mais §1 interdit toute relance → réponse de refus.

Le refus est **honnête et correct** compte tenu du contexte disponible. La
chaîne causale part néanmoins d'une classification erronée.

**Mesure de cadrage.** Sur un échantillon de 25 questions du jeu d'évaluation
passant par le chemin LLM (aucun motif regex déclenché), la répartition
observée est : MULTI_HOP 11, COMPARATIVE 9, SIMPLE 3, AMBIGUOUS 2. Les 200
questions du jeu étant toutes bridge ou comparison, **les 3 classées SIMPLE
sont des erreurs par construction**. Cela ne mesure pas la qualité générale
de l'Analyzer — HotpotQA ne contient aucune question réellement simple, le
taux d'erreur sur SIMPLE y est donc surestimé — mais établit que le cas se
produit.

---

## §7 — Réserve ouverte du Sprint I3 : le score affiché au Critic

**Rappel.** `Critic._format_chunks` (`critic.py:325`) compose chaque chunk
sous la forme `[Chunk 1 — source.txt | score=0.96]: contenu…`. Le
`relevance_score` du moteur est donc **présent dans le prompt**, alors qu'il
n'est qu'un rang normalisé intra-requête (Sprint I3, §4.1). La réserve
ouverte était : le LLM du Critic s'y laisse-t-il prendre ?

**Éléments factuels recueillis dans les paliers 1 à 4.**

| Palier | Meilleur score moteur affiché | Contexte réellement pertinent ? | Verdict du Critic |
|---|---|---|---|
| 1 | 0,96000 (`Brown_State_Fishing_Lake.txt`) | partiellement — l'article gold est là, l'information demandée non | `is_sufficient=False`, score **0,0** |
| 2 | 0,92271 (`A_Kiss_for_Corliss.txt`) | non — mauvais film | `is_sufficient=False`, score **0,6** |
| 4.2 | 0,86164 (`Enrico_Fermi.txt`) | non — totalement hors-sujet | `is_sufficient=False`, score **0,0** |

**Observation.** Dans les trois cas mesurés, **le Critic a rejeté**, avec des
`missing_aspects` précis et manifestement dérivés du contenu textuel, non des
scores. Le cas le plus net est le palier 4.2 : le meilleur chunk affiche
`score=0.86` et parle d'Enrico Fermi face à une question sur une entreprise
fictive ; le Critic attribue `0.0` et liste les cinq aspects manquants un par
un.

**Ce que cela établit — et ce que cela n'établit pas.** Aucune des exécutions
observées ne montre le Critic acceptant un contexte hors-sujet, ni justifiant
une décision par les scores. **La réserve n'est confirmée par aucune mesure.**

Elle n'est pas non plus réfutée : quatre exécutions, dont **aucune** ne
comporte de verdict `is_sufficient=True`, ne constituent pas un échantillon
permettant de conclure. Le biais redouté — être poussé vers la suffisance —
ne peut se manifester que sur des cas limites, où le contexte est
partiellement pertinent. Aucun tel cas n'a été rencontré.

### §7 bis — Le cas limite manquant, observé après le Lot A

Le Sprint I4 concluait que le biais redouté « ne peut se manifester que sur des
cas limites, où le contexte est partiellement pertinent. Aucun tel cas n'a été
rencontré ». **Le Lot A en a produit un**, sans le chercher.

En rendant `step_2` atteignable (§4), l'exécution a lancé pour la première
fois un retrieval sur la sous-requête générique de la seconde étape :

> `What government position did the identified woman hold?`

Cette requête — qui ne porte pas l'entité résolue, faute du mécanisme du §3 —
a ramené cinq articles génériques sur les fonctions publiques :

```
1314  Hannah_Gale.txt
3012  Transition_House_Association_of_Nova_Scotia.txt
1903  Lord_High_Treasurer.txt
3119  Village_accountant.txt
2573  Secretary_of_State_for_Constitutional_Affairs.txt
```

Aucun n'a le moindre rapport avec Corliss Archer, *Kiss and Tell* ou Shirley
Temple. **Le Critic a pourtant accepté :**

```
#4  step=step_2  is_sufficient=True  relevance_score=0.85
    missing_aspects : []
    feedback : 'The context is mostly relevant but lacks specific details
                about the government position held by Hannah Gale.
                More information about her role would be beneficial.'
```

**C'est le premier `is_sufficient=True` observé sur du retrieval réel dans
tout le projet**, et il porte sur un contexte manifestement hors-sujet.

Deux choses méritent d'être relevées séparément :

1. **Le verdict est faux.** Score 0,85, bien au-dessus du seuil de 0,70, sur
   un contexte sans rapport avec la question.
2. **Le feedback révèle une substitution d'entité.** Le Critic écrit « the
   government position held by **Hannah Gale** » — il a pris la première
   entité du contexte pour la cible de la question. La sous-requête disait
   « the identified woman » sans nommer personne ; le modèle a comblé le vide
   avec ce qu'il avait sous les yeux.

**Ce que cette observation établit — et ce qu'elle n'établit pas.** Elle
fournit enfin le cas limite qui manquait : un contexte hors-sujet accepté, avec
un score moteur élevé affiché dans le prompt. Elle est **compatible** avec
l'hypothèse d'un biais induit par ce score.

Elle ne la **démontre pas**. Une explication au moins aussi plausible tient
sans invoquer le score : la sous-requête « What government position did the
identified woman hold? » est, prise isolément, correctement satisfaite par des
articles sur des fonctions gouvernementales. Le Critic évalue une étape hors du
contexte de la question globale — il n'a aucun moyen de savoir de quelle femme
il s'agit. Le défaut visible ici pourrait donc être celui du §3 (l'entité non
propagée), pas celui du §7.

**Les deux hypothèses ne se départagent pas sur ce seul cas.** L'expérience qui
les sépare reste la même : rejouer le jeu d'évaluation avec et sans le score
affiché dans le prompt du Critic, et comparer les taux d'acceptation. Une
seconde expérience, désormais motivée, consisterait à corriger le §3 d'abord
puis à réobserver ce cas — si le verdict devient correct une fois l'entité
propagée, le §7 n'y était pour rien.

**Aucune correction n'a été apportée à ce sujet.** Le score reste affiché dans
le prompt, le seuil reste à 0,70.

**Recommandation inchangée** : trancher par une évaluation comparative sur le
jeu complet (avec et sans score affiché), et non sur ces quelques points.

---

## §8 — Profil de latence

Latences par nœud, en millisecondes :

| Nœud | Palier 1 | Palier 2 | Palier 4.2 | Palier 4.1 (dégradé) |
|---|---|---|---|---|
| `analyze_query` | 9 782 | **7** | 24 182 | **5** |
| `plan` | 1 | 40 292 | 1 | 16 684 |
| `retrieve` (chacun) | 2 496 | 2 533 / 2 762 / 2 505 | 2 985 | 4 339 / 4 402 / 4 360 |
| `critique` (chacun) | 81 484 | 42 153 / 12 069 / 13 292 | 55 345 | 1 / 3 / 1 |
| `generate_answer` | 44 111 | 102 508 | 41 804 | 14 961 |
| `verify` | 118 236 | 205 093 | 61 073 | 2 |
| **TOTAL** | **256 110** | **423 216** | **185 391** | **44 759** |

**Ce que ces chiffres disent.**

- **Le retrieval n'est pas le goulot d'étranglement.** Le module ACTION
  répond en 2,5–3 s de façon très stable, soit **moins de 2 %** du temps
  total. L'essentiel est consommé par les appels LLM locaux.
- **`verify` est le poste le plus lourd** (118 s et 205 s), devant
  `generate_answer` puis `critique`. Tous trois utilisent `qwen2.5:7b`.
- **Le chemin dégradé est 9 fois plus rapide** (45 s contre 423 s) parce que
  le Critic et le Verifier court-circuitent leur appel LLM en l'absence de
  chunks — `critique` tombe à 1 ms, `verify` à 2 ms. Le repli fail-closed est
  donc aussi une économie.
- `analyze_query` à 5–7 ms signale que le **pré-classificateur regex** a
  tranché sans appel LLM ; à 9,8 s ou 24,2 s, que le LLM a été sollicité.
- Le `plan` à 1 ms des paliers 1 et 4.2 correspond au chemin SIMPLE, où le
  Planner produit une étape unique sans appel LLM.

Ces valeurs sont indicatives : mesurées sur une machine unique, sans
répétition, avec des appels LLM dont la variabilité est de l'ordre de
plusieurs dizaines de pourcents.

---

## §9 — Incidents d'infrastructure observés

Sans lien avec la logique du pipeline, mais rencontrés pendant les mesures et
traités par les replis existants.

**Timeouts Ollama.** Trois occurrences sur l'ensemble de la campagne :

```
Erreur inattendue lors de la classification LLM
(APIConnectionError: OllamaException - litellm.Timeout:
 Connection timed out after 20.0 seconds.) — fallback.
```

L'Analyzer bascule sur son heuristique de repli, avec une confiance abaissée
(0,55 au lieu de 0,85–0,95). Le pipeline continue. Le seuil de 20 s paraît
serré pour un modèle local sous charge.

**Sortie TOON non conforme de l'Analyzer.** Une occurrence :

```
Réponse LLM non parseable TOON (ValidationError: detected_entities
  Input should be a valid list [input_value='Poison:Shut Up, Make Love',
  input_type=str]) — fallback heuristique.
```

Le modèle a produit `detected_entities` comme une chaîne au lieu d'une liste,
sur une question contenant un titre avec deux-points — le séparateur `::` du
format TOON. Repli correct, mais la collision entre la ponctuation des données
et la syntaxe TOON mérite d'être notée.

**Qualité du repli heuristique.** Palier 4.2, après timeout :
`detected_entities = ['what', 'exact', 'catalogue', 'number', 'zorblatt']` —
le repli retient des mots vides comme entités. Sans conséquence ici
(`detected_entities` n'est pas consommé par le retrieval), mais le champ n'est
pas exploitable en l'état.

**`ACTION_BASE_URL` n'est lu qu'une fois, à l'import.** Découvert en écrivant
le test du palier 4.1. `ActionClient.__init__` déclare
`base_url: str = _ACTION_BASE_URL` (`action_client.py:118`) : la valeur par
défaut est évaluée **une seule fois, à la définition de la fonction**, au
moment de l'import du module. Modifier ensuite la variable d'environnement —
ou réassigner `reasoning.action_client._ACTION_BASE_URL` — n'a aucun effet sur
les instances créées après coup.

Sans conséquence en exploitation normale, où l'environnement est fixé avant le
démarrage. Deux implications tout de même : reconfigurer l'adresse du module
ACTION exige un redémarrage du processus, et tout test cherchant à rediriger
le client doit remplacer la fabrique (`reasoning.graph.graph.ActionClient`)
plutôt que la globale. C'est ce que fait
`test_action_module_down_degrades_cleanly` — une première version patchant la
globale passait à côté et atteignait l'API réelle.

---

## §10 — Récapitulatif et suites

| § | Constat | Gravité | Suite proposée |
|---|---|---|---|
| **2** | ~~La relance rapporte des chunks identiques~~ | ✅ **Corrigé (Lot A)** | Enrichissement de la sous-requête par les `missing_aspects` |
| **4** | ~~Les relances consomment le budget des étapes suivantes~~ | ✅ **Corrigé (Lot A)** | Garde locale prioritaire quand elle fait progresser ; nouvelle borne actée |
| **3** | La seconde sous-requête ne porte pas l'entité résolue | **Élevée** | Substituer l'entité résolue entre `critique` et `retrieve` |
| **7 bis** | Contexte hors-sujet accepté (`is_sufficient=True`, 0,85) sur `step_2` | **Élevée** | N'instruit pas le §7 à lui seul — corriger le §3 d'abord, puis réobserver |
| **2 bis** | Le Critic rejette un contexte contenant l'information par inférence | **Élevée** | Mesurer avant d'agir : un seul cas observé |
| **1** | Boucle inatteignable pour SIMPLE | Moyenne | Décider si `reasoning_budget=1` est l'intention |
| **6** | Questions bridge classées SIMPLE | Moyenne | Relève du réglage de l'Analyzer, déjà instrumenté |
| **5** | `is_grounded` ne signale pas « pas de réponse » | Faible | Documenter l'usage ; ne pas s'en servir comme tel |
| **9** | Timeouts Ollama, collision `::` en TOON, `ACTION_BASE_URL` figé à l'import | Faible | Relever le timeout ; échapper les `:` ; documenter |
| **7** | Score moteur affiché au Critic | **Non tranchée** | Évaluation comparative sur le jeu complet |

Les §2 et §4 se composaient : ensemble, ils faisaient que **le mécanisme de
rattrapage du système ne rattrapait rien** et empêchaient par surcroît les
plans multi-étapes d'aboutir. Ils ont été traités conjointement au **Lot A**
(19 août 2026).

**Ce que le Lot A a débloqué, et ce qu'il a révélé.** Le plan multi-hop
s'exécute désormais en entier, et les relances ont un effet réel. Mais rendre
`step_2` atteignable a immédiatement exposé le §3 : la seconde sous-requête,
privée de l'entité résolue, ramène du bruit générique — que le Critic accepte
(§7 bis). **Le §3 est donc le prochain verrou**, et sa correction conditionne
l'interprétation du §7.
