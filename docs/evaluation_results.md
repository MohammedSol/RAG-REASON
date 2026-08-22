# Résultats d'évaluation — module REASONING

**Version 1 — 21 août 2026.** Socle chiffré du rapport de soutenance.

Ce document se lit sans connaissance préalable du projet. Il rassemble
toutes les mesures produites, leur verdict face au cahier des charges, et
les limites qui en bornent la portée.

---

## Ce dont il s'agit, en trois phrases

**RAG-REASON** est le module de *raisonnement* d'un agent de question-réponse
documentaire. Là où un RAG classique enchaîne « une question → une recherche
→ une réponse », ce module classe la question, la décompose en sous-questions,
juge la qualité des passages retrouvés, relance la recherche si nécessaire,
puis vérifie que la réponse produite est bien soutenue par ses sources.

Il s'appuie sur un second module, **ACTION**, qui détient le moteur de
recherche documentaire. La question posée ici est simple : **cette
architecture plus élaborée fait-elle mieux qu'un RAG naïf, et à quel coût ?**

---

## 1. Les KPI du cahier des charges

| # | Indicateur | Cible | Mesuré | Verdict |
|---|---|---|---|---|
| 1 | Gain de *faithfulness* vs RAG naïf | +15 pts (indicatif) | **−1,7 pt** | ❌ **NON ATTEINT** |
| 2 | *Answer relevancy* sur multi-hop | amélioration démontrée | **+22,4 pts** | ✅ Atteint |
| 3 | *Context precision* sur multi-hop | amélioration démontrée | **+2,9 pts** | ⚠️ Marginal (dans le bruit) |
| 4 | Précision / rappel du Verifier | mesuré | P 0,821 · R 0,821 · F1 0,821 *(ensemble)* — mais **P 0,286 sur les cas réels** | ⚠️ Mesuré, voir §1.1 |
| 5 | Plans valides produits par le Planner | ≥ 95 % | **100 %** (25/25) | ✅ Atteint |
| 6 | Plafond configurable d'appels LLM | existe et s'applique | **12**, mesures 4 à 8 | ✅ Atteint |
| 7 | Traçabilité des sorties LLM intermédiaires | complète | **Langfuse**, 12 traces / 23 observations | ✅ Atteint |

### 1.1 Précision et rappel du Verifier

Cette mesure demande une **annotation humaine** : les métriques automatiques
du §2 sont calculées par le même modèle que celui qui a produit les réponses,
ce qui les rend impropres à juger le composant chargé de détecter les
hallucinations.

Un jeu de 50 exemples a été construit à cette fin
(`tests/evaluation/verifier_annotation_set.json`) : 25 réponses réelles issues
des campagnes, et 25 réponses dans lesquelles une affirmation absente des
sources a été injectée délibérément — dont le label est donc connu par
construction.

**Qui a annoté.** L'annotation a été réalisée par un **modèle tiers**
(Claude Opus 5), à la demande du porteur du projet, et non par un humain. Ce
point doit être lu avec attention :

- le biais principal que cette mesure devait lever — *juge et partie*,
  `qwen2.5:7b` évaluant ses propres réponses — **est bien levé** : l'annotateur
  n'est ni le générateur ni le juge RAGAS ;
- mais il s'agit d'un jugement de modèle, **pas d'un jugement humain**. La
  mesure est plus solide que les métriques automatiques du §2, moins solide
  qu'une annotation humaine. Elle reste à refaire par un humain pour être
  pleinement probante.

Les 25 cas synthétiques sont étiquetés **par construction** — les injections
ayant été fabriquées par l'annotateur, leur label ne doit rien au jugement.
La valeur d'appréciation porte donc sur les **25 cas réels**.

#### Résultats — classe positive : `hallucinated`

| Population | Précision | Rappel | F1 | Exactitude |
|---|---:|---:|---:|---:|
| **Ensemble** (n = 50) | **0,821** | **0,821** | **0,821** | 0,800 |
| Cas synthétiques (n = 25) | 1,000 | 0,840 | 0,913 | 0,840 |
| **Cas réels** (n = 25) | **0,286** | **0,667** | **0,400** | 0,760 |

**L'agrégat flatte le Verifier, et il faut le dire.** Les deux populations ne
sont pas comparables : les hallucinations injectées sont franches (une date
fausse, une entité substituée), les cas réels sont subtils. Ne retenir que le
0,821 global reviendrait à masquer le résultat qui compte.

**Sur les hallucinations fabriquées, le Verifier est fiable quand il alerte.**
Précision 1,000 : ses 21 alertes sont toutes justes, aucune fausse. Il en
laisse passer 4 sur 25 (rappel 0,840).

**Sur les cas réels, il crie au loup.** Matrice de confusion :

|  | Verifier : alerte | Verifier : fondée |
|---|---:|---:|
| Annotateur : hallucinée | 2 | 1 |
| Annotateur : fondée | **5** | 17 |

Sept alertes pour trois vraies hallucinations. **Cinq réponses parfaitement
fondées sur seize ont été signalées à tort** — c'est ce qui écrase la
précision à 0,286. En exploitation, un utilisateur verrait cinq
avertissements injustifiés pour deux justifiés, et cesserait vite d'y prêter
attention.

#### Détection par type d'injection

| Type | n | Rappel |
|---|---:|---:|
| Substitution d'entité | 7 | **1,000** |
| Ajout d'un fait absent | 10 | 0,800 |
| Contradiction d'un fait présent | 8 | 0,750 |

Le Verifier repère systématiquement une entité remplacée, et laisse passer un
quart des contradictions. C'est cohérent : substituer « Mondelez » par
« Kraft Heinz » crée une entité introuvable dans les sources, tandis que
contredire une date déjà présente demande de comparer deux valeurs.

#### Contrôle de qualité des injections

La concordance entre `expected_label` et le label d'annotation est de
**25/25 (1,000)**. Aucune injection ratée, aucune erreur d'annotation sur
cette moitié — la validité du protocole est vérifiée.

#### Les trois hallucinations réelles détectées par l'annotation

Elles éclairent des défauts que les métriques automatiques ne voyaient pas :

1. **`5a76387d` (système complet)** — « Am Rong est né en premier ». La
   réponse admet que la date de naissance d'Ava DuVernay est absente du
   contexte, **puis conclut quand même** qui est né en premier. Le baseline,
   sur la même question, avait récupéré la fiche d'Ava DuVernay et conclut
   valablement : c'est un défaut de retrieval qui devient un défaut de
   raisonnement.
2. **`5ab3306a` (système complet)** — refus erroné. Le système déclare que le
   contexte ne donne pas la société de production, alors que
   `The_Great_Locomotive_Chase.txt` dit « 1956 **Walt Disney Productions** ».
   Il refuse alors qu'il avait la réponse sous les yeux.
3. **`5ab611cc` (baseline)** — « ils ne partagent aucune autre profession »,
   alors que les deux sources décrivent Simon **et** Cannon comme
   *producer*. Contradiction directe d'un fait présent.

### 1.2 Détail des KPI 5 à 7

**Plans valides — 100 %.** Sur 25 questions, le Planner a produit 25 plans
syntaxiquement valides (format TOON) et 25 graphes de dépendances acycliques.
À noter : 8 de ces 25 plans (32 %) proviennent du repli heuristique et non du
LLM. Le taux de validité est donc de 100 %, mais un tiers des plans n'a pas
été produit par le modèle.

**Plafond d'appels LLM — 12.** Configurable par `MAX_LLM_CALLS_PER_QUERY`.
Fixé à 1,5 fois le pire cas mesuré : 4 appels pour une question simple,
8 pour une question multi-hop. Au dépassement, le graphe sort proprement vers
la génération de réponse — il ne lève jamais d'exception.

**Traçabilité — Langfuse.** Chaque appel LLM des cinq composants est tracé,
et tous les appels d'une même question sont regroupés sous un identifiant de
session commun. Vérification sur deux questions réelles : 4 traces et
7 observations pour une question simple, 8 traces et 16 observations pour une
question multi-hop.

---

## 2. Les mesures comparatives : système complet contre RAG naïf

### 2.1 Protocole

Deux systèmes ont traité **les mêmes 20 questions**, tirées du jeu HotpotQA
avec une graine fixe et stratifiées 10 *bridge* (à deux sauts) / 10
*comparison* (comparaison de deux entités) :

* **Le baseline RAG naïf** — une recherche, une génération. Aucun analyseur,
  aucun planificateur, aucun critique, aucun vérificateur.
* **Le système complet** — le module REASONING dans son intégralité.

L'équité de la comparaison a été construite explicitement : même corpus, même
moteur de recherche, même `top_k`, même modèle de génération, mêmes
paramètres, et **le même prompt de génération** — importé plutôt que recopié,
pour que l'écart mesuré porte sur l'architecture et non sur une formulation.

### 2.2 Résultats globaux

| Métrique | RAG naïf | Système complet | Écart |
|---|---:|---:|---:|
| `faithfulness` | 0,750 | 0,733 | **−1,7 pt** |
| `answer_relevancy` | 0,509 | 0,672 | **+16,3 pts** |
| `context_precision` | 0,208 | 0,214 | +0,6 pt |
| `context_recall` | 0,750 | 0,867 | **+11,7 pts** |
| Appels LLM par question | 1,0 | 5,55 | ×5,6 |
| Latence moyenne | 37,6 s | 321,6 s | ×8,6 |

### 2.3 Ventilation par type de question

**Bridge — les questions multi-hop**, celles que l'architecture vise :

| Métrique | RAG naïf | Système complet | Écart |
|---|---:|---:|---:|
| `faithfulness` | 0,722 | 0,683 | −3,9 pts |
| `answer_relevancy` | 0,431 | 0,655 | **+22,4 pts** |
| `context_precision` | 0,267 | 0,296 | +2,9 pts |
| `context_recall` | 0,600 | 0,800 | **+20,0 pts** |

**Comparison :**

| Métrique | RAG naïf | Système complet | Écart |
|---|---:|---:|---:|
| `faithfulness` | 0,775 | 0,783 | +0,8 pt |
| `answer_relevancy` | 0,587 | 0,688 | +10,2 pts |
| `context_precision` | 0,150 | 0,133 | −1,7 pt |
| `context_recall` | 0,900 | 0,933 | +3,3 pts |

**Lecture.** Les gains sont concentrés là où l'architecture est censée agir :
sur les questions à deux sauts, le système retrouve nettement mieux
l'information nécessaire (+20 pts de rappel) et répond bien plus à la question
posée (+22,4 pts de pertinence). En revanche, il n'est pas plus fidèle à ses
sources que le baseline.

---

## 3. Verdict formel du Sprint I5-B

Le sprint qui a produit ces mesures comportait huit critères de validation
formels. **Les huit sont remplis :**

| # | Critère | Verdict | Mesure |
|---|---|---|---|
| F1 | Baseline construit, équité documentée | ✅ | 1,0 appel LLM par question |
| F2 | Jeu de données validé | ✅ | 20 entrées, 45 phrases justificatives, sources gold vérifiées |
| F3 | Campagne baseline complète | ✅ | 20/20, 0 échec, 12,5 min |
| F4 | Campagne système complet | ✅ | 20/20, 0 échec, 107 min |
| F5 | Métriques produites | ⚠️ | 4 métriques × 2 systèmes ; 1 `faithfulness` manquante (19/20 côté baseline) |
| F6 | Comparaison exploitable | ✅ | Global + ventilation bridge/comparison, JSON et Markdown |
| F7 | Non-régression | ✅ | 429 tests verts, ruff et mypy strict propres |
| F8 | Moteur figé pendant les campagnes | ✅ | `git diff src/reasoning/` vide, HEAD inchangé |

> ### Le sprint est néanmoins déclaré **NON VALIDÉ**
>
> Les critères formels mesurent la conduite de l'évaluation, pas son
> résultat. Le KPI principal du cahier des charges — un gain de *faithfulness*
> d'environ +15 points — est **manqué**, l'écart mesuré étant de −1,7 point.
>
> Un sprint dont tous les indicateurs de procédure sont au vert mais dont
> l'objet même n'est pas atteint ne peut pas être déclaré validé. La
> distinction est faite ici explicitement plutôt que dissoute dans un tableau
> majoritairement vert.

---

## 4. Pourquoi le gain de fidélité n'est pas au rendez-vous

C'est le point que la soutenance doit expliquer, et il mérite mieux qu'un
constat d'échec.

### 4.1 Le système répond plus souvent au lieu de refuser

Sur les 20 questions :

| | RAG naïf | Système complet |
|---|---:|---:|
| Réponses substantielles | 9 | 10 |
| Refus explicites | 11 | 10 |

Le système complet, ayant de meilleurs passages en main (+11,7 pts de rappel),
**tente une réponse là où le baseline renonce**. Or un refus est trivialement
fidèle : il n'avance aucune affirmation que les sources ne soutiennent pas, et
`faithfulness` le récompense pleinement. Répondre expose ; refuser protège la
métrique.

C'est un **effet de composition** : les deux moyennes ne portent pas sur les
mêmes populations de réponses.

### 4.2 Sur les questions comparables, le système est plus fidèle

Restreint aux **7 questions où les deux systèmes ont produit une réponse
substantielle** — seul périmètre où la comparaison porte sur des objets de
même nature :

| | RAG naïf | Système complet |
|---|---:|---:|
| `faithfulness` moyenne (n = 7) | **0,857** | **1,000** |

Le système complet obtient une fidélité **parfaite** sur ces 7 questions,
contre 0,857 pour le baseline. À périmètre comparable, l'écart s'inverse donc :
**+14,3 points en faveur du système complet**, soit presque exactement la cible
du cahier des charges.

> **Ce chiffre ne doit pas être surinterprété, et voici pourquoi.**
> Le détail par question montre que **6 des 7 sont à 1,00 pour les deux
> systèmes**. L'écart tout entier repose sur **une seule question**
> (`5ac3165c`, « Quel réalisateur américain a présenté la 18e cérémonie des
> Independent Spirit Awards ? »), où le baseline obtient 0,00 et le système
> complet 1,00.
>
> Un écart de +14,3 points porté par un unique point de mesure n'est pas un
> résultat démontré. Il est **compatible** avec l'hypothèse que l'architecture
> améliore la fidélité, sans l'établir. Le mentionner sans cette réserve
> reviendrait à retourner en argument favorable un échantillon trop petit pour
> conclure dans un sens comme dans l'autre.

**Ce que l'on peut affirmer :** l'architecture améliore la **couverture** —
+11,7 points de rappel de contexte globalement, +20,0 sur les questions
multi-hop — et rien dans les mesures n'indique qu'elle dégrade la fidélité par
réponse.

### 4.3 L'écart global est dans le bruit

Sur les 20 questions, **10 obtiennent 1,00 de fidélité pour les deux
systèmes**. L'écart global se joue donc sur une poignée de points. Or la bande
de bruit mesurée sur ce projet est de ±2 points, et les métriques RAGAS sont
calculées par un LLM — doublement non déterministes.

**−1,7 point n'est ni un gain ni une dégradation démontrée : c'est du bruit.**
Les écarts de +16,3 et +20,0, eux, sont hors bande et constituent les
résultats solides de cette évaluation.

---

## 5. Les mesures antérieures — réglage des composants

Avant l'évaluation de bout en bout, chaque composant a été mesuré et réglé
séparément. Ces campagnes portent sur les **200 questions** du jeu
d'évaluation, et non sur l'échantillon de 20.

### 5.1 Bande de bruit — ±2 points

Établie avant tout arbitrage, en rejouant deux fois la même évaluation sans
rien changer. Malgré `temperature=0`, le LLM n'est pas déterministe
(ordonnancement des kernels, batching, cache KV). Mesures : ±0,50 pt sur
l'accuracy globale, ±2,00 pts sur le sous-ensemble bridge.

**Aucune amélioration inférieure à cette bande n'a été revendiquée.**

### 5.2 Analyzer — classification des questions

| Étape | Accuracy globale | Accuracy bridge |
|---|---:|---:|
| Avant réglage | 57,75 % | 30 % |
| Après (Lots 1, 2, 2b) | **75,25 %** *(moyenne 2 exécutions)* | **56,5 %** |
| Dernière exécution versionnée | 75,50 % | 57,00 % |

L'écart entre la moyenne sur deux exécutions et la dernière mesure — 0,25 et
0,5 point — illustre concrètement la bande de bruit du §5.1.

Détail par chemin de décision, sur la dernière exécution : le
**pré-classificateur par expressions régulières** traite 39,5 % des questions
avec **93,67 %** d'exactitude ; le **chemin LLM** traite les 60,5 % restantes
avec **63,64 %**.

**Audit du pré-classificateur (Lot 3).** Pour écarter le soupçon de règles
taillées sur les données, les 200 questions ont été séparées en DEV et TEST
avec graine fixe, les règles n'étant conçues qu'à partir du DEV :

* DEV : **92,59 %**
* TEST : **96,00 %**

L'exactitude est plus élevée sur les données jamais consultées que sur celles
ayant servi à écrire les règles : **aucun surapprentissage**.

### 5.3 Planner — validité des plans

Sur 25 questions : **100 %** de plans au format valide, **100 %** de graphes
de dépendances acycliques. 2,2 étapes par plan en moyenne. 32 % des plans
proviennent du repli heuristique.

---

## 6. Conditions de mesure

| | |
|---|---|
| **Modèles** | `qwen2.5:7b` (planification, critique, génération, vérification) · `qwen2.5:3b` (classification, synthèses intermédiaires) |
| **Serveur d'inférence** | Ollama en local, `temperature=0` |
| **Corpus** | HotpotQA, configuration `distractor`, split `validation` — 1 966 articles, 3 239 passages indexés |
| **Moteur de recherche** | Module ACTION (`astraexec`), recherche hybride TF-IDF + BM25, 5 passages par requête |
| **Échantillon bout en bout** | 20 questions, graine `20260820`, 10 bridge / 10 comparison, articles de référence intégralement présents dans le corpus |
| **Échantillon composants** | 200 questions (Analyzer), 25 (Planner) |
| **Juge des métriques** | `qwen2.5:7b` via RAGAS · embeddings `BAAI/bge-small-en-v1.5` |
| **Annotateur du Verifier** | **Modèle tiers (Claude Opus 5), non humain** — 50 exemples, dont 25 étiquetés par construction |
| **Matériel** | Poste de développement unique, Windows 10, exécution séquentielle |
| **Dates** | Campagnes 20 août 2026 · métriques 21 août 2026 · annotation 22 août 2026 |

**Pourquoi 20 questions et pas 200.** Une campagne du système complet prend
107 minutes pour 20 questions. Les 200 demanderaient environ 18 heures pour un
seul passage — et la non-reproductibilité du LLM en exige plusieurs. Le
plafond est le temps de calcul disponible, pas la méthode.

---

## 7. Limites — ce que ces chiffres ne disent pas

Elles sont énoncées ici plutôt que découvertes en questions.

**1. Le juge est le générateur.** Les métriques RAGAS sont calculées en
interrogeant `qwen2.5:7b` — le modèle qui a produit les réponses évaluées. Un
modèle juge avec indulgence sa propre production : **les valeurs absolues sont
optimistes**. Ce qui reste interprétable est l'**écart** entre les deux
systèmes, le biais leur étant commun. L'annotation du §1.1 lève ce biais
précis, puisque son auteur n'est ni le générateur ni le juge RAGAS.

**1 bis. Mais l'annotation n'est pas humaine.** Le §1.1 devait apporter un
jugement humain ; il a été réalisé par un **modèle tiers**, à la demande du
porteur du projet. La mesure du Verifier se situe donc entre les deux : elle
échappe au biais *juge et partie*, mais reste un jugement de machine. Refaire
cette annotation par un humain est le premier travail à reprendre si les
chiffres du Verifier doivent être défendus. Le jeu et l'interface sont en
place pour le faire sans rien reconstruire.

**2. Échantillon réduit.** 20 questions. Une seule question qui bascule
déplace une moyenne de 5 points. Aucun test de significativité statistique n'a
été conduit — il n'aurait pas de sens à cette taille.

**3. `answer_relevancy` dégradée.** La métrique compare normalement la
question à **trois** reformulations générées depuis la réponse. Ollama
n'acceptant pas le paramètre qui les produit en un appel, elle est calculée
sur **une seule** reformulation. La mesure reste valide mais plus bruitée.
Le biais s'applique aux deux systèmes.

**4. `context_precision` basse pour les deux systèmes** (0,21 et 0,21). Ce
n'est pas un défaut du module : le corpus HotpotQA `distractor` contient
**par construction** 8 passages distracteurs pour 2 passages pertinents. Un
moteur qui remonte 5 passages ne peut structurellement pas dépasser une
précision élevée. Les distracteurs ont été conservés délibérément — les
retirer aurait rendu la recherche triviale et gonflé artificiellement toutes
les métriques.

**5. Un doute méthodologique non tranché.** Le score de pertinence renvoyé par
le moteur de recherche est un **rang normalisé au sein d'une même requête**, et
non une mesure absolue : le meilleur passage obtient toujours une valeur proche
de 1, même sur une question dont le corpus ne contient pas la réponse (mesuré :
0,96 sur une question sans réponse). Or ce score est affiché dans le prompt du
Critic. Peut-il induire le composant en erreur ? Les observations recueillies
ne permettent ni de l'affirmer ni de l'exclure. Trancher demanderait une
évaluation comparative avec et sans ce score affiché.

**6. Deux défauts documentés et non traités.** Le verdict `is_grounded` du
Verifier ne distingue pas « réponse fondée » de « pas de réponse trouvée » — un
refus est fidèle à ses sources, donc jugé fondé. Et le Critic exige parfois une
correspondance littérale là où le corpus n'offre qu'une chaîne d'inférence,
rejetant un contexte qui contient pourtant l'information. Les deux sont
consignés dans `docs/integration_e2e_findings.md`.

---

## 8. Anomalies rencontrées pendant les campagnes

Consignées pour la reproductibilité, et parce qu'elles ont coûté du temps.

**1. Blocage silencieux de RAGAS en mode parallèle.** Lancé avec quatre
travailleurs et une barre de progression, le calcul des métriques s'est figé :
plus aucune requête n'atteignait le serveur d'inférence, et le processus ne
consommait plus de CPU — sans erreur ni trace. **Cinquante minutes perdues**
avant que le diagnostic ne soit posé, l'horodatage d'expiration du modèle
chargé côté Ollama étant resté figé. Contourné en calculant question par
question, en séquentiel, avec sauvegarde après chacune.

**2. Deux métriques initialement non calculées.** `answer_relevancy` échouait
pour deux raisons superposées : Ollama refuse le paramètre `n`, et la
télémétrie de RAGAS attend un nom de modèle d'embeddings sous forme de chaîne
là où elle recevait l'objet. `context_precision` était bien calculée mais
jamais lue : RAGAS nomme sa colonne d'après la classe de métrique
(`llm_context_precision_with_reference`) et non d'après le nom usuel.

**3. Une valeur de `faithfulness` manquante.** Question `5adf37a9`, côté
baseline, sans message d'erreur enregistré. La métrique globale du baseline
porte donc sur 19 questions au lieu de 20 — écart signalé plutôt que masqué.
Cause non élucidée.

---

## 9. Où retrouver les données

| Fichier | Contenu |
|---|---|
| `tests/evaluation/dataset_v1.json` | Les 20 questions et leur vérité terrain |
| `tests/evaluation/results/run_naive.json` | Campagne baseline, réponse et contextes par question |
| `tests/evaluation/results/run_full.json` | Campagne système complet, avec plan, verdicts et traces |
| `tests/evaluation/reports/comparison_v1.md` | Tableau comparatif des métriques |
| `tests/evaluation/reports/metrics_*.json` | Scores par question |
| `tests/evaluation/verifier_annotation_set.json` | Les 50 exemples destinés à l'annotation humaine |
| `configs/evaluation_thresholds.toml` | Planchers de non-régression, avec leur justification |
| `data/evaluation/*.json` | Campagnes Analyzer et Planner sur 200 questions |
| `docs/integration_e2e_findings.md` | Défauts constatés au cours de l'intégration |

Toutes ces mesures sont **versionnées** dans le dépôt : le LLM n'étant pas
déterministe, les rejouer ne les reproduirait pas à l'identique.
