# Tests de contrat croisés REASONING × ACTION — Sprint I3

Ce répertoire contient les **fixtures JSON partagées** entre les deux modules du
projet, et les tests de conformité côté **consommateur** (REASONING).

## Pourquoi des fixtures partagées

Les deux modules définissent chacun leur copie du contrat Pydantic :

| Module | Fichier |
|---|---|
| REASONING (consommateur) | `src/reasoning/contracts/action_interface.py` |
| ACTION (fournisseur) | `app/schemas/retrieval_contract.py` (dépôt `astraexec-integration`) |

Ces deux copies sont aujourd'hui identiques champ pour champ, mais rien ne le
garantit dans le temps : elles vivent dans deux dépôts, modifiés par deux
personnes. Un test de bout en bout finirait par détecter une divergence, mais
tard, et avec un diagnostic confus (« la réponse est vide » pouvant signifier
aussi bien « rien trouvé » que « le champ a été renommé »).

Les fixtures ci-dessous sont donc des **documents JSON figés**, chargés à
l'identique des deux côtés et validés contre la copie locale du contrat. Une
divergence de schéma fait échouer un test unitaire rapide, du côté qui a
divergé, avec le nom du champ fautif.

## Fixtures

| Fichier | Rôle |
|---|---|
| `retrieval_request_nominal.json` | Requête standard, `top_k` égal au plafond du moteur |
| `retrieval_request_with_filters.json` | `filters` renseignés — dont une clé **non reconnue** par le fournisseur, pour documenter son traitement |
| `retrieval_request_top_k_high.json` | `top_k = 20`, supérieur au plafond du moteur (5) — documente le comportement réel, ne le corrige pas |
| `retrieval_response_nominal.json` | Réponse à 3 chunks, valeurs relevées sur un appel réel à l'API |
| `retrieval_response_empty.json` | Aucun chunk — cas métier valide, pas une panne |

Les `chunk_id`, `source`, `content` et `relevance_score` des fixtures de réponse
sont **repris verbatim d'un appel réel** à `POST /retrieve` sur le corpus
HotpotQA distractor indexé au Sprint I1. Les `source` correspondent donc à des
fichiers réellement présents dans le corpus (`Scott_Derrickson.txt`,
`Ed_Wood.txt`, `Ed_Wood_(film).txt`).

## Consommation côté fournisseur

Le dépôt `astraexec-integration` lit **ces mêmes fichiers**. Il localise ce
répertoire dans cet ordre :

1. la variable d'environnement `REASONING_CONTRACT_FIXTURES`, si elle est
   définie ;
2. sinon, le chemin frère `../RAG-REASON/tests/contracts/fixtures`.

Si aucun des deux ne résout, le test échoue explicitement en nommant la
variable à définir — il n'est jamais ignoré en silence, ce qui masquerait
précisément la divergence qu'il est censé détecter.

## Exécution

```bash
uv run pytest tests/contracts/ -v
```
