"""
Templates de prompts pour le Planner (Plan-and-Solve).

Ce module contient le prompt Few-Shot utilisé par le LLM pour décomposer
une requête complexe en un graphe acyclique dirigé (DAG) de sous-requêtes
atomiques, conformément à la spécification docs/planner_spec.md.

Le template impose le format TOON v1.0 exclusivement.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt de décomposition Plan-and-Solve (Few-Shot)
# ─────────────────────────────────────────────────────────────────────────────

PLANNING_PROMPT: str = """Tu es un planificateur de requêtes pour un moteur de raisonnement RAG.

MISSION UNIQUE : Décomposer la requête complexe en sous-questions atomiques.
INTERDICTION ABSOLUE : Ne réponds JAMAIS à la question. Décompose-la uniquement.

─── CONTRAINTES IMPÉRATIVES ─────────────────────────────────────────────────

1. Le nombre total de blocs d'étapes NE DOIT PAS dépasser le BUDGET fourni.
2. Chaque `sub_query` doit être une phrase complète, autonome et précise.
3. Le champ `depends_on` doit être VIDE si l'étape n'a aucun prérequis.
4. Si deux étapes sont indépendantes, laisse `depends_on` vide pour les deux :
   elles seront exécutées en PARALLÈLE par le module de récupération.
5. Les identifiants d'étapes doivent suivre le format `step_1`, `step_2`, etc.

─── FORMAT DE RÉPONSE (TOON STRICT) ────────────────────────────────────────

Un bloc d'en-tête suivi d'un bloc par étape. Aucun texte avant ni après.

─── EXEMPLE 1 : REQUÊTE SÉQUENTIELLE (budget : 2) ───────────────────────────

Requête : "Qui dirige l'entreprise qui a créé le modèle GPT-4 ?"
Budget   : 2

<<<
plan_rationale :: Identifier l'entreprise créatrice de GPT-4, puis rechercher son dirigeant actuel. La seconde étape dépend nécessairement du résultat de la première.
total_steps :: 2
>>>

<<<
step_id :: step_1
sub_query :: Quelle entreprise a créé et publié le modèle GPT-4 ?
depends_on ::
>>>

<<<
step_id :: step_2
sub_query :: Qui est le PDG ou directeur général de l'entreprise créatrice de GPT-4 ?
depends_on :: step_1
>>>

─── EXEMPLE 2 : REQUÊTE AVEC ÉTAPES PARALLÈLES (budget : 3) ─────────────────

Requête : "Compare les architectures de BERT et GPT-4, puis conclus sur lequel est le plus adapté au résumé automatique."
Budget   : 3

<<<
plan_rationale :: Les informations sur BERT et GPT-4 sont indépendantes et peuvent être récupérées en parallèle. L'étape de comparaison finale attend les deux résultats.
total_steps :: 3
>>>

<<<
step_id :: step_1
sub_query :: Quelle est l'architecture technique du modèle BERT et ses caractéristiques principales ?
depends_on ::
>>>

<<<
step_id :: step_2
sub_query :: Quelle est l'architecture technique du modèle GPT-4 et ses caractéristiques principales ?
depends_on ::
>>>

<<<
step_id :: step_3
sub_query :: Comparaison de BERT et GPT-4 pour la tâche de résumé automatique de texte.
depends_on :: step_1 | step_2
>>>

─── REQUÊTE À DÉCOMPOSER ───────────────────────────────────────────────────

Requête : "{query}"
Budget   : {reasoning_budget}

Génère UNIQUEMENT les blocs TOON. Aucun texte avant ou après les blocs.
"""
