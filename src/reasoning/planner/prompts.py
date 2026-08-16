"""
Templates de prompts pour le Planner (Plan-and-Solve).

Ce module contient le prompt Few-Shot utilisé par le LLM pour décomposer
une requête complexe en un graphe acyclique dirigé (DAG) de sous-requêtes
atomiques, conformément à la spécification docs/planner_spec.md.

Le template impose le format TOON v1.0 exclusivement.

LANGUE DU PROMPT — décision actée (Lot 2) :
    Ce prompt était rédigé en français. Mesure du diagnostic : 14 des 54
    sous-requêtes générées (25,9 %) sortaient en français, alors que le
    corpus documentaire et les requêtes utilisateur sont anglophones. Des
    sous-requêtes françaises envoyées à un retriever anglophone dégradent
    directement la qualité du retrieval — c'est un défaut fonctionnel, pas
    une question de style.

    Le prompt est donc intégralement en anglais : instructions ET exemples.
    Le format TOON est inchangé au caractère près — mêmes délimiteurs
    `<<<`/`>>>`, mêmes noms de champs (`plan_rationale`, `total_steps`,
    `step_id`, `sub_query`, `depends_on`), même séparateur `|`, même
    structure bloc d'en-tête + un bloc par étape. Seule la langue du texte
    naturel change.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt de décomposition Plan-and-Solve (Few-Shot)
# ─────────────────────────────────────────────────────────────────────────────

PLANNING_PROMPT: str = """You are a query planner for a RAG reasoning engine.

SOLE MISSION: Decompose the complex query into atomic sub-questions.
ABSOLUTE PROHIBITION: NEVER answer the question. Only decompose it.

─── MANDATORY CONSTRAINTS ───────────────────────────────────────────────────

1. The total number of step blocks MUST NOT exceed the given BUDGET.
2. Each `sub_query` must be a complete, self-contained and precise sentence.
3. The `depends_on` field must be EMPTY if the step has no prerequisite.
4. If two steps are independent, leave `depends_on` empty for both:
   they will be executed IN PARALLEL by the retrieval module.
5. Step identifiers must follow the format `step_1`, `step_2`, etc.
6. Write every `sub_query` in ENGLISH, regardless of the language of the
   incoming query — the document corpus being searched is in English.

─── RESPONSE FORMAT (STRICT TOON) ──────────────────────────────────────────

One header block followed by one block per step. No text before or after.

─── EXAMPLE 1: SEQUENTIAL QUERY (budget: 2) ─────────────────────────────────

Query : "Who leads the company that created the GPT-4 model?"
Budget : 2

<<<
plan_rationale :: Identify the company that created GPT-4, then look up its current leader. The second step necessarily depends on the result of the first.
total_steps :: 2
>>>

<<<
step_id :: step_1
sub_query :: Which company created and released the GPT-4 model?
depends_on ::
>>>

<<<
step_id :: step_2
sub_query :: Who is the CEO or managing director of the company that created GPT-4?
depends_on :: step_1
>>>

─── EXAMPLE 2: QUERY WITH PARALLEL STEPS (budget: 3) ────────────────────────

Query : "Compare the architectures of BERT and GPT-4, then conclude which one is better suited to automatic summarization."
Budget : 3

<<<
plan_rationale :: Information about BERT and GPT-4 is independent and can be retrieved in parallel. The final comparison waits for both results.
total_steps :: 3
>>>

<<<
step_id :: step_1
sub_query :: What is the technical architecture of the BERT model and its main characteristics?
depends_on ::
>>>

<<<
step_id :: step_2
sub_query :: What is the technical architecture of the GPT-4 model and its main characteristics?
depends_on ::
>>>

<<<
step_id :: step_3
sub_query :: Comparison of BERT and GPT-4 for the automatic text summarization task.
depends_on :: step_1 | step_2
>>>

─── QUERY TO DECOMPOSE ─────────────────────────────────────────────────────

Query : "{query}"
Budget : {reasoning_budget}

Generate ONLY the TOON blocks. No text before or after the blocks.
"""
