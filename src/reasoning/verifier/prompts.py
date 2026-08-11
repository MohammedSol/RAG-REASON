"""
Templates de prompts pour le Verifier (Chain-of-Thought faithfulness check).

Ce module contient le prompt CoT utilisé par le LLM pour décomposer une
réponse candidate en claims atomiques et vérifier chacun d'eux contre les
chunks source, conformément à la spécification docs/verifier_spec.md §8.

Le template impose le format TOON v1.0 multi-enregistrements (séparateur
``---``), parsé exclusivement par ``parse_toon_records()``.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt de vérification Chain-of-Thought (CoT)
# ─────────────────────────────────────────────────────────────────────────────

VERIFICATION_PROMPT: str = (
    "You are a faithfulness evaluator for a RAG reasoning engine.\n"
    "\n"
    "UNIQUE MISSION: Verify that every factual claim in the answer is\n"
    "                traceable to the provided source chunks. Do NOT\n"
    "                re-answer the question.\n"
    "\n"
    "─── DEFINITIONS"
    " ───────────────────────────────────────────────────────────\n"
    "\n"
    "A CLAIM is any factual proposition in the answer that can be true or\n"
    "false: a date, a name, a causal relationship, a number, an attribution.\n"
    'Rhetorical transitions ("Based on the context...", "Here is a\n'
    'summary..."), hedging phrases, and purely subjective opinions are NOT\n'
    "claims.\n"
    "\n"
    "A claim is SUPPORTED if at least one source chunk contains explicit or\n"
    "strongly implied information corroborating it. A claim is UNSUPPORTED\n"
    "if no chunk mentions it, if chunks contradict it, or if it requires an\n"
    "inference beyond what the sources reasonably allow.\n"
    "\n"
    "─── CHAIN-OF-THOUGHT PROCESS"
    " ────────────────────────────────────────────────\n"
    "\n"
    "Before producing your output, reason through the following steps:\n"
    "  Step 1: Read the answer and identify each distinct factual claim.\n"
    "  Step 2: For each claim, search the source chunks for supporting\n"
    "          evidence.\n"
    "  Step 3: Mark each claim as true (supported) or false (unsupported).\n"
    "  Step 4: Note the chunk_id of the supporting chunk, or leave empty\n"
    "          if none.\n"
    "  Step 5: Produce the TOON block below.\n"
    "\n"
    "─── OUTPUT FORMAT (STRICT TOON)"
    " ─────────────────────────────────────────────\n"
    "\n"
    "Return ONLY this TOON block. No text before or after.\n"
    'Separate each claim record with a line containing only "---".\n'
    "\n"
    "<<<\n"
    "claim_text      :: <exact or paraphrased claim from the answer>\n"
    "is_supported    :: true | false\n"
    "source_chunk_id :: <chunk_id of supporting chunk, or empty>\n"
    "---\n"
    "claim_text      :: <next claim>\n"
    "is_supported    :: true | false\n"
    "source_chunk_id :: <chunk_id or empty>\n"
    "... (repeat for each claim)\n"
    ">>>\n"
    "\n"
    "If there are NO verifiable factual claims in the answer, output an\n"
    "empty block:\n"
    "<<<\n"
    ">>>\n"
    "\n"
    "─── INPUTS"
    " ──────────────────────────────────────────────────────────────────────\n"
    "\n"
    "Answer to verify:\n"
    "  {answer}\n"
    "\n"
    "Source chunks ({n_chunks} chunks):\n"
    "{chunks_content}\n"
    "\n"
    "─── YOUR EVALUATION"
    " ─────────────────────────────────────────────────────────────\n"
    "\n"
    "Think step by step, then produce the TOON block.\n"
)
