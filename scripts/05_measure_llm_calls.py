"""
Mesure des appels LLM par requête — Sprint I5-A.
Projet : RAG-REASON (intégration avec le module ACTION)

Exécute le graphe complet sur des questions du jeu d'évaluation, avec
l'instrumentation de `reasoning.observability`, et relève pour chacune :

    - le nombre d'appels LLM réellement émis,
    - la latence totale,
    - le chemin parcouru dans le graphe,
    - la classe attribuée par l'Analyzer et le budget associé.

Sert à deux choses : fixer le plafond `MAX_LLM_CALLS_PER_QUERY` sur des
mesures plutôt qu'à l'aveugle, et vérifier que les traces Langfuse remontent.

Prérequis : Ollama lancé, API du module ACTION sur le port 8000,
`USE_REAL_ACTION=true`.

Usage :
    uv run python scripts/05_measure_llm_calls.py
    uv run python scripts/05_measure_llm_calls.py <qid> [<qid> ...]
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_REAL_ACTION", "true")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.WARNING)
for noisy in ("LiteLLM", "httpx", "litellm", "opentelemetry"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_SET = BASE_DIR / "data" / "processed" / "hotpotqa_sprint3.json"

# Deux questions du jeu d'évaluation, une par profil de parcours.
#   SIMPLE     : budget 1, plan à une étape, aucune relance possible
#   MULTI_HOP  : budget 3, plan à deux étapes, relances + synthèse Lot B
DEFAULT_QIDS = ("5a75e05c55429976ec32bc5f", "5a8c7595554299585d9e36b6")


def question_of(qid: str) -> str:
    """Texte de la question du jeu d'évaluation."""
    for record in json.loads(EVAL_SET.read_text(encoding="utf-8")):
        if record["id"] == qid:
            text: str = record["question"]
            return text
    raise SystemExit(f"question {qid} absente de {EVAL_SET.name}")


async def measure(qid: str, question: str) -> dict[str, Any]:
    """Exécute le graphe sur une question et relève les compteurs."""
    from reasoning import observability
    from reasoning.graph.graph import build_graph
    from reasoning.graph.state import build_initial_state

    graph = build_graph()
    path: list[str] = []
    final: dict[str, Any] = {}

    started = time.perf_counter()
    with observability.trace_query(qid):
        async for event in graph.astream(
            build_initial_state(question), stream_mode="updates"
        ):
            for node, update in event.items():
                path.append(node)
                final.update(update)
        calls = observability.llm_call_count()
    total_ms = round((time.perf_counter() - started) * 1000)

    agent = final["agent_state"]
    analysis = agent.analysis
    return {
        "qid": qid,
        "question": question,
        "llm_calls": calls,
        "total_ms": total_ms,
        "path": path,
        "query_type": analysis.query_type.value if analysis else "?",
        "reasoning_budget": analysis.reasoning_budget if analysis else -1,
        "n_steps": len(agent.plan.steps) if agent.plan else 0,
        "n_evaluations": len(agent.evaluations),
        "n_retrieve": path.count("retrieve"),
        "step_answers": final.get("step_answers") or {},
        "llm_calls_state": final.get("llm_calls"),
    }


def main() -> None:
    """Point d'entrée : instrumente, mesure, résume."""
    from reasoning import observability

    print("\n[STEP 1] Activation de la traçabilité...")
    langfuse_on = observability.configure_langfuse()
    instrumented = observability.instrument()
    print(f"   Langfuse actif   : {langfuse_on}")
    print(f"   Modules tracés   : {len(instrumented)}")
    for name in instrumented:
        print(f"      - {name}")

    qids = sys.argv[1:] or list(DEFAULT_QIDS)
    print(f"\n[STEP 2] Mesure sur {len(qids)} question(s)...")

    results: list[dict[str, Any]] = []
    for qid in qids:
        question = question_of(qid)
        print(f"\n   --- {qid} ---")
        print(f"   {question}")
        result = asyncio.run(measure(qid, question))
        results.append(result)
        print(
            f"   query_type       : {result['query_type']} "
            f"(budget {result['reasoning_budget']})"
        )
        print(f"   etapes du plan   : {result['n_steps']}")
        print(f"   retrievals       : {result['n_retrieve']}")
        print(f"   evaluations      : {result['n_evaluations']}")
        print(f"   reponses interm. : {len(result['step_answers'])}")
        print(f"   APPELS LLM       : {result['llm_calls']}")
        print(f"   llm_calls (etat) : {result['llm_calls_state']}")
        print(f"   latence totale   : {result['total_ms']} ms")

    observability.flush()

    print("\n" + "=" * 68)
    print(f"  {'question':26} {'type':12} {'appels':>7} {'latence':>10}")
    print("-" * 68)
    for r in results:
        print(
            f"  {r['qid'][:24]:26} {r['query_type']:12} "
            f"{r['llm_calls']:>7} {r['total_ms']:>9} ms"
        )
    if results:
        peak = max(r["llm_calls"] for r in results)
        print("-" * 68)
        print(f"  maximum observé : {peak} appels")
        print(f"  plafond suggéré : {peak + 4} (maximum + marge de 4)")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
