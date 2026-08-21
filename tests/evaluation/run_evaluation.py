"""
Pipeline de campagne d'évaluation — Sprint I5-B.

Exécute l'un des deux systèmes sur les 20 questions de `dataset_v1.json` et
consigne, pour chacune, tout ce dont l'étape RAGAS aura besoin.

ROBUSTESSE — une campagne dure des heures
==========================================
Chacune de ces propriétés répond à un incident réellement survenu sur ce
projet, ou à un risque mesuré :

* **Sauvegarde après CHAQUE question.** Le fichier de résultats est réécrit
  à chaque itération. Une interruption au bout de 3 h ne perd qu'une
  question, jamais la campagne.
* **Reprise automatique.** Au démarrage, les questions déjà présentes dans
  le fichier sont sautées. Relancer le script reprend là où il s'était
  arrêté ; `--force` retraite tout.
* **Aucune exception ne remonte.** Un timeout Ollama marque la question en
  échec et l'exécution continue. Les questions en échec sont réessayées à la
  relance suivante, puisqu'elles ne sont pas considérées comme acquises.
* **Écriture dans le projet, jamais dans un répertoire temporaire.** Le
  scratchpad et `/tmp` ont été purgés en cours d'exécution à deux reprises
  sur ce projet, détruisant des mesures.
* **Temps par question journalisé**, pour repérer une dérive au fil de la
  campagne (chauffe du modèle, contention mémoire).

Usage :
    uv run python tests/evaluation/run_evaluation.py naive
    uv run python tests/evaluation/run_evaluation.py full
    uv run python tests/evaluation/run_evaluation.py full --force
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_REAL_ACTION", "true")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "tests" / "evaluation"))

RESULTS_DIR = BASE_DIR / "tests" / "evaluation" / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
for noisy in ("LiteLLM", "httpx", "litellm", "opentelemetry", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
logger = logging.getLogger("campagne")

SYSTEMS = ("naive", "full")


# ─────────────────────────────────────────────────────────────────────────────
# Persistance incrémentale
# ─────────────────────────────────────────────────────────────────────────────


def results_path(system: str) -> Path:
    """Fichier de résultats d'un système."""
    return RESULTS_DIR / f"run_{system}.json"


def load_existing(system: str) -> dict[str, dict[str, Any]]:
    """Résultats déjà acquis, indexés par identifiant de question.

    Les entrées en échec ne sont PAS considérées comme acquises : elles
    seront réessayées à la relance.
    """
    path = results_path(system)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Fichier de résultats illisible (%s) — reprise à zéro.", exc)
        return {}
    return {
        item["id"]: item
        for item in payload.get("results", [])
        if item.get("error") is None
    }


def save(system: str, results: dict[str, dict[str, Any]]) -> None:
    """Réécrit le fichier de résultats. Appelé après CHAQUE question."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results.values(), key=lambda r: (r.get("type", ""), r["id"]))
    payload = {
        "system": system,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n": len(ordered),
        "n_failed": sum(1 for r in ordered if r.get("error")),
        "results": ordered,
    }
    # Écriture atomique : un fichier temporaire VOISIN puis un remplacement,
    # pour qu'une interruption pendant l'écriture ne laisse pas un JSON
    # tronqué à la place des heures de mesures déjà acquises.
    tmp = results_path(system).with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(results_path(system))


# ─────────────────────────────────────────────────────────────────────────────
# Exécution d'une question — baseline
# ─────────────────────────────────────────────────────────────────────────────


def run_naive(sample: Any) -> dict[str, Any]:
    """Traite une question avec le baseline RAG naïf."""
    from naive_rag import NaiveRag

    from reasoning import observability

    engine = NaiveRag()
    with observability.trace_query(f"naive-{sample.id}"):
        result = engine.answer(sample.question, query_id=f"naive-{sample.id}")
        llm_calls = observability.llm_call_count()

    return {
        "id": sample.id,
        "type": sample.type,
        "question": sample.question,
        "answer": result.answer,
        "contexts": result.contexts,
        "sources": result.sources,
        "llm_calls": llm_calls,
        "latency_ms": result.total_ms,
        "retrieval_ms": result.retrieval_ms,
        "generation_ms": result.generation_ms,
        # Le baseline n'a pas de Verifier : ces champs restent nuls, et c'est
        # précisément ce que le système complet apporte.
        "is_grounded": None,
        "faithfulness_score": None,
        "unsupported_claims": None,
        "error": result.error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Exécution d'une question — système complet
# ─────────────────────────────────────────────────────────────────────────────


async def _run_graph(question: str, query_id: str) -> dict[str, Any]:
    """Exécute le graphe et collecte l'état final."""
    from reasoning.graph.graph import build_graph
    from reasoning.graph.state import build_initial_state

    graph = build_graph()
    path: list[str] = []
    final: dict[str, Any] = {}

    async for event in graph.astream(
        build_initial_state(question), stream_mode="updates"
    ):
        for node, update in event.items():
            path.append(node)
            final.update(update)

    return {"path": path, "final": final}


def run_full(sample: Any) -> dict[str, Any]:
    """Traite une question avec le système complet."""
    from reasoning import observability

    query_id = f"full-{sample.id}"
    started = time.perf_counter()
    with observability.trace_query(query_id):
        try:
            outcome = asyncio.run(_run_graph(sample.question, query_id))
            error: str | None = None
        except Exception as exc:  # noqa: BLE001 — la campagne ne doit pas casser
            logger.error(
                "%s : exécution du graphe en échec (%s: %s).",
                sample.id,
                type(exc).__name__,
                exc,
            )
            outcome = {"path": [], "final": {}}
            error = f"{type(exc).__name__}: {exc}"
        llm_calls = observability.llm_call_count()
    latency_ms = round((time.perf_counter() - started) * 1000)

    final = outcome["final"]
    agent = final.get("agent_state")
    chunks = final.get("retrieved_chunks") or []
    verification = getattr(agent, "verification", None) if agent else None
    analysis = getattr(agent, "analysis", None) if agent else None
    plan = getattr(agent, "plan", None) if agent else None

    answer = final.get("answer") or ""
    if verification is not None and verification.final_answer:
        answer = verification.final_answer

    return {
        "id": sample.id,
        "type": sample.type,
        "question": sample.question,
        "answer": answer,
        "contexts": [c.content for c in chunks],
        "sources": [c.source for c in chunks],
        "llm_calls": llm_calls,
        "latency_ms": latency_ms,
        "path": outcome["path"],
        "query_type": analysis.query_type.value if analysis else None,
        "reasoning_budget": analysis.reasoning_budget if analysis else None,
        "n_steps": len(plan.steps) if plan else 0,
        "sub_queries": [s.sub_query for s in plan.steps] if plan else [],
        "step_answers": final.get("step_answers") or {},
        "evaluations": [
            {
                "step_id": e.step_id,
                "is_sufficient": e.is_sufficient,
                "relevance_score": e.relevance_score,
                "missing_aspects": e.missing_aspects,
                "feedback": e.feedback,
            }
            for e in (agent.evaluations if agent else [])
        ],
        # Verdict du Verifier — servira au jeu annoté du Sprint I5-C.
        "is_grounded": verification.is_grounded if verification else None,
        "faithfulness_score": (
            verification.faithfulness_score if verification else None
        ),
        "unsupported_claims": (
            verification.unsupported_claims if verification else None
        ),
        "error": error if error else (None if answer else "réponse vide"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Boucle de campagne
# ─────────────────────────────────────────────────────────────────────────────


def preflight() -> None:
    """Vérifie que les deux services requis répondent, avant d'engager des heures."""
    import httpx

    for label, url in (
        (
            "module ACTION",
            f"{os.getenv('ACTION_BASE_URL', 'http://localhost:8000')}/health",
        ),
        (
            "Ollama",
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags",
        ),
    ):
        try:
            httpx.get(url, timeout=8.0).raise_for_status()
            logger.info("%s : disponible.", label)
        except (httpx.HTTPError, OSError) as exc:
            raise SystemExit(
                f"{label} indisponible sur {url} ({type(exc).__name__}: {exc}). "
                "Campagne annulée avant d'engager du temps de calcul."
            ) from exc


def main() -> None:
    """Point d'entrée : prépare, exécute, sauvegarde à chaque pas."""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if len(sys.argv) < 2 or sys.argv[1] not in SYSTEMS:
        raise SystemExit(f"usage : run_evaluation.py {{{'|'.join(SYSTEMS)}}} [--force]")
    system = sys.argv[1]
    force = "--force" in sys.argv

    from dataset import load

    from reasoning import observability

    samples = load()
    logger.info("Jeu de données : %d questions.", len(samples))

    preflight()

    langfuse = observability.configure_langfuse()
    instrumented = observability.instrument()
    logger.info(
        "Traçabilité : Langfuse=%s, %d modules instrumentés.",
        langfuse,
        len(instrumented),
    )

    results = {} if force else load_existing(system)
    todo = [s for s in samples if s.id not in results]
    logger.info(
        "Campagne « %s » : %d acquises, %d à traiter.",
        system,
        len(results),
        len(todo),
    )
    if not todo:
        logger.info("Rien à faire — campagne déjà complète.")
        return

    runner = run_naive if system == "naive" else run_full
    campaign_start = time.perf_counter()

    for index, sample in enumerate(todo, start=1):
        logger.info(
            "[%d/%d] %s (%s) — %s",
            index,
            len(todo),
            sample.id,
            sample.type,
            sample.question[:58],
        )
        started = time.perf_counter()
        try:
            record = runner(sample)
        except Exception as exc:  # noqa: BLE001 — jamais interrompre la campagne
            logger.error(
                "[%d/%d] %s : ÉCHEC (%s: %s) — question marquée, campagne poursuivie.",
                index,
                len(todo),
                sample.id,
                type(exc).__name__,
                exc,
            )
            record = {
                "id": sample.id,
                "type": sample.type,
                "question": sample.question,
                "answer": "",
                "contexts": [],
                "sources": [],
                "llm_calls": 0,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
            }

        results[sample.id] = record
        save(system, results)  # ← après CHAQUE question

        elapsed = time.perf_counter() - campaign_start
        remaining = (elapsed / index) * (len(todo) - index)
        logger.info(
            "     %d appels LLM · %.1f s · %d chunks%s | écoulé %.0f min, "
            "reste ~%.0f min",
            record.get("llm_calls", 0),
            record.get("latency_ms", 0) / 1000,
            len(record.get("contexts") or []),
            "" if not record.get("error") else f" · ERREUR {record['error'][:60]}",
            elapsed / 60,
            remaining / 60,
        )

    observability.flush()

    failed = [r for r in results.values() if r.get("error")]
    total_calls = sum(r.get("llm_calls", 0) for r in results.values())
    total_min = (time.perf_counter() - campaign_start) / 60

    print("\n" + "=" * 72)
    print(f"  campagne          : {system}")
    print(f"  questions traitées: {len(results)}/{len(samples)}")
    print(f"  en échec          : {len(failed)}")
    for r in failed:
        print(f"      {r['id']} — {str(r.get('error'))[:70]}")
    print(
        f"  appels LLM totaux : {total_calls} "
        f"({total_calls / max(len(results), 1):.1f} par question)"
    )
    print(f"  durée             : {total_min:.0f} min")
    print(f"  résultats         : {results_path(system).relative_to(BASE_DIR)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
