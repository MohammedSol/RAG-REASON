"""
Métriques RAGAS et comparaison des deux systèmes — Sprint I5-B.

Calcule `faithfulness`, `answer_relevancy`, `context_precision` et
`context_recall` sur les résultats de campagne, puis produit un rapport
comparatif JSON et Markdown, global et ventilé bridge/comparison.

BIAIS À SIGNALER, PAS À MASQUER
================================
**Le même modèle génère et juge.** Les réponses évaluées sont produites par
`qwen2.5:7b` ; les métriques RAGAS sont calculées en interrogeant… le même
`qwen2.5:7b`. Un modèle est un juge indulgent de sa propre production : les
valeurs absolues sont donc à considérer avec réserve.

Ce que cela n'invalide pas : la COMPARAISON entre les deux systèmes. Le
biais s'applique identiquement au baseline et au système complet, puisque
tous deux génèrent avec le même modèle et sont jugés par le même juge.
L'ÉCART reste interprétable ; les niveaux absolus, beaucoup moins.

Lever ce biais demanderait un juge distinct — un modèle plus puissant, ou une
annotation humaine. Le jeu annoté du Sprint I5-C est précisément destiné à
fournir ce point de contrôle indépendant.

EMBEDDINGS
----------
`answer_relevancy` exige un modèle d'embeddings. Le serveur Ollama de cette
machine refuse les deux endpoints d'embedding (« This server does not support
embeddings. Start it with `--embeddings` »). On utilise donc `fastembed`
(ONNX, BAAI/bge-small-en-v1.5, 384 dimensions) — local, sans torch, et
indépendant du serveur Ollama.

Usage :
    uv run python tests/evaluation/compute_metrics.py            # les deux
    uv run python tests/evaluation/compute_metrics.py naive      # un seul
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Télémétrie RAGAS désactivée — AVANT tout import de `ragas`.
#
# Elle valide un `EmbeddingUsageEvent` dont le champ `model` doit être une
# chaîne. Avec `FastEmbedEmbeddings`, RAGAS lui passe l'OBJET d'embeddings et
# la validation Pydantic échoue, ce qui fait échouer `answer_relevancy` —
# alors que le calcul lui-même est parfaitement valide. Cette télémétrie
# n'apporte rien à l'évaluation ; l'écarter est le correctif le plus direct,
# et il évite en prime d'envoyer des données d'usage à un tiers.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "tests" / "evaluation"))

RESULTS_DIR = BASE_DIR / "tests" / "evaluation" / "results"
REPORTS_DIR = BASE_DIR / "tests" / "evaluation" / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
for noisy in ("LiteLLM", "httpx", "litellm", "opentelemetry", "urllib3", "ragas"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
logger = logging.getLogger("metriques")

JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "ollama/qwen2.5:7b")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

# RAGAS nomme ses colonnes d'après la CLASSE de métrique, pas d'après le nom
# usuel : `LLMContextPrecisionWithReference` produit la colonne
# `llm_context_precision_with_reference`. Sans cette table, le score existe
# mais n'est jamais lu — il ressort à `n/d`, ce qui s'est produit au premier
# essai. Les alias sont essayés dans l'ordre, le premier trouvé l'emporte.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "faithfulness": ("faithfulness",),
    "answer_relevancy": ("answer_relevancy", "response_relevancy"),
    "context_precision": (
        "context_precision",
        "llm_context_precision_with_reference",
    ),
    "context_recall": ("context_recall", "llm_context_recall"),
}


def build_judge() -> tuple[Any, Any]:
    """Construit le LLM juge et le modèle d'embeddings pour RAGAS."""
    import litellm
    from langchain_community.chat_models import ChatLiteLLM
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # `ResponseRelevancy` demande N reformulations en un appel via le
    # paramètre OpenAI `n`, qu'Ollama ne supporte pas :
    #     UnsupportedParamsError: ollama does not support parameters: ['n']
    # Sans ce réglage, la métrique échoue et ressort à `n/d`. On laisse donc
    # LiteLLM écarter les paramètres non supportés.
    #
    # CONSÉQUENCE À CONNAÎTRE : `answer_relevancy` est alors calculée sur UNE
    # reformulation au lieu de trois (`strictness` par défaut). La métrique
    # reste valide mais son estimation est plus bruitée. Le biais s'applique
    # identiquement aux deux systèmes, donc la comparaison reste juste.
    litellm.drop_params = True

    judge = ChatLiteLLM(
        model=JUDGE_MODEL,
        api_base=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.0,
        max_tokens=1024,
        request_timeout=300,
    )
    return LangchainLLMWrapper(judge), LangchainEmbeddingsWrapper(_fastembed())


def _fastembed() -> Any:
    """Modèle d'embeddings local, adapté aux attentes de RAGAS.

    POURQUOI UN ADAPTATEUR. RAGAS journalise chaque usage d'embeddings dans un
    `EmbeddingUsageEvent` dont le champ `model` doit être une CHAÎNE. Il le lit
    sur l'objet d'embeddings — or `FastEmbedEmbeddings.model` est le moteur
    ONNX lui-même, pas son nom. La validation Pydantic échoue, et
    `answer_relevancy` ressort à `nan` alors que le calcul est valide.

    Positionner `RAGAS_DO_NOT_TRACK` ne suffit pas : l'événement est construit
    avant d'être éventuellement écarté. Sous-classer ne suffit pas non plus —
    `model` est un CHAMP pydantic, qu'une propriété de sous-classe ne masque
    pas. D'où ce délégateur simple, hors pydantic, qui expose `model` comme une
    chaîne et transmet tout le reste à l'implémentation réelle.
    """
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    inner = FastEmbedEmbeddings(model_name=EMBED_MODEL)

    class _NamedEmbeddings:
        """Délégateur exposant `model` sous forme de chaîne."""

        model = EMBED_MODEL

        def embed_query(self, text: str) -> list[float]:
            result: list[float] = inner.embed_query(text)
            return result

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            result: list[list[float]] = inner.embed_documents(texts)
            return result

        async def aembed_query(self, text: str) -> list[float]:
            result: list[float] = await inner.aembed_query(text)
            return result

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            result: list[list[float]] = await inner.aembed_documents(texts)
            return result

    return _NamedEmbeddings()


def build_dataset(system: str) -> tuple[Any, list[dict[str, Any]]]:
    """Assemble le jeu RAGAS depuis les résultats de campagne.

    Returns:
        Le `EvaluationDataset` et les enregistrements retenus, dans le même
        ordre — indispensable pour rattacher chaque score à sa question.
    """
    from dataset import load
    from ragas import EvaluationDataset, SingleTurnSample

    truth = {s.id: s for s in load()}
    payload = json.loads((RESULTS_DIR / f"run_{system}.json").read_text("utf-8"))

    samples: list[Any] = []
    kept: list[dict[str, Any]] = []
    for record in payload["results"]:
        if record.get("error") or not record.get("answer"):
            logger.warning(
                "%s/%s : écarté du calcul (erreur ou réponse vide).",
                system,
                record["id"],
            )
            continue
        reference = truth[record["id"]]
        samples.append(
            SingleTurnSample(
                user_input=record["question"],
                response=record["answer"],
                retrieved_contexts=record["contexts"] or [""],
                reference=reference.ground_truth_answer,
                reference_contexts=reference.ground_truth_contexts,
            )
        )
        kept.append(record)

    return EvaluationDataset(samples=samples), kept


def _partial_path(system: str) -> Path:
    """Fichier des scores déjà calculés, question par question."""
    return REPORTS_DIR / f"metrics_{system}.partial.json"


def _load_partial(system: str) -> dict[str, dict[str, Any]]:
    """Scores déjà acquis, indexés par identifiant de question."""
    path = _partial_path(system)
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {row["id"]: row for row in rows}


def evaluate_system(system: str) -> dict[str, Any]:
    """Calcule les quatre métriques, UNE QUESTION À LA FOIS.

    POURQUOI PAS UN SEUL APPEL `evaluate()` SUR LES 20 QUESTIONS. Un premier
    essai en lot, avec `max_workers=4` et la barre de progression, s'est
    BLOQUÉ : plus aucune requête n'atteignait Ollama après quelques minutes,
    et le processus ne consommait plus de CPU. Aucune trace, aucune erreur —
    50 minutes perdues avant que le diagnostic ne soit posé, l'horodatage
    d'expiration du modèle chargé côté Ollama étant resté figé.

    En séquentiel (`max_workers=1`, sans barre), RAGAS fonctionne. Le calcul
    est donc découpé par question, avec sauvegarde après chacune :

    * une question bloquée ou en échec ne coûte plus toute la campagne ;
    * la progression est observable, dans le journal et sur disque ;
    * relancer le script reprend là où il s'était arrêté.

    Le coût de ce découpage est négligeable : le goulot est le modèle local,
    pas l'orchestration.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    dataset, kept = build_dataset(system)
    llm, embeddings = build_judge()
    metrics = [
        faithfulness,
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    run_config = RunConfig(max_workers=1, timeout=600, max_retries=1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    acquired = _load_partial(system)
    logger.info(
        "« %s » : %d questions, %d scores déjà acquis.",
        system,
        len(kept),
        len(acquired),
    )

    started = time.perf_counter()
    for position, record in enumerate(kept, start=1):
        if record["id"] in acquired:
            continue

        row: dict[str, Any] = {
            "id": record["id"],
            "type": record["type"],
            "llm_calls": record.get("llm_calls"),
            "latency_ms": record.get("latency_ms"),
        }
        question_started = time.perf_counter()
        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=[dataset.samples[position - 1]]),
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
                show_progress=False,
            )
            # `evaluate` est typé `EvaluationResult | Executor` ; seul le
            # premier cas se produit en appel synchrone, ce qui est le nôtre.
            frame = result.to_pandas()  # type: ignore[union-attr]
            scores = frame.iloc[0]
            for name in METRIC_NAMES:
                column = next(
                    (c for c in _COLUMN_ALIASES[name] if c in frame.columns), None
                )
                value = scores[column] if column is not None else None
                # NaN != NaN : c'est ainsi qu'on repère un score non calculé.
                row[name] = (
                    float(value) if value is not None and value == value else None
                )
        except Exception as exc:  # noqa: BLE001 — ne jamais perdre la campagne
            logger.error(
                "%s/%s : calcul en échec (%s: %s) — question marquée, suite.",
                system,
                record["id"],
                type(exc).__name__,
                exc,
            )
            for name in METRIC_NAMES:
                row[name] = None
            row["error"] = f"{type(exc).__name__}: {exc}"

        acquired[record["id"]] = row
        _partial_path(system).write_text(
            json.dumps(list(acquired.values()), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "[%d/%d] %s : %s | %.0f s | écoulé %.0f min",
            position,
            len(kept),
            record["id"],
            " ".join(
                f"{n.split('_')[0]}={row[n]:.2f}"
                if row[n] is not None
                else f"{n.split('_')[0]}=n/d"
                for n in METRIC_NAMES
            ),
            time.perf_counter() - question_started,
            (time.perf_counter() - started) / 60,
        )

    elapsed = time.perf_counter() - started
    per_question = [acquired[r["id"]] for r in kept if r["id"] in acquired]
    logger.info("« %s » : terminé en %.0f min.", system, elapsed / 60)

    return {
        "system": system,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBED_MODEL,
        "n": len(per_question),
        "elapsed_minutes": round(elapsed / 60, 1),
        "aggregate": _aggregate(per_question),
        "by_type": {
            label: _aggregate([q for q in per_question if q["type"] == label])
            for label in ("bridge", "comparison")
        },
        "per_question": per_question,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Moyenne chaque métrique en ignorant les valeurs manquantes."""
    summary: dict[str, Any] = {"n": len(rows)}
    for name in METRIC_NAMES:
        values = [r[name] for r in rows if r.get(name) is not None]
        summary[name] = round(sum(values) / len(values), 4) if values else None
        summary[f"{name}_n"] = len(values)
    calls = [r["llm_calls"] for r in rows if r.get("llm_calls") is not None]
    latency = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    summary["llm_calls_mean"] = round(sum(calls) / len(calls), 2) if calls else None
    summary["latency_s_mean"] = (
        round(sum(latency) / len(latency) / 1000, 1) if latency else None
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Rapport comparatif
# ─────────────────────────────────────────────────────────────────────────────


def _delta(full: float | None, naive: float | None) -> str:
    """Écart en points, formaté, ou `n/d` si une valeur manque."""
    if full is None or naive is None:
        return "n/d"
    return f"{(full - naive) * 100:+.1f}"


def _table(naive: dict[str, Any], full: dict[str, Any]) -> list[str]:
    """Tableau Markdown d'un groupe de résultats."""
    lines = [
        "| Métrique | Baseline naïf | Système complet | Écart (points) |",
        "|---|---:|---:|---:|",
    ]
    for name in METRIC_NAMES:
        a, b = naive.get(name), full.get(name)
        lines.append(
            f"| `{name}` | {a if a is not None else 'n/d'} "
            f"| {b if b is not None else 'n/d'} | **{_delta(b, a)}** |"
        )
    lines.append(
        f"| appels LLM / question | {naive.get('llm_calls_mean')} "
        f"| {full.get('llm_calls_mean')} | — |"
    )
    lines.append(
        f"| latence moyenne (s) | {naive.get('latency_s_mean')} "
        f"| {full.get('latency_s_mean')} | — |"
    )
    return lines


def write_report(reports: dict[str, dict[str, Any]]) -> None:
    """Écrit le rapport comparatif JSON et Markdown."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    naive, full = reports.get("naive"), reports.get("full")
    if naive is None or full is None:
        logger.warning("Une seule campagne évaluée — pas de comparatif.")
        return

    comparison = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBED_MODEL,
        "bias_notice": (
            "Le modèle qui génère les réponses est aussi celui qui les juge "
            "(qwen2.5:7b). Les valeurs absolues sont optimistes ; l'ÉCART "
            "entre les deux systèmes reste interprétable, le biais leur étant "
            "commun."
        ),
        "aggregate": {"naive": naive["aggregate"], "full": full["aggregate"]},
        "by_type": {"naive": naive["by_type"], "full": full["by_type"]},
        "faithfulness_gain_points": (
            round(
                (full["aggregate"]["faithfulness"] - naive["aggregate"]["faithfulness"])
                * 100,
                1,
            )
            if full["aggregate"]["faithfulness"] is not None
            and naive["aggregate"]["faithfulness"] is not None
            else None
        ),
    }
    (REPORTS_DIR / "comparison_v1.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Évaluation comparative — Sprint I5-B",
        "",
        f"Généré le {comparison['generated_at']}.",
        "",
        f"- **Juge** : `{JUDGE_MODEL}` · **Embeddings** : `{EMBED_MODEL}`",
        f"- **Questions** : {naive['aggregate']['n']} (baseline) / "
        f"{full['aggregate']['n']} (système complet)",
        "",
        "> **Biais à connaître.** Le modèle qui génère les réponses est aussi",
        "> celui qui les juge. Les valeurs absolues sont donc optimistes.",
        "> L'écart entre les deux systèmes reste interprétable : le biais leur",
        "> est commun.",
        "",
        "## Global",
        "",
        *_table(naive["aggregate"], full["aggregate"]),
        "",
        "## Bridge — multi-hop",
        "",
        *_table(naive["by_type"]["bridge"], full["by_type"]["bridge"]),
        "",
        "## Comparison",
        "",
        *_table(naive["by_type"]["comparison"], full["by_type"]["comparison"]),
        "",
        "## KPI principal",
        "",
        f"**Gain de faithfulness : {comparison['faithfulness_gain_points']} points** "
        "(cible indicative du cahier des charges : +15).",
        "",
    ]
    (REPORTS_DIR / "comparison_v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    logger.info("Rapports écrits dans %s", REPORTS_DIR.relative_to(BASE_DIR))

    print("\n" + "\n".join(lines))


def main() -> None:
    """Point d'entrée : calcule, agrège, rapporte."""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    wanted = [a for a in sys.argv[1:] if a in ("naive", "full")] or ["naive", "full"]
    reports: dict[str, dict[str, Any]] = {}

    for system in wanted:
        path = RESULTS_DIR / f"run_{system}.json"
        if not path.is_file():
            logger.warning(
                "%s absent — campagne « %s » non évaluée.", path.name, system
            )
            continue
        report = evaluate_system(system)
        out = REPORTS_DIR / f"metrics_{system}.json"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Rapport « %s » : %s", system, out.relative_to(BASE_DIR))
        reports[system] = report

    # Recharge un rapport déjà calculé, pour permettre de n'en relancer qu'un.
    for system in ("naive", "full"):
        cached = REPORTS_DIR / f"metrics_{system}.json"
        if system not in reports and cached.is_file():
            reports[system] = json.loads(cached.read_text("utf-8"))

    write_report(reports)


if __name__ == "__main__":
    main()
