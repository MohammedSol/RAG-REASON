"""
Script d'evaluation du Query Analyzer — Sprint 3 (Phase 2)
Projet : RAG-REASON

Charge le dataset hotpotqa_sprint3.json (200 requetes) et envoie chaque
question au QueryAnalyzer. Compare la prediction avec la ground truth :
  bridge     -> attendu MULTI_HOP
  comparison -> attendu COMPARATIVE

Sauvegarde les resultats detailles dans data/evaluation/analyzer_results.json
et affiche un tableau recapitulatif avec rich.

Usage :
    uv run python scripts/02_evaluate_analyzer.py
"""

from __future__ import annotations

import io
import json
import statistics
import sys
import time
from pathlib import Path
from typing import TypedDict

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

import reasoning.analyzer.analyzer as analyzer_module
from reasoning.analyzer import QueryAnalyzer
from reasoning.contracts.internal_models import QueryType

# ─────────────────────────────────────────────────────────────────────────────
# Fix encoding Windows (cp1252 ne gere pas tous les caracteres rich)
# ─────────────────────────────────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — chemins
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "hotpotqa_sprint3.json"
OUTPUT_FILE = BASE_DIR / "data" / "evaluation" / "analyzer_results.json"

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Types — structure d'un resultat d'iteration
# ─────────────────────────────────────────────────────────────────────────────


class EvalRecord(TypedDict):
    """Resultat detaille pour une seule requete evaluee."""

    id: str
    question: str
    hotpot_type: str  # ground truth originale (bridge / comparison)
    expected: str  # QueryType attendu selon notre mapping
    predicted: str  # QueryType predit par notre Analyzer
    confidence: float  # confidence score retourne par l'Analyzer
    is_correct: bool
    latency_ms: float  # temps d'inference en millisecondes
    decision_path: str  # "regex" | "llm" | "heuristic_fallback"
    is_cold_start: bool  # 1re requete : inclut le chargement du modele Ollama


# ─────────────────────────────────────────────────────────────────────────────
# Instrumentation du chemin de decision
# ─────────────────────────────────────────────────────────────────────────────
#
# POURQUOI : l'Analyzer a trois chemins (pre-classificateur regex -> LLM ->
# repli heuristique) et publiait une accuracy et une latence GLOBALES qui les
# melangeaient. Or ces chemins ont des profils radicalement differents
# (le regex repond en 0 ms, le LLM en ~7 s), ce qui rendait les moyennes
# publiees non representatives d'aucun des deux.
#
# CONTRAINTE : `AnalysisResult` est un contrat GELE ; il n'expose pas le
# chemin emprunte, et `src/reasoning/` ne doit pas etre modifie. Le chemin est
# donc reconstitue cote script, sans toucher au moteur :
#   1. `_pre_classify()` est pur et deterministe -> le rejouer indique
#      exactement si le regex a decide (methode validee : 0 incoherence sur
#      les 200 requetes du run precedent).
#   2. Un wrapper transparent autour de `completion` indique si le LLM a
#      reellement ete appele et s'il a leve une exception.
# Le wrapper delegue a l'implementation d'origine sans rien changer au
# comportement observable du moteur.

_llm_probe: dict[str, bool] = {"called": False, "raised": False}
# Suit si le 1er appel LLM reel (= le chargement du modele) a deja eu lieu.
_cold_start_seen: dict[str, bool] = {"done": False}
_original_completion = analyzer_module.completion


def _instrumented_completion(*args: object, **kwargs: object) -> object:
    """Wrapper transparent : trace l'appel LLM sans alterer son comportement."""
    _llm_probe["called"] = True
    try:
        return _original_completion(*args, **kwargs)
    except Exception:
        _llm_probe["raised"] = True
        raise


analyzer_module.completion = _instrumented_completion


def resolve_decision_path(question: str, confidence: float) -> str:
    """Determine le chemin reellement emprunte pour une requete.

    Args:
        question: La requete soumise a l'Analyzer.
        confidence: La confidence retournee par l'Analyzer.

    Returns:
        "regex", "llm" ou "heuristic_fallback".
    """
    # Niveau 0 : le pre-classificateur a tranche -> aucun appel LLM.
    if QueryAnalyzer._pre_classify(question) is not None:
        return "regex"
    # Niveau 2 : le LLM a echoue (exception) ou sa reponse etait inexploitable.
    # `_FALLBACK_CONFIDENCE` (0.55) est la signature documentee du repli.
    if _llm_probe["raised"] or confidence == analyzer_module._FALLBACK_CONFIDENCE:
        return "heuristic_fallback"
    # Niveau 1 : classification LLM nominale.
    return "llm"


# ─────────────────────────────────────────────────────────────────────────────
# Mapping ground truth HotpotQA → QueryType de notre systeme
# ─────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH_MAP: dict[str, QueryType] = {
    "bridge": QueryType.MULTI_HOP,
    "comparison": QueryType.COMPARATIVE,
}


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale d'evaluation
# ─────────────────────────────────────────────────────────────────────────────


def run_evaluation() -> None:
    """Point d'entree du script d'evaluation."""

    # ── Chargement du dataset ─────────────────────────────────────────────────
    console.print("\n[bold cyan][STEP 1][/bold cyan] Chargement du dataset...")

    if not INPUT_FILE.exists():
        console.print(f"[red][ERREUR] Fichier introuvable : {INPUT_FILE}[/red]")
        console.print(
            "[yellow]Lancez d'abord : uv run python scripts/prepare_hotpotqa.py[/yellow]"
        )
        sys.exit(1)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        records: list[dict[str, str]] = json.load(f)

    # On prend 12 questions 'bridge' (au début) et 13 questions 'comparison' (à la fin)
    # records = records[:12] + records[-13:]
    console.print(
        f"   [green][OK][/green] {len(records)} requetes chargees depuis {INPUT_FILE.name}"
    )

    # ── Initialisation du QueryAnalyzer ──────────────────────────────────────
    console.print(
        "\n[bold cyan][STEP 2][/bold cyan] Initialisation du QueryAnalyzer..."
    )
    analyzer = (
        QueryAnalyzer()
    )  # utilise les vars d'env OLLAMA_BASE_URL + DEFAULT_FAST_MODEL
    console.print(
        f"   [green][OK][/green] Modele : {analyzer.model} | api_base : {analyzer.api_base}"
    )

    # ── Boucle d'evaluation avec barre de progression rich ────────────────────
    console.print("\n[bold cyan][STEP 3][/bold cyan] Evaluation en cours...\n")

    eval_results: list[EvalRecord] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Analyse des requetes", total=len(records))

        for row in records:
            qid: str = row["id"]
            question: str = row["question"]
            hotpot_type: str = row["type"]  # "bridge" ou "comparison"

            # Ground truth : on mappe le type HotpotQA vers notre QueryType
            expected: QueryType = GROUND_TRUTH_MAP.get(hotpot_type, QueryType.AMBIGUOUS)

            # Inference avec mesure de latence
            _llm_probe["called"] = False
            _llm_probe["raised"] = False
            t_start = time.perf_counter()
            try:
                result = analyzer.analyze(question)
                predicted: QueryType = result.query_type
                confidence: float = result.confidence
            except Exception as exc:  # noqa: BLE001
                # En cas d'erreur LLM, on log et on marque comme incorrect
                console.print(f"\n[yellow][WARN] Erreur sur id={qid}: {exc}[/yellow]")
                predicted = QueryType.AMBIGUOUS
                confidence = 0.0
            t_end = time.perf_counter()

            latency_ms: float = (t_end - t_start) * 1000
            is_correct: bool = predicted == expected

            eval_results.append(
                EvalRecord(
                    id=qid,
                    question=question,
                    hotpot_type=hotpot_type,
                    expected=str(expected),
                    predicted=str(predicted),
                    confidence=round(confidence, 4),
                    is_correct=is_correct,
                    latency_ms=round(latency_ms, 1),
                    decision_path=resolve_decision_path(question, confidence),
                    # Le demarrage a froid est le 1er appel LLM REEL, pas la
                    # 1re requete : celle-ci peut etre captee par le regex et
                    # ne rien charger du tout (mesure a 0 ms, non pertinente).
                    # C'est cet appel qui supporte le chargement du modele
                    # Ollama (~32 s mesurees contre ~5 s a chaud).
                    is_cold_start=(
                        _llm_probe["called"] and not _cold_start_seen["done"]
                    ),
                )
            )
            if _llm_probe["called"]:
                _cold_start_seen["done"] = True

            progress.advance(task)

    # ── Calcul des metriques ──────────────────────────────────────────────────
    console.print("\n[bold cyan][STEP 4][/bold cyan] Calcul des metriques...\n")

    total = len(eval_results)
    correct = sum(1 for r in eval_results if r["is_correct"])
    accuracy = correct / total * 100

    # Precision par categorie
    bridge_records = [r for r in eval_results if r["hotpot_type"] == "bridge"]
    comparison_records = [r for r in eval_results if r["hotpot_type"] == "comparison"]

    bridge_correct = sum(1 for r in bridge_records if r["is_correct"])
    comparison_correct = sum(1 for r in comparison_records if r["is_correct"])

    bridge_acc = bridge_correct / len(bridge_records) * 100 if bridge_records else 0.0
    comparison_acc = (
        comparison_correct / len(comparison_records) * 100
        if comparison_records
        else 0.0
    )

    # ── Latence : ventilee par chemin, mediane + moyenne, hors demarrage a froid ──
    #
    # POURQUOI : l'ancienne metrique `avg_latency_ms` moyennait ensemble le
    # chemin regex (0 ms) et le chemin LLM (~7 s). Le resultat (4899 ms) ne
    # decrivait aucun des deux et variait avec la seule proportion de requetes
    # captees par le regex. On publie desormais mediane ET moyenne par chemin,
    # le demarrage a froid etant isole car il mesure le chargement du modele,
    # pas la performance du composant.
    warm = [r for r in eval_results if not r["is_cold_start"]]
    cold = [r for r in eval_results if r["is_cold_start"]]

    def _lat_stats(records: list[EvalRecord]) -> dict[str, float | int]:
        """Mediane, moyenne et volume des latences d'un sous-ensemble."""
        if not records:
            return {"n": 0, "median_ms": 0.0, "mean_ms": 0.0}
        values = sorted(r["latency_ms"] for r in records)
        return {
            "n": len(values),
            "median_ms": round(statistics.median(values), 1),
            "mean_ms": round(statistics.fmean(values), 1),
        }

    latency_by_path = {
        path: _lat_stats([r for r in warm if r["decision_path"] == path])
        for path in ("regex", "llm", "heuristic_fallback")
    }
    latency_warm_all = _lat_stats(warm)
    cold_start_ms = round(cold[0]["latency_ms"], 1) if cold else 0.0

    # ── Accuracy ventilee par chemin de decision ──────────────────────────────
    accuracy_by_path: dict[str, dict[str, float | int]] = {}
    for path in ("regex", "llm", "heuristic_fallback"):
        subset = [r for r in eval_results if r["decision_path"] == path]
        n_ok = sum(1 for r in subset if r["is_correct"])
        accuracy_by_path[path] = {
            "n": len(subset),
            "correct": n_ok,
            "accuracy_pct": round(n_ok / len(subset) * 100, 2) if subset else 0.0,
            "share_pct": round(len(subset) / total * 100, 2),
        }

    # Conservee pour compatibilite ascendante, mais non representative :
    # cf. l'explication ci-dessus.
    avg_latency = sum(r["latency_ms"] for r in eval_results) / total

    # ── Sauvegarde des resultats detailles ────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Structure complete du fichier de sortie : summary + details
    output_payload = {
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy_pct": round(accuracy, 2),
            "bridge_accuracy_pct": round(bridge_acc, 2),
            "comparison_accuracy_pct": round(comparison_acc, 2),
            # Moyenne globale toutes-latences confondues : conservee pour
            # comparaison historique, mais DILUEE (melange regex 0 ms et LLM).
            # Utiliser `latency` ci-dessous pour toute lecture serieuse.
            "avg_latency_ms": round(avg_latency, 1),
            "accuracy_by_path": accuracy_by_path,
            "latency": {
                "cold_start_ms": cold_start_ms,
                "warm_all": latency_warm_all,
                "by_path": latency_by_path,
            },
        },
        "results": eval_results,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    console.print(
        f"   [green][OK][/green] Resultats sauvegardes dans {OUTPUT_FILE.relative_to(BASE_DIR)}"
    )

    # ── Tableau recapitulatif rich ────────────────────────────────────────────
    table = Table(
        title="\nResultats de l'evaluation — Query Analyzer vs HotpotQA",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        min_width=60,
    )

    table.add_column("Metrique", style="bold", justify="left", min_width=30)
    table.add_column("Valeur", justify="right", min_width=20)

    # Couleur dynamique selon le score : vert si bon, jaune si moyen, rouge si faible
    def _color(val: float) -> str:
        if val >= 75:
            return "green"
        if val >= 50:
            return "yellow"
        return "red"

    table.add_row("Total requetes evaluees", str(total))
    table.add_row("Correctement classifiees", str(correct))
    table.add_row(
        "Accuracy globale",
        f"[{_color(accuracy)}]{accuracy:.1f}%[/{_color(accuracy)}]",
    )
    table.add_row(
        "Accuracy bridge (→ MULTI_HOP)",
        f"[{_color(bridge_acc)}]{bridge_acc:.1f}%  ({bridge_correct}/{len(bridge_records)})[/{_color(bridge_acc)}]",
    )
    table.add_row(
        "Accuracy comparison (→ COMPARATIVE)",
        f"[{_color(comparison_acc)}]{comparison_acc:.1f}%  ({comparison_correct}/{len(comparison_records)})[/{_color(comparison_acc)}]",
    )
    table.add_row("", "")
    table.add_row(
        "[dim]Latence moyenne GLOBALE (diluee)[/dim]",
        f"[dim]{avg_latency:.0f} ms[/dim]",
    )
    table.add_row("Resultats sauvegardes dans", str(OUTPUT_FILE.relative_to(BASE_DIR)))

    console.print(table)

    # ── Tableau : accuracy ventilee par chemin de decision ────────────────────
    path_table = Table(
        title="\nVentilation par chemin de decision",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        min_width=60,
    )
    path_table.add_column("Chemin", style="bold", min_width=20)
    path_table.add_column("Volume", justify="right")
    path_table.add_column("Accuracy", justify="right")
    path_table.add_column("Latence med.", justify="right")
    path_table.add_column("Latence moy.", justify="right")

    labels = {
        "regex": "Pre-classif. regex",
        "llm": "LLM",
        "heuristic_fallback": "Repli heuristique",
    }
    for path, label in labels.items():
        acc = accuracy_by_path[path]
        lat = latency_by_path[path]
        if acc["n"] == 0:
            continue
        path_table.add_row(
            label,
            f"{acc['n']} ({acc['share_pct']:.1f}%)",
            f"[{_color(float(acc['accuracy_pct']))}]{acc['accuracy_pct']:.1f}%[/{_color(float(acc['accuracy_pct']))}]",
            f"{lat['median_ms']:.0f} ms",
            f"{lat['mean_ms']:.0f} ms",
        )
    path_table.add_row("", "", "", "", "")
    path_table.add_row(
        "[bold]Toutes (a chaud)[/bold]",
        str(latency_warm_all["n"]),
        "—",
        f"{latency_warm_all['median_ms']:.0f} ms",
        f"{latency_warm_all['mean_ms']:.0f} ms",
    )

    console.print(path_table)
    console.print(
        f"\n[yellow]Note :[/yellow] la 1re requete ({cold_start_ms:.0f} ms) inclut le "
        "chargement du modele Ollama et est exclue des statistiques ci-dessus.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
