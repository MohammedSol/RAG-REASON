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
import sys
import time
from pathlib import Path
from typing import TypedDict

from reasoning.analyzer import QueryAnalyzer
from reasoning.contracts.internal_models import QueryType
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

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
                )
            )

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

    # Latence moyenne
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
            "avg_latency_ms": round(avg_latency, 1),
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
    table.add_row("Latence moyenne par requete", f"{avg_latency:.0f} ms")
    table.add_row("Resultats sauvegardes dans", str(OUTPUT_FILE.relative_to(BASE_DIR)))

    console.print(table)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
