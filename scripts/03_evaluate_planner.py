"""
Script d'evaluation du Planner — Sprint 3 (Phase 3)
Projet : RAG-REASON

Charge hotpotqa_sprint3.json, filtre les requetes complexes (bridge = MULTI_HOP,
comparison = COMPARATIVE), tire un echantillon aleatoire de 25 requetes, les
envoie au Planner et valide chaque plan selon deux criteres :
  1. Format TOON   : la reponse LLM respecte bien le contrat <<<...>>>
  2. DAG valide    : pas de cycle, pas de reference a une etape inexistante

Affichage via rich (barre de progression, panels par requete, tableau recap).

Usage :
    uv run python scripts/03_evaluate_planner.py
"""

from __future__ import annotations

import io
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import TypedDict

from reasoning.contracts.internal_models import (
    AnalysisResult,
    ExecutionPlan,
    QueryType,
)
from reasoning.planner import Planner
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# ─────────────────────────────────────────────────────────────────────────────
# Fix encoding Windows — cp1252 ne gere pas tous les caracteres rich
# ─────────────────────────────────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "hotpotqa_sprint3.json"
OUTPUT_FILE = BASE_DIR / "data" / "evaluation" / "planner_results.json"

SAMPLE_SIZE = 25  # nombre de requetes a evaluer (compromis duree / fiabilite)
RANDOM_SEED = 42  # reproductibilite de l'echantillonnage

# Mapping ground-truth HotpotQA → QueryType de notre systeme
HOTPOT_TYPE_MAP: dict[str, QueryType] = {
    "bridge": QueryType.MULTI_HOP,
    "comparison": QueryType.COMPARATIVE,
}

# Budget par query_type — conforme a internal_models.py
BUDGET_MAP: dict[QueryType, int] = {
    QueryType.MULTI_HOP: 3,
    QueryType.COMPARATIVE: 2,
}

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Types — structure d'un resultat par requete
# ─────────────────────────────────────────────────────────────────────────────


class PlannerEvalRecord(TypedDict):
    """Resultat detaille pour une seule requete evaluee par le Planner."""

    id: str
    question: str
    hotpot_type: str  # "bridge" ou "comparison"
    query_type: str  # QueryType utilise pour la planification
    n_steps: int  # nombre d'etapes dans le plan genere
    toon_valid: bool  # la reponse LLM respectait le format TOON
    dag_valid: bool  # le graphe est acyclique et coherent
    is_fallback: bool  # le planner a utilise le fallback sequentiel
    latency_ms: float
    steps_summary: list[str]  # [step_id: sub_query] pour affichage


# ─────────────────────────────────────────────────────────────────────────────
# Validation algorithmique du DAG (independante du Planner)
# ─────────────────────────────────────────────────────────────────────────────


def validate_dag(plan: ExecutionPlan) -> tuple[bool, str]:
    """Verifie que le plan est un DAG valide via tri topologique (Kahn's algo).

    Deux criteres valides :
    1. Aucune dependance vers une etape inexistante dans le plan.
    2. Absence de cycle (algo de Kahn : si tous les noeuds sont traites, pas de cycle).

    Args:
        plan: L'ExecutionPlan genere par le Planner.

    Returns:
        Tuple (is_valid, reason) — reason decrit la cause d'invalidite si False.
    """
    step_ids = {s.step_id for s in plan.steps}

    # Verif 1 : references vers des etapes inexistantes
    for step in plan.steps:
        for dep in step.depends_on:
            if dep not in step_ids:
                return False, f"Dependance inconnue : {dep!r} dans {step.step_id}"

    # Verif 2 : detection de cycles via Kahn's algo O(V+E)
    # in_degree[step_id] = nombre de dependances entrantes
    in_degree: dict[str, int] = {s.step_id: len(s.depends_on) for s in plan.steps}

    # Voisins : pour chaque etape, les etapes qui en dependent
    dependents: dict[str, list[str]] = {s.step_id: [] for s in plan.steps}
    for step in plan.steps:
        for dep in step.depends_on:
            dependents[dep].append(step.step_id)

    # File d'attente : etapes sans dependances (pretes a executer)
    queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    processed = 0

    while queue:
        current = queue.popleft()
        processed += 1
        for neighbor in dependents[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if processed != len(plan.steps):
        return (
            False,
            f"Cycle detecte : seulement {processed}/{len(plan.steps)} etapes traitees",
        )

    return True, "OK"


def detect_fallback(plan: ExecutionPlan) -> bool:
    """Detecte si le Planner a utilise le fallback sequentiel.

    Le fallback produit toujours des etapes avec des depends_on sequentiels
    (step_2 depend de step_1, step_3 depend de step_2, etc.) et des
    sub_queries generees automatiquement plutot que decomposees par le LLM.

    Heuristique simple : plan de 1 seule etape sur une requete complexe
    ou chaine strictement sequentielle avec sub_queries identiques ou vides.
    """
    if len(plan.steps) == 1:
        return True  # plan mono-etape sur requete complexe = fallback

    # Chaine strictement sequentielle : each step depends only on previous
    is_sequential_chain = all(
        (i == 0 and step.depends_on == [])
        or (i > 0 and step.depends_on == [plan.steps[i - 1].step_id])
        for i, step in enumerate(plan.steps)
    )

    # Si chaine sequentielle ET toutes les sub_queries sont des fragments
    # de la question originale → c'est probablement le fallback
    if is_sequential_chain:
        original = plan.original_query.lower()
        # Sub-queries du fallback contiennent souvent le texte original divise
        all_subqueries_are_fragments = all(
            any(word in step.sub_query.lower() for word in original.split()[:5])
            for step in plan.steps
        )
        return all_subqueries_are_fragments

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Affichage rich — panel par requete
# ─────────────────────────────────────────────────────────────────────────────


def render_plan_panel(
    idx: int,
    record: PlannerEvalRecord,
) -> None:
    """Affiche un panel rich pour une requete avec son plan genere."""

    # Status icons
    toon_icon = (
        "[green]FORMAT OK[/green]" if record["toon_valid"] else "[red]FORMAT KO[/red]"
    )
    dag_icon = "[green]DAG OK[/green]" if record["dag_valid"] else "[red]DAG KO[/red]"
    fb_icon = "[yellow](FALLBACK)[/yellow]" if record["is_fallback"] else ""

    # Construit la liste des etapes pour affichage
    steps_text = (
        "\n".join(f"  [dim]{s}[/dim]" for s in record["steps_summary"])
        or "  [dim](aucune etape)[/dim]"
    )

    content = (
        f"[bold]Q:[/bold] {record['question'][:100]}\n"
        f"[bold]Type:[/bold] {record['query_type']}  |  "
        f"[bold]Etapes:[/bold] {record['n_steps']}  |  "
        f"[bold]Latence:[/bold] {record['latency_ms']:.0f}ms\n"
        f"[bold]Statut:[/bold] {toon_icon}  {dag_icon}  {fb_icon}\n\n"
        f"[bold]Plan genere :[/bold]\n{steps_text}"
    )

    border_color = "green" if record["toon_valid"] and record["dag_valid"] else "red"
    console.print(
        Panel(
            content,
            title=f"[bold]#{idx:02d} — {record['hotpot_type'].upper()}[/bold]",
            border_style=border_color,
            expand=False,
            width=min(console.width, 100),
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────


def run_evaluation() -> None:
    """Point d'entree du script d'evaluation du Planner."""

    # ── Step 1 : Chargement et filtrage du dataset ────────────────────────────
    console.print(
        "\n[bold cyan][STEP 1][/bold cyan] Chargement et filtrage du dataset..."
    )

    if not INPUT_FILE.exists():
        console.print(f"[red][ERREUR] Fichier introuvable : {INPUT_FILE}[/red]")
        console.print(
            "[yellow]Lancez d'abord : uv run python scripts/prepare_hotpotqa.py[/yellow]"
        )
        sys.exit(1)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        all_records: list[dict[str, str]] = json.load(f)

    # Garder uniquement les requetes complexes (bridge ET comparison)
    complex_records = [r for r in all_records if r["type"] in HOTPOT_TYPE_MAP]
    console.print(
        f"   [green][OK][/green] {len(all_records)} requetes totales → "
        f"{len(complex_records)} requetes complexes filtrees"
    )

    # Echantillonnage reproductible
    random.seed(RANDOM_SEED)
    sample = random.sample(complex_records, min(SAMPLE_SIZE, len(complex_records)))
    console.print(
        f"   [green][OK][/green] Echantillon : {len(sample)} requetes "
        f"(seed={RANDOM_SEED})\n"
    )

    # ── Step 2 : Initialisation du Planner ────────────────────────────────────
    console.print("[bold cyan][STEP 2][/bold cyan] Initialisation du Planner...")
    planner = Planner()
    console.print(
        f"   [green][OK][/green] Modele : {planner.model} | "
        f"api_base : {planner.api_base}\n"
    )

    # ── Step 3 : Boucle d'evaluation avec barre de progression ────────────────
    console.print("[bold cyan][STEP 3][/bold cyan] Evaluation en cours...\n")

    eval_results: list[PlannerEvalRecord] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Decomposition des requetes", total=len(sample))

        for idx, row in enumerate(sample, start=1):
            question: str = row["question"]
            hotpot_type: str = row["type"]
            query_type: QueryType = HOTPOT_TYPE_MAP[hotpot_type]
            budget: int = BUDGET_MAP[query_type]

            # Construction de l'AnalysisResult factice (ground truth)
            # On utilise les vraies valeurs du dataset, pas la prediction du Analyzer
            analysis = AnalysisResult(
                query_type=query_type,
                confidence=0.90,
                detected_entities=[],
                reasoning_budget=budget,
            )

            # Inference avec mesure de latence
            toon_valid = True  # suppose vrai, mis a False si le Planner fallback
            t_start = time.perf_counter()
            try:
                plan: ExecutionPlan = planner.decompose(
                    query=question, analysis=analysis
                )
            except Exception as exc:  # noqa: BLE001 — catch-all defensif
                console.print(f"\n[red][WARN] Erreur sur idx={idx}: {exc}[/red]")
                # En cas d'echec complet, on cree un plan vide pour continuer
                from reasoning.contracts.internal_models import PlanStep, StepStatus

                plan = ExecutionPlan(
                    plan_id=f"error-{idx}",
                    original_query=question,
                    steps=[
                        PlanStep(
                            step_id="step_1",
                            sub_query=question,
                            depends_on=[],
                            status=StepStatus.PENDING,
                        )
                    ],
                    dependencies_graph={"step_1": []},
                )
                toon_valid = False

            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000

            # Validation DAG independante
            dag_ok, dag_reason = validate_dag(plan)

            # Detection du fallback sequentiel
            is_fallback = detect_fallback(plan)

            # Si fallback utilise = le LLM n'a pas respecte le format TOON
            if is_fallback and toon_valid:
                toon_valid = False  # fallback = echec de parsing TOON

            # Resume des etapes pour affichage
            steps_summary = [
                f"{s.step_id}: {s.sub_query[:70]}"
                f"{'...' if len(s.sub_query) > 70 else ''}"
                f" [depends_on={s.depends_on}]"
                for s in plan.steps
            ]

            record = PlannerEvalRecord(
                id=row["id"],
                question=question,
                hotpot_type=hotpot_type,
                query_type=str(query_type),
                n_steps=len(plan.steps),
                toon_valid=toon_valid,
                dag_valid=dag_ok,
                is_fallback=is_fallback,
                latency_ms=round(latency_ms, 1),
                steps_summary=steps_summary,
            )
            eval_results.append(record)
            progress.advance(task)

    # ── Step 4 : Affichage des panels par requete ─────────────────────────────
    console.print("\n[bold cyan][STEP 4][/bold cyan] Resultats par requete :\n")
    for idx, record in enumerate(eval_results, start=1):
        render_plan_panel(idx, record)

    # ── Step 5 : Calcul des metriques globales ────────────────────────────────
    total = len(eval_results)
    n_toon_valid = sum(1 for r in eval_results if r["toon_valid"])
    n_dag_valid = sum(1 for r in eval_results if r["dag_valid"])
    n_fallback = sum(1 for r in eval_results if r["is_fallback"])
    n_both_valid = sum(1 for r in eval_results if r["toon_valid"] and r["dag_valid"])

    # Par type de requete
    bridge_res = [r for r in eval_results if r["hotpot_type"] == "bridge"]
    comp_res = [r for r in eval_results if r["hotpot_type"] == "comparison"]

    avg_latency = sum(r["latency_ms"] for r in eval_results) / total
    avg_steps = sum(r["n_steps"] for r in eval_results) / total

    toon_pct = n_toon_valid / total * 100
    dag_pct = n_dag_valid / total * 100
    both_pct = n_both_valid / total * 100

    bridge_toon = sum(1 for r in bridge_res if r["toon_valid"])
    comp_toon = sum(1 for r in comp_res if r["toon_valid"])

    # ── Step 6 : Sauvegarde JSON ──────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "summary": {
            "total": total,
            "toon_valid": n_toon_valid,
            "dag_valid": n_dag_valid,
            "both_valid": n_both_valid,
            "fallback_count": n_fallback,
            "toon_validity_pct": round(toon_pct, 2),
            "dag_validity_pct": round(dag_pct, 2),
            "full_validity_pct": round(both_pct, 2),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_steps_per_plan": round(avg_steps, 2),
        },
        "results": list(eval_results),
    }
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    console.print(
        f"\n   [green][OK][/green] Resultats sauvegardes dans "
        f"{OUTPUT_FILE.relative_to(BASE_DIR)}"
    )

    # ── Step 7 : Tableau recapitulatif rich ────────────────────────────────────

    def _pct_color(val: float) -> str:
        """Couleur dynamique selon le pourcentage."""
        if val >= 70:
            return "green"
        if val >= 50:
            return "yellow"
        return "red"

    table = Table(
        title="\nResultats de l'evaluation — Planner (Plan-and-Solve)",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        min_width=65,
    )
    table.add_column("Metrique", style="bold", justify="left", min_width=38)
    table.add_column("Valeur", justify="right", min_width=22)

    table.add_row("Total requetes evaluees", str(total))
    table.add_row("Requetes bridge (MULTI_HOP)", str(len(bridge_res)))
    table.add_row("Requetes comparison (COMPARATIVE)", str(len(comp_res)))
    table.add_row("", "")  # separateur visuel

    tc = _pct_color(toon_pct)
    table.add_row(
        "Taux validite TOON (format <<<...>>>)",
        f"[{tc}]{toon_pct:.1f}% ({n_toon_valid}/{total})[/{tc}]",
    )
    dc = _pct_color(dag_pct)
    table.add_row(
        "Taux DAG valides (sans cycle, refs OK)",
        f"[{dc}]{dag_pct:.1f}% ({n_dag_valid}/{total})[/{dc}]",
    )
    bc = _pct_color(both_pct)
    table.add_row(
        "Taux plans 100% valides (TOON + DAG)",
        f"[{bc}]{both_pct:.1f}% ({n_both_valid}/{total})[/{bc}]",
    )
    table.add_row("", "")

    if bridge_res:
        bt = bridge_toon / len(bridge_res) * 100
        btc = _pct_color(bt)
        table.add_row(
            "Validite TOON bridge (MULTI_HOP)",
            f"[{btc}]{bt:.1f}% ({bridge_toon}/{len(bridge_res)})[/{btc}]",
        )
    if comp_res:
        ct = comp_toon / len(comp_res) * 100
        ctc = _pct_color(ct)
        table.add_row(
            "Validite TOON comparison (COMPARATIVE)",
            f"[{ctc}]{ct:.1f}% ({comp_toon}/{len(comp_res)})[/{ctc}]",
        )

    table.add_row("", "")
    table.add_row("Fallbacks sequentiels detectes", str(n_fallback))
    table.add_row("Nombre moyen d'etapes par plan", f"{avg_steps:.1f}")
    table.add_row("Latence moyenne par requete", f"{avg_latency:.0f} ms")
    table.add_row(
        "Resultats sauvegardes dans",
        str(OUTPUT_FILE.relative_to(BASE_DIR)),
    )

    console.print(table)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
