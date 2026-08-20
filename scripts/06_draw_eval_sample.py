"""
Tirage de l'échantillon d'évaluation de bout en bout — Sprint I5-A.
Projet : RAG-REASON (intégration avec le module ACTION)

Tire 20 questions parmi les 200 du jeu d'évaluation, stratifiées
bridge/comparison, avec une graine fixe. La liste d'identifiants est écrite
dans `tests/evaluation/e2e_sample.json` et VERSIONNÉE : sans elle, deux
campagnes ne porteraient pas sur les mêmes questions et leurs résultats ne
seraient pas comparables.

POURQUOI 20 ET PAS PLUS
-----------------------
Contrainte de coût, mesurée et non supposée. Latences totales relevées sur
des exécutions complètes du graphe (Sprints I4, Lots A et B) :

    profil       exécutions mesurées                    moyenne
    SIMPLE       256 s · 185 s                          ~220 s
    MULTI_HOP    741 s · 710 s · 675 s · 662 s          ~697 s

Le jeu est composé à parts égales de bridge et comparison, tous multi-hop par
construction. En retenant la moyenne MULTI_HOP comme référence :

    20 questions x ~697 s  ~=  3 h 52 min pour UNE campagne
    200 questions          ~=  38 h 45 min

Un passage sur les 200 questions est hors d'atteinte sur cette machine. 20
questions tiennent dans une demi-journée, ce qui autorise plusieurs campagnes
comparatives — indispensable puisque le LLM reste non déterministe malgré
`temperature=0`.

STRATIFICATION
--------------
10 bridge + 10 comparison, soit exactement la proportion du jeu complet
(100/100). Le tirage est restreint aux questions dont les articles gold sont
INTÉGRALEMENT présents dans le corpus : mesurer `context_recall` sur une
question dont la réponse est absente ne dirait rien du module REASONING.

Usage :
    uv run python scripts/06_draw_eval_sample.py
"""

from __future__ import annotations

import io
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_SET = BASE_DIR / "data" / "processed" / "hotpotqa_sprint3.json"
MANIFEST = BASE_DIR / "data" / "corpus" / "corpus_manifest.json"
OUTPUT = BASE_DIR / "tests" / "evaluation" / "e2e_sample.json"

SEED = 20260820
PER_STRATUM = 10

# Latences moyennes mesurées, en secondes (voir la docstring de module).
MEAN_SECONDS_MULTI_HOP = 697


def main() -> None:
    """Point d'entrée : filtre, stratifie, tire, écrit."""
    print("\n[STEP 1] Chargement du jeu d'évaluation et du manifeste...")
    records: list[dict[str, Any]] = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    complete = {q["id"] for q in manifest["questions"] if q["is_complete"]}
    gold_of = {q["id"]: q["gold_titles"] for q in manifest["questions"]}
    print(f"   {len(records)} questions, {len(complete)} à gold complet")

    eligible = [r for r in records if r["id"] in complete]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in eligible:
        by_type.setdefault(record["type"], []).append(record)
    for questions in by_type.values():
        questions.sort(key=lambda r: r["id"])  # ordre déterministe avant tirage

    print("\n[STEP 2] Strates disponibles :")
    for label, questions in sorted(by_type.items()):
        print(f"   {label:12} {len(questions):>4} éligibles")

    print(f"\n[STEP 3] Tirage stratifié (graine {SEED}, {PER_STRATUM} par strate)...")
    rng = random.Random(SEED)
    drawn: list[dict[str, Any]] = []
    for label in sorted(by_type):
        pool = by_type[label]
        if len(pool) < PER_STRATUM:
            raise SystemExit(f"strate {label} : {len(pool)} < {PER_STRATUM} requis")
        drawn.extend(rng.sample(pool, PER_STRATUM))
    drawn.sort(key=lambda r: (r["type"], r["id"]))

    counts = Counter(r["type"] for r in drawn)
    print(f"   {len(drawn)} questions tirées : {dict(counts)}")

    estimate_s = len(drawn) * MEAN_SECONDS_MULTI_HOP
    payload = {
        "seed": SEED,
        "drawn_from": EVAL_SET.name,
        "n": len(drawn),
        "stratification": dict(counts),
        "selection_rule": (
            "questions dont les articles gold sont intégralement présents dans "
            "le corpus (corpus_manifest.json, is_complete), triées par id puis "
            f"échantillonnées par random.Random({SEED}).sample sur chaque strate"
        ),
        "cost_estimate": {
            "mean_seconds_per_question": MEAN_SECONDS_MULTI_HOP,
            "total_seconds": estimate_s,
            "total_human": f"{estimate_s // 3600} h {(estimate_s % 3600) // 60} min",
            "basis": (
                "latences totales mesurées sur exécutions complètes du graphe : "
                "MULTI_HOP 741/710/675/662 s, SIMPLE 256/185 s (Sprints I4, "
                "Lots A et B)"
            ),
        },
        "questions": [
            {
                "id": r["id"],
                "type": r["type"],
                "level": r["level"],
                "question": r["question"],
                "gold_titles": gold_of[r["id"]],
            }
            for r in drawn
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n[STEP 4] Échantillon :")
    for r in drawn:
        print(f"   {r['id']}  [{r['type']:10} / {r['level']:6}]  {r['question'][:62]}")

    print("\n" + "=" * 70)
    print(f"  écrit dans        : {OUTPUT.relative_to(BASE_DIR)}")
    print(f"  graine            : {SEED}")
    print(f"  stratification    : {dict(counts)}")
    print(
        f"  coût estimé       : {payload['cost_estimate']['total_human']} par campagne"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
