"""
Précision et rappel du Verifier — Sprint I5-C, étape 3.

Confronte les 50 labels humains de `verifier_annotation_set.json` aux
verdicts `is_grounded` produits par le Verifier. C'est la seule mesure
INDÉPENDANTE de ce composant : les métriques RAGAS du Sprint I5-B ont pour
juge le modèle qui a aussi généré les réponses.

CONVENTION — CE QUE L'ON CHERCHE À DÉTECTER
============================================
La classe POSITIVE est `hallucinated`. Le Verifier est un détecteur
d'hallucinations : ce qu'on lui demande, c'est de lever une alerte quand une
réponse n'est pas soutenue par ses sources.

    vrai positif   hallucination réelle, Verifier a alerté
                   (human=hallucinated, is_grounded=False)
    faux positif   réponse fondée, Verifier a alerté à tort
                   (human=grounded,     is_grounded=False)
    faux négatif   hallucination MANQUÉE
                   (human=hallucinated, is_grounded=True)
    vrai négatif   réponse fondée, Verifier l'a validée
                   (human=grounded,     is_grounded=True)

Le **rappel** est ici la métrique critique : un faux négatif est une
hallucination qui atteint l'utilisateur. Un faux positif ne coûte qu'un refus
excessif.

VENTILATION RÉEL / SYNTHÉTIQUE
------------------------------
Les deux populations ne sont pas comparables et sont donc mesurées à part.
Les cas synthétiques portent une hallucination INJECTÉE, dont on connaît la
nature exacte — ils mesurent la sensibilité du Verifier à des défauts
caractérisés. Les cas réels mesurent son comportement en usage.

Usage :
    uv run python tests/evaluation/verifier_metrics.py
"""

from __future__ import annotations

import io
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

BASE_DIR = Path(__file__).resolve().parents[2]
ANNOTATION_SET = BASE_DIR / "tests" / "evaluation" / "verifier_annotation_set.json"
REPORT = BASE_DIR / "tests" / "evaluation" / "reports" / "verifier_metrics.json"

GROUNDED = "grounded"
HALLUCINATED = "hallucinated"


class Confusion(NamedTuple):
    """Matrice de confusion, classe positive = `hallucinated`."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    @property
    def precision(self) -> float | None:
        """Parmi les alertes levées, combien sont justifiées."""
        alerts = self.true_positive + self.false_positive
        return self.true_positive / alerts if alerts else None

    @property
    def recall(self) -> float | None:
        """Parmi les hallucinations réelles, combien sont détectées."""
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else None

    @property
    def f1(self) -> float | None:
        """Moyenne harmonique de la précision et du rappel."""
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float | None:
        """Part de verdicts corrects, toutes classes confondues."""
        if not self.total:
            return None
        return (self.true_positive + self.true_negative) / self.total


def confusion_of(entries: list[dict[str, Any]]) -> Confusion:
    """Construit la matrice de confusion d'un groupe d'entrées."""
    tp = fp = fn = tn = 0
    for entry in entries:
        human = entry["human_label"]
        # `is_grounded=False` = le Verifier alerte = il prédit `hallucinated`.
        predicted_hallucination = entry["verifier_verdict"] is False
        if human == HALLUCINATED:
            if predicted_hallucination:
                tp += 1
            else:
                fn += 1
        else:
            if predicted_hallucination:
                fp += 1
            else:
                tn += 1
    return Confusion(tp, fp, fn, tn)


def _fmt(value: float | None) -> str:
    """Formate une métrique, ou `n/d` si elle n'est pas définie."""
    return f"{value:.4f}" if value is not None else "n/d"


def _as_dict(confusion: Confusion) -> dict[str, Any]:
    """Sérialise une matrice et ses métriques dérivées."""
    return {
        "n": confusion.total,
        "confusion": {
            "true_positive": confusion.true_positive,
            "false_positive": confusion.false_positive,
            "false_negative": confusion.false_negative,
            "true_negative": confusion.true_negative,
        },
        "precision": confusion.precision,
        "recall": confusion.recall,
        "f1": confusion.f1,
        "accuracy": confusion.accuracy,
    }


def _print_matrix(label: str, confusion: Confusion) -> None:
    """Affiche une matrice de confusion et ses métriques."""
    print(f"\n  ── {label} (n = {confusion.total}) " + "─" * max(0, 44 - len(label)))
    print("                         Verifier : alerte   Verifier : fondée")
    print(
        f"    humain hallucinée         {confusion.true_positive:>6} (VP)"
        f"      {confusion.false_negative:>6} (FN)"
    )
    print(
        f"    humain fondée             {confusion.false_positive:>6} (FP)"
        f"      {confusion.true_negative:>6} (VN)"
    )
    print(
        f"    précision {_fmt(confusion.precision)}"
        f"  ·  rappel {_fmt(confusion.recall)}"
        f"  ·  F1 {_fmt(confusion.f1)}"
        f"  ·  exactitude {_fmt(confusion.accuracy)}"
    )


def main() -> None:
    """Point d'entrée : contrôle, mesure, rapporte."""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if not ANNOTATION_SET.is_file():
        raise SystemExit(
            f"{ANNOTATION_SET.name} absent. Le construire par :\n"
            "  uv run python tests/evaluation/build_annotation_set.py"
        )

    payload = json.loads(ANNOTATION_SET.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = payload["entries"]

    # ── Contrôles préalables ─────────────────────────────────────────────
    unlabelled = [e["id"] for e in entries if not e.get("human_label")]
    if unlabelled:
        raise SystemExit(
            f"{len(unlabelled)}/{len(entries)} exemples non annotés. "
            "Compléter l'annotation via l'onglet « Annotation » du dashboard :\n"
            "  uv run streamlit run frontend/app.py\n"
            f"  premiers manquants : {unlabelled[:5]}"
        )

    bad_labels = [
        (e["id"], e["human_label"])
        for e in entries
        if e["human_label"] not in (GROUNDED, HALLUCINATED)
    ]
    if bad_labels:
        raise SystemExit(f"labels hors nomenclature : {bad_labels}")

    # Une entrée sans verdict est ÉCARTÉE plutôt que comptée à tort : le
    # Verifier a échoué dessus, il n'a pas prédit « fondée ».
    usable = [e for e in entries if e["verifier_verdict"] is not None]
    discarded = [e["id"] for e in entries if e["verifier_verdict"] is None]
    if discarded:
        print(
            f"\n  [!] {len(discarded)} entrée(s) écartée(s), verdict du Verifier "
            f"indisponible : {discarded}"
        )

    real = [e for e in usable if e["origin"] == "real"]
    synthetic = [e for e in usable if e["origin"] == "synthetic"]

    overall = confusion_of(usable)
    real_confusion = confusion_of(real)
    synthetic_confusion = confusion_of(synthetic)

    print("\n" + "=" * 72)
    print("  PRÉCISION ET RAPPEL DU VERIFIER — classe positive = `hallucinated`")
    print("=" * 72)
    _print_matrix("Ensemble", overall)
    _print_matrix("Cas réels", real_confusion)
    _print_matrix("Cas synthétiques (hallucinations injectées)", synthetic_confusion)

    # ── Concordance sur les synthétiques ─────────────────────────────────
    # `expected_label` vaut `hallucinated` par construction. Un désaccord
    # signale soit une injection sans effet perceptible, soit une erreur
    # d'annotation — dans les deux cas, une information à ne pas taire.
    mismatches = [
        {
            "id": e["id"],
            "kind": (e.get("transformation") or {}).get("kind"),
            "expected": e["expected_label"],
            "human": e["human_label"],
            "rationale": (e.get("transformation") or {}).get("rationale"),
        }
        for e in synthetic
        if e["expected_label"] and e["human_label"] != e["expected_label"]
    ]
    agreement = 1 - len(mismatches) / len(synthetic) if synthetic else None

    print("\n  ── Concordance label attendu / label humain (synthétiques) ──")
    print(
        f"    concordance : {_fmt(agreement)}  "
        f"({len(synthetic) - len(mismatches)}/{len(synthetic)})"
    )
    if mismatches:
        print("    désaccords — injection sans effet, ou erreur d'annotation :")
        for m in mismatches:
            print(
                f"      {m['id']} ({m['kind']}) : attendu={m['expected']}, "
                f"humain={m['human']}"
            )
            print(f"          {m['rationale']}")
    else:
        print("    aucun désaccord : les 25 injections ont bien été perçues.")

    # ── Ventilation par type d'injection ─────────────────────────────────
    by_kind: dict[str, Confusion] = {}
    for kind in ("addition", "contradiction", "substitution"):
        group = [
            e for e in synthetic if (e.get("transformation") or {}).get("kind") == kind
        ]
        if group:
            by_kind[kind] = confusion_of(group)

    print("\n  ── Détection par type d'injection ──")
    for kind, conf in by_kind.items():
        print(f"    {kind:14} n={conf.total:>2}  rappel {_fmt(conf.recall)}")

    # ── Répartition des labels ───────────────────────────────────────────
    counts = Counter(e["human_label"] for e in usable)
    print("\n  ── Répartition des labels humains ──")
    for label, count in sorted(counts.items()):
        print(f"    {label:14} {count:>2} / {len(usable)}")

    # ── Rapport ──────────────────────────────────────────────────────────
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "positive_class": HALLUCINATED,
                "n_entries": len(entries),
                "n_usable": len(usable),
                "discarded": discarded,
                "overall": _as_dict(overall),
                "real": _as_dict(real_confusion),
                "synthetic": _as_dict(synthetic_confusion),
                "synthetic_by_injection_kind": {
                    kind: _as_dict(conf) for kind, conf in by_kind.items()
                },
                "expected_vs_human_agreement": agreement,
                "expected_vs_human_mismatches": mismatches,
                "label_distribution": dict(counts),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n  rapport écrit : {REPORT.relative_to(BASE_DIR)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
