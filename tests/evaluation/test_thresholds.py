"""
Tests de non-régression de l'évaluation — Sprint I5-C.

Comparent les rapports de campagne aux planchers de
`configs/evaluation_thresholds.toml`. Un test rouge ici signale qu'une
modification du moteur a dégradé une métrique au-delà du bruit de mesure.

CE QUE CES TESTS NE SONT PAS
=============================
Ce ne sont pas des tests de qualité. Les seuils valent la mesure du Sprint
I5-B moins 5 points de marge, sur un échantillon de 20 questions, avec un
juge LLM qui est aussi le générateur. Ils bornent une DÉRIVE ; ils ne
certifient aucun niveau.

Ils ne relancent pas non plus de campagne : ils relisent les rapports
existants. Une campagne coûte des heures et ne peut pas vivre dans une suite
de tests unitaires. Si les rapports sont absents, les tests sont ignorés
explicitement, avec la commande à lancer.

Exécution :
    uv run pytest tests/evaluation/test_thresholds.py -v
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

BASE_DIR = Path(__file__).resolve().parents[2]
THRESHOLDS = BASE_DIR / "configs" / "evaluation_thresholds.toml"
COMPARISON = BASE_DIR / "tests" / "evaluation" / "reports" / "comparison_v1.json"
VERIFIER_REPORT = (
    BASE_DIR / "tests" / "evaluation" / "reports" / "verifier_metrics.json"
)

_REBUILD = (
    "Relancer les campagnes puis le calcul :\n"
    "  uv run python tests/evaluation/run_evaluation.py naive\n"
    "  uv run python tests/evaluation/run_evaluation.py full\n"
    "  uv run python tests/evaluation/compute_metrics.py"
)


@pytest.fixture(scope="module")
def thresholds() -> dict[str, Any]:
    """Seuils déclarés. Leur absence est une erreur, pas un cas à ignorer."""
    assert THRESHOLDS.is_file(), f"fichier de seuils absent : {THRESHOLDS}"
    with THRESHOLDS.open("rb") as handle:
        loaded: dict[str, Any] = tomllib.load(handle)
    return loaded


@pytest.fixture(scope="module")
def comparison() -> dict[str, Any]:
    """Rapport comparatif du Sprint I5-B."""
    if not COMPARISON.is_file():
        pytest.skip(
            f"{COMPARISON.name} absent — aucune campagne à contrôler.\n{_REBUILD}"
        )
    payload: dict[str, Any] = json.loads(COMPARISON.read_text(encoding="utf-8"))
    return payload


def _fail_message(name: str, measured: float, floor: float, scope: str) -> str:
    """Message d'échec portant l'écart, pas seulement le verdict."""
    return (
        f"{scope} · {name} = {measured:.4f}, sous le plancher {floor:.4f} "
        f"(écart {(measured - floor) * 100:+.1f} points).\n"
        "Soit le moteur a régressé, soit le seuil doit être révisé — mais "
        "alors avec une mesure à l'appui, pas par confort."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Métriques absolues du système complet
# ─────────────────────────────────────────────────────────────────────────────


class TestFullSystemFloors:
    """Le système complet reste au-dessus de ses planchers mesurés."""

    @pytest.mark.parametrize(
        "metric",
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
    )
    def test_global_metric_above_floor(
        self, thresholds: dict[str, Any], comparison: dict[str, Any], metric: str
    ) -> None:
        """Chaque métrique globale dépasse son plancher."""
        floor = float(thresholds["full"][metric])
        measured = comparison["aggregate"]["full"][metric]

        assert measured is not None, f"{metric} non calculée sur le système complet"
        assert measured >= floor, _fail_message(metric, measured, floor, "global")

    @pytest.mark.parametrize("metric", ["answer_relevancy", "context_recall"])
    def test_bridge_metric_above_floor(
        self, thresholds: dict[str, Any], comparison: dict[str, Any], metric: str
    ) -> None:
        """Sur les bridge — le sous-ensemble multi-hop du cahier des charges."""
        floor = float(thresholds["full_bridge"][metric])
        measured = comparison["by_type"]["full"]["bridge"][metric]

        assert measured is not None, f"{metric} non calculée sur les bridge"
        assert measured >= floor, _fail_message(metric, measured, floor, "bridge")


# ─────────────────────────────────────────────────────────────────────────────
# Écarts par rapport au baseline
# ─────────────────────────────────────────────────────────────────────────────


class TestGainsOverBaseline:
    """Les écarts au baseline naïf restent au-dessus de leurs planchers.

    Ce sont eux qui portent le sens : le biais du juge LLM est commun aux
    deux systèmes, donc il s'annule dans la différence.
    """

    @staticmethod
    def _gain(comparison: dict[str, Any], scope: str, metric: str) -> float:
        """Écart système complet − baseline, en points."""
        if scope == "global":
            full = comparison["aggregate"]["full"][metric]
            naive = comparison["aggregate"]["naive"][metric]
        else:
            full = comparison["by_type"]["full"][scope][metric]
            naive = comparison["by_type"]["naive"][scope][metric]
        assert full is not None and naive is not None, f"{metric} manquante ({scope})"
        return round((float(full) - float(naive)) * 100, 1)

    @pytest.mark.parametrize(
        "metric", ["faithfulness", "answer_relevancy", "context_recall"]
    )
    def test_global_gain_above_floor(
        self, thresholds: dict[str, Any], comparison: dict[str, Any], metric: str
    ) -> None:
        """Écarts globaux.

        Le plancher de `faithfulness` est NÉGATIF (−5) : l'écart mesuré vaut
        −1,7 point. Il borne une dégradation supplémentaire, il n'affirme pas
        que l'écart soit favorable.
        """
        floor = float(thresholds["gains_vs_naive"][f"{metric}_points"])
        gain = self._gain(comparison, "global", metric)

        assert gain >= floor, (
            f"global · gain de {metric} = {gain:+.1f} points, sous le plancher "
            f"{floor:+.1f}."
        )

    @pytest.mark.parametrize("metric", ["answer_relevancy", "context_recall"])
    def test_bridge_gain_above_floor(
        self, thresholds: dict[str, Any], comparison: dict[str, Any], metric: str
    ) -> None:
        """Écarts sur les bridge — l'amélioration multi-hop revendiquée."""
        floor = float(thresholds["gains_vs_naive_bridge"][f"{metric}_points"])
        gain = self._gain(comparison, "bridge", metric)

        assert gain >= floor, (
            f"bridge · gain de {metric} = {gain:+.1f} points, sous le plancher "
            f"{floor:+.1f}."
        )

    def test_faithfulness_floor_is_not_an_unmet_target(
        self, thresholds: dict[str, Any], comparison: dict[str, Any]
    ) -> None:
        """Garde-fou méthodologique : le plancher reste SOUS la mesure.

        Un seuil placé au-dessus de ce qui a été mesuré n'est pas un
        garde-fou — c'est un objectif non atteint déguisé, et le test
        échouerait dès sa création sans rien apprendre.
        """
        floor = float(thresholds["gains_vs_naive"]["faithfulness_points"])
        measured = self._gain(comparison, "global", "faithfulness")

        assert floor <= measured, (
            f"le plancher {floor:+.1f} dépasse la mesure {measured:+.1f} : "
            "ce n'est plus un seuil de non-régression."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Coût
# ─────────────────────────────────────────────────────────────────────────────


class TestCostCeilings:
    """Le compromis qualité/coût : bornes HAUTES sur les appels LLM."""

    def test_mean_llm_calls_below_ceiling(
        self, thresholds: dict[str, Any], comparison: dict[str, Any]
    ) -> None:
        """La moyenne d'appels par question reste sous son plafond."""
        ceiling = float(thresholds["cost"]["mean_llm_calls_per_question"])
        measured = comparison["aggregate"]["full"]["llm_calls_mean"]

        assert measured is not None, "appels LLM non relevés"
        assert measured <= ceiling, (
            f"{measured} appels LLM par question en moyenne, au-dessus du "
            f"plafond {ceiling} — dérive du coût."
        )

    def test_no_question_exceeds_the_configured_cap(
        self, thresholds: dict[str, Any]
    ) -> None:
        """Aucune question ne dépasse `MAX_LLM_CALLS_PER_QUERY`.

        Contrôle du plafond du Sprint I5-A sur données réelles : s'il était
        inopérant, une question l'aurait franchi au cours de la campagne.
        """
        results = BASE_DIR / "tests" / "evaluation" / "results" / "run_full.json"
        if not results.is_file():
            pytest.skip(f"{results.name} absent.\n{_REBUILD}")

        cap = int(thresholds["cost"]["max_llm_calls_per_question"])
        payload = json.loads(results.read_text(encoding="utf-8"))
        offenders = [
            (r["id"], r.get("llm_calls"))
            for r in payload["results"]
            if (r.get("llm_calls") or 0) > cap
        ]

        assert offenders == [], f"questions au-dessus du plafond {cap} : {offenders}"


# ─────────────────────────────────────────────────────────────────────────────
# Verifier — actif seulement une fois l'annotation faite
# ─────────────────────────────────────────────────────────────────────────────


class TestVerifierFloors:
    """Précision et rappel du Verifier face à l'annotation humaine."""

    @pytest.mark.parametrize("metric", ["precision", "recall", "f1"])
    def test_verifier_metric_above_floor(
        self, thresholds: dict[str, Any], metric: str
    ) -> None:
        """Chaque métrique du Verifier dépasse son plancher.

        Ignoré tant que `[verifier] enabled = false` : les seuils ne sont
        fixés qu'APRÈS l'annotation des 50 exemples. Un seuil inventé avant
        mesure ne vaut rien.
        """
        config = thresholds["verifier"]
        if not config.get("enabled", False):
            pytest.skip(
                "seuils du Verifier non encore établis — annoter les 50 "
                "exemples via l'onglet Annotation, puis lancer "
                "`uv run python tests/evaluation/verifier_metrics.py`."
            )
        if not VERIFIER_REPORT.is_file():
            pytest.skip(f"{VERIFIER_REPORT.name} absent.")

        report = json.loads(VERIFIER_REPORT.read_text(encoding="utf-8"))
        floor = float(config[metric])
        measured = float(report["overall"][metric])

        assert measured >= floor, _fail_message(metric, measured, floor, "verifier")


# ─────────────────────────────────────────────────────────────────────────────
# Cohérence du fichier de seuils lui-même
# ─────────────────────────────────────────────────────────────────────────────


class TestThresholdsFileIsSane:
    """Le fichier de seuils est lui-même contrôlé."""

    def test_metadata_is_present(self, thresholds: dict[str, Any]) -> None:
        """La provenance des seuils est déclarée, pas implicite."""
        meta = thresholds["meta"]

        assert meta["sample_size"] == 20
        assert meta["source"].endswith("comparison_v1.json")
        assert float(meta["noise_margin"]) > 0

    def test_every_floor_is_a_probability(self, thresholds: dict[str, Any]) -> None:
        """Les seuils de métriques RAGAS restent dans [0, 1]."""
        for section in ("full", "full_bridge"):
            for name, value in thresholds[section].items():
                assert 0.0 <= float(value) <= 1.0, f"{section}.{name} = {value}"
