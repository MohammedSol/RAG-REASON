"""
Jeu de données d'évaluation — Sprint I5-B.

Construit et valide `dataset_v1.json` : les 20 questions de `e2e_sample.json`,
augmentées de leur vérité terrain (réponse attendue, passages attendus).

SOURCES DE LA VÉRITÉ TERRAIN
============================
`e2e_sample.json` ne porte que les questions et les titres gold. La réponse
et les phrases justificatives viennent du dataset HotpotQA lui-même, split
`distractor` — le même que celui dont le corpus a été extrait au Sprint I1 :

    ground_truth_answer    <- champ `answer` de HotpotQA
    ground_truth_contexts  <- phrases désignées par `supporting_facts`
                              (couple titre + indice de phrase), extraites
                              du `context` de la question

Les `supporting_facts` désignent les phrases PRÉCISES qui justifient la
réponse, pas les articles entiers. C'est ce que `context_recall` doit
mesurer : le système a-t-il ramené l'information nécessaire, et non
simplement le bon article.

Usage :
    uv run python tests/evaluation/dataset.py          # construit et valide
    uv run python tests/evaluation/dataset.py --check  # valide seulement
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parents[2]
SAMPLE = BASE_DIR / "tests" / "evaluation" / "e2e_sample.json"
MANIFEST = BASE_DIR / "data" / "corpus" / "corpus_manifest.json"
CORPUS_DIR = BASE_DIR / "data" / "corpus"
DATASET = BASE_DIR / "tests" / "evaluation" / "dataset_v1.json"

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"


class EvaluationSample(BaseModel):
    """Une question du jeu d'évaluation, avec sa vérité terrain.

    Attributes:
        id: Identifiant HotpotQA, clé de jointure avec `e2e_sample.json`.
        question: La question posée, verbatim.
        type: `bridge` ou `comparison` — sert à la ventilation des résultats.
        level: Niveau de difficulté déclaré par HotpotQA.
        ground_truth_answer: La réponse attendue.
        ground_truth_contexts: Les phrases justificatives (supporting_facts).
        gold_titles: Titres des articles gold.
        gold_sources: Noms de fichiers correspondants dans le corpus indexé.
    """

    id: str
    question: str
    type: str
    level: str
    ground_truth_answer: str
    ground_truth_contexts: list[str] = Field(min_length=1)
    gold_titles: list[str] = Field(min_length=1)
    gold_sources: list[str] = Field(min_length=1)

    @field_validator("question", "ground_truth_answer")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Refuse une question ou une réponse vide."""
        if not value.strip():
            raise ValueError("champ vide")
        return value

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        """Refuse un type hors de la taxonomie HotpotQA."""
        if value not in {"bridge", "comparison"}:
            raise ValueError(f"type inattendu : {value!r}")
        return value

    @field_validator("gold_sources")
    @classmethod
    def _sources_exist_on_disk(cls, value: list[str]) -> list[str]:
        """Chaque source gold doit être un fichier réel du corpus indexé.

        Sans ce contrôle, `context_recall` mesurerait la capacité du système
        à retrouver des documents qui ne sont pas dans l'index.
        """
        missing = [name for name in value if not (CORPUS_DIR / name).is_file()]
        if missing:
            raise ValueError(f"fichiers absents du corpus : {missing}")
        return value


def build() -> list[EvaluationSample]:
    """Construit le jeu de données depuis l'échantillon et HotpotQA."""
    from datasets import load_dataset

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    title_to_file: dict[str, str] = manifest["title_to_file"]

    wanted = {q["id"]: q for q in sample["questions"]}
    print(f"   {len(wanted)} questions dans {SAMPLE.name}")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    position = {qid: i for i, qid in enumerate(dataset["id"])}

    built: list[EvaluationSample] = []
    for qid, entry in wanted.items():
        row = dataset[position[qid]]
        context = row["context"]
        sentences_by_title = dict(
            zip(context["title"], context["sentences"], strict=True)
        )

        # `supporting_facts` = liste de (titre, indice de phrase).
        facts = row["supporting_facts"]
        contexts: list[str] = []
        for title, sent_id in zip(facts["title"], facts["sent_id"], strict=True):
            sentences = sentences_by_title.get(title)
            if sentences is None or sent_id >= len(sentences):
                continue  # référence brisée côté HotpotQA — rare, on l'ignore
            text = sentences[sent_id].strip()
            if text and text not in contexts:
                contexts.append(text)

        gold_titles = list(entry["gold_titles"])
        built.append(
            EvaluationSample(
                id=qid,
                question=entry["question"],
                type=entry["type"],
                level=entry["level"],
                ground_truth_answer=row["answer"],
                ground_truth_contexts=contexts,
                gold_titles=gold_titles,
                gold_sources=[
                    title_to_file[t] for t in gold_titles if t in title_to_file
                ],
            )
        )

    built.sort(key=lambda s: (s.type, s.id))
    return built


def load() -> list[EvaluationSample]:
    """Charge et VALIDE `dataset_v1.json`.

    Raises:
        ValidationError: Si une entrée viole le schéma.
    """
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    return [EvaluationSample.model_validate(item) for item in payload["samples"]]


def main() -> None:
    """Point d'entrée : construit (ou valide) et rapporte."""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if "--check" in sys.argv:
        print("\n[CHECK] Validation de dataset_v1.json...")
        samples = load()
        print(f"   {len(samples)} entrées valides")
    else:
        print("\n[STEP 1] Construction du jeu de données...")
        samples = build()
        payload = {
            "version": 1,
            "source_sample": SAMPLE.name,
            "hotpotqa": {
                "dataset": DATASET_NAME,
                "config": DATASET_CONFIG,
                "split": DATASET_SPLIT,
            },
            "ground_truth": {
                "answer": "champ `answer` de HotpotQA",
                "contexts": (
                    "phrases désignées par `supporting_facts` (titre + indice), "
                    "extraites du `context` de la question — les phrases "
                    "justificatives précises, pas les articles entiers"
                ),
            },
            "n": len(samples),
            "samples": [s.model_dump() for s in samples],
        }
        DATASET.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"   écrit : {DATASET.relative_to(BASE_DIR)}")

        print("\n[STEP 2] Revalidation depuis le disque...")
        samples = load()
        print(f"   {len(samples)} entrées relues et validées")

    by_type: dict[str, int] = {}
    total_contexts = 0
    for s in samples:
        by_type[s.type] = by_type.get(s.type, 0) + 1
        total_contexts += len(s.ground_truth_contexts)

    print("\n" + "=" * 72)
    print(f"  entrées                : {len(samples)}")
    print(f"  répartition            : {by_type}")
    print(
        f"  phrases justificatives : {total_contexts} "
        f"({total_contexts / max(len(samples), 1):.1f} par question)"
    )
    print(f"  sources gold vérifiées : toutes présentes dans {CORPUS_DIR.name}/")
    print("=" * 72 + "\n")

    for s in samples[:3]:
        print(f"  [{s.type}] {s.question[:66]}")
        print(f"      réponse  : {s.ground_truth_answer!r}")
        print(f"      contextes: {len(s.ground_truth_contexts)} phrases")
        print(f"      sources  : {s.gold_sources}")
        print()


if __name__ == "__main__":
    main()


__all__ = ["DATASET", "EvaluationSample", "build", "load"]
