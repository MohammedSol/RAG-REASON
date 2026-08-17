"""
Construction du corpus HotpotQA aligne sur le jeu d'evaluation — Sprint I1.
Projet : RAG-REASON (integration avec le module ACTION)

Extrait les paragraphes de contexte des 200 questions du jeu d'evaluation et
ecrit un fichier .txt par article dans data/corpus/, plus un manifeste.

CHOIX DU SPLIT — decision actee au Sprint I1
--------------------------------------------
Ce script charge le split `distractor`, alors que `prepare_hotpotqa.py`
(Sprint 3) chargeait `fullwiki`. Mesure a l'origine de ce choix :

    couverture des paragraphes gold (supporting_facts presents dans context)
        fullwiki   :  45/200 = 22,5 %   -> 155 questions sans reponse possible
        distractor : 200/200 = 100,0 %

En `distractor`, le contexte est construit comme 2 paragraphes gold + 8
distracteurs : les gold sont garantis. En `fullwiki`, le contexte est le
resultat d'une recuperation sur tout Wikipedia et ne contient souvent pas
les gold — c'est ce qui fait la difficulte du reglage open-domain, mais cela
rend le corpus inexploitable pour mesurer `context_recall`.

Les 200 `id` du jeu d'evaluation sont presents dans les DEUX splits : le
changement ne modifie ni les questions, ni leur ordre, ni les resultats des
evaluations anterieures. `data/processed/hotpotqa_sprint3.json` reste
INCHANGE ; ce script s'y refere uniquement pour filtrer les `id`.

Les distracteurs sont volontairement CONSERVES : un corpus reduit aux seuls
paragraphes gold rendrait le retrieval trivial et gonflerait artificiellement
les metriques.

CONVENTION DE NOMMAGE
---------------------
Le nom de fichier derive du titre de l'article. `DocumentManager` du module
ACTION fixe `chunk["source"] = file_path.name` : le champ `source` des chunks
correspondra donc directement aux titres des `supporting_facts`, ce qui rend
`context_precision` et `context_recall` mesurables sans alignement manuel.

Usage :
    uv run python scripts/04_build_corpus.py
"""

from __future__ import annotations

import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_SET = BASE_DIR / "data" / "processed" / "hotpotqa_sprint3.json"
CORPUS_DIR = BASE_DIR / "data" / "corpus"
MANIFEST = CORPUS_DIR / "corpus_manifest.json"

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"

# Caracteres interdits dans un nom de fichier Windows.
_WINDOWS_INVALID = r'<>:"/\|?*'
_INVALID_RE = re.compile(f"[{re.escape(_WINDOWS_INVALID)}]")
_CONTROL_RE = re.compile(r"[\x00-\x1f]")


def normalize_filename(title: str) -> str:
    """Derive un nom de fichier .txt valide sous Windows depuis un titre.

    Espaces -> underscores, caracteres interdits retires, points et espaces
    de fin supprimes (Windows les rejette silencieusement).

    Args:
        title: Titre de l'article Wikipedia.

    Returns:
        Nom de fichier, extension .txt incluse.
    """
    name = _CONTROL_RE.sub("", title)
    name = _INVALID_RE.sub("", name)
    name = name.replace(" ", "_")
    name = name.strip(". ")
    if not name:
        name = "article"
    return f"{name}.txt"


def build_article_text(title: str, sentences: list[str]) -> str:
    """Compose le contenu d'un fichier article : titre puis phrases."""
    body = " ".join(s.strip() for s in sentences if s.strip())
    return f"{title}\n\n{body}\n"


def main() -> None:
    """Point d'entree : extraction, deduplication, ecriture, manifeste."""
    print("\n[STEP 1] Chargement du jeu d'evaluation et du dataset source...")
    eval_records: list[dict[str, str]] = json.loads(
        EVAL_SET.read_text(encoding="utf-8")
    )
    eval_ids = [r["id"] for r in eval_records]
    print(f"   [OK] {len(eval_ids)} questions dans {EVAL_SET.name}")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    position = {qid: i for i, qid in enumerate(dataset["id"])}
    missing_ids = [q for q in eval_ids if q not in position]
    if missing_ids:
        raise SystemExit(
            f"{len(missing_ids)} id du jeu d'evaluation absents du split "
            f"{DATASET_CONFIG} — extraction impossible."
        )
    print(
        f"   [OK] {DATASET_CONFIG}/{DATASET_SPLIT} : {len(dataset)} entrees, "
        f"{len(eval_ids)}/{len(eval_ids)} id retrouves"
    )

    # ── Extraction et deduplication par titre ────────────────────────────────
    print("\n[STEP 2] Extraction des paragraphes de contexte...")
    articles: dict[str, list[str]] = {}  # titre -> phrases
    questions_meta: list[dict[str, Any]] = []

    for qid in eval_ids:
        row = dataset[position[qid]]
        ctx = row["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"], strict=True):
            if title not in articles:
                articles[title] = list(sentences)

        gold_titles = sorted(set(row["supporting_facts"]["title"]))
        present = [t for t in gold_titles if t in articles or t in ctx["title"]]
        questions_meta.append(
            {
                "id": qid,
                "type": row["type"],
                "gold_titles": gold_titles,
                "gold_present": sorted(set(present)),
                "gold_missing": sorted(set(gold_titles) - set(present)),
                "is_complete": len(set(gold_titles) - set(present)) == 0,
            }
        )
    print(f"   [OK] {len(articles)} articles distincts apres deduplication")

    # ── Nommage et detection des collisions ──────────────────────────────────
    print("\n[STEP 3] Normalisation des noms de fichiers...")
    # Windows est insensible a la casse : on regroupe sur la forme repliee.
    by_name: dict[str, list[str]] = defaultdict(list)
    for title in articles:
        by_name[normalize_filename(title).lower()].append(title)

    title_to_file: dict[str, str] = {}
    collisions: list[dict[str, Any]] = []
    for folded, titles in by_name.items():
        if len(titles) == 1:
            title_to_file[titles[0]] = normalize_filename(titles[0])
            continue
        # Collision : suffixe deterministe, aucun ecrasement silencieux.
        ordered = sorted(titles)
        assigned = []
        for rank, title in enumerate(ordered, start=1):
            base = normalize_filename(title)
            name = base if rank == 1 else f"{base[:-4]}__{rank}.txt"
            title_to_file[title] = name
            assigned.append({"title": title, "file": name})
        collisions.append(
            {"normalized": folded, "titles": ordered, "resolution": assigned}
        )

    if collisions:
        print(
            f"   [!] {len(collisions)} collision(s) de nom detectee(s) et resolue(s) :"
        )
        for c in collisions:
            print(f"       {c['normalized']} <- {c['titles']}")
    else:
        print("   [OK] aucune collision de nom")

    # ── Ecriture des fichiers ────────────────────────────────────────────────
    print(f"\n[STEP 4] Ecriture dans {CORPUS_DIR.relative_to(BASE_DIR)} ...")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for old in CORPUS_DIR.glob("*.txt"):
        old.unlink()

    written = 0
    for title, sentences in articles.items():
        path = CORPUS_DIR / title_to_file[title]
        path.write_text(build_article_text(title, sentences), encoding="utf-8")
        written += 1
    print(f"   [OK] {written} fichiers .txt ecrits (UTF-8)")

    # ── Manifeste ────────────────────────────────────────────────────────────
    incomplete = [q for q in questions_meta if not q["is_complete"]]
    manifest = {
        "source": {
            "dataset": DATASET_NAME,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "selection": (
                "filtre sur les id de data/processed/hotpotqa_sprint3.json "
                "(inchange) — 100 bridge + 100 comparison"
            ),
            "why_distractor": (
                "Le split fullwiki, utilise au Sprint 3, ne contient les "
                "paragraphes gold que pour 45/200 questions (22,5 %) : son "
                "context est le resultat d'une recuperation sur tout Wikipedia. "
                "Le split distractor garantit les 2 paragraphes gold + 8 "
                "distracteurs, soit 200/200 (100 %). Les 200 id sont presents "
                "dans les deux splits : ni les questions, ni leur ordre, ni les "
                "resultats des evaluations anterieures ne changent. Les "
                "distracteurs sont conserves pour ne pas rendre le retrieval "
                "trivial."
            ),
            "coverage_measured": {"fullwiki": "45/200", "distractor": "200/200"},
        },
        "naming_convention": (
            "titre -> espaces en underscores, caracteres Windows invalides "
            f"({_WINDOWS_INVALID}) retires, extension .txt. Objectif : "
            "chunk['source'] du module ACTION == nom de fichier == titre "
            "d'article, pour aligner supporting_facts et resultats de "
            "retrieval sans table de correspondance externe."
        ),
        "counts": {
            "articles": len(articles),
            "questions": len(questions_meta),
            "questions_complete": len(questions_meta) - len(incomplete),
            "questions_incomplete": len(incomplete),
            "coverage_pct": round(
                (len(questions_meta) - len(incomplete)) / len(questions_meta) * 100, 2
            ),
            "collisions": len(collisions),
        },
        "collisions": collisions,
        "incomplete_questions": incomplete,
        "title_to_file": dict(sorted(title_to_file.items())),
        "questions": questions_meta,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"   [OK] manifeste : {MANIFEST.relative_to(BASE_DIR)}")

    # ── Resume ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  articles ecrits          : {len(articles)}")
    print(f"  questions couvertes      : {len(questions_meta)}")
    print(
        f"  gold complets            : {manifest['counts']['questions_complete']} "
        f"({manifest['counts']['coverage_pct']} %)"
    )
    print(f"  gold incomplets          : {len(incomplete)}")
    print(f"  collisions de nom        : {len(collisions)}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
