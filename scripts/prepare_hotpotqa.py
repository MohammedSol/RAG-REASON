"""
Script de preparation du dataset HotpotQA -- Sprint 3 (Ingestion & Preprocessing)
Projet : RAG-REASON

Ce script telecharge le split validation de hotpot_qa (fullwiki),
extrait uniquement les colonnes legeres, et sauvegarde un echantillon
equilibre de 200 requetes (100 bridge + 100 comparison) en JSON.

Usage :
    uv run python scripts/prepare_hotpotqa.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from datasets import load_dataset

# Force UTF-8 pour la console Windows (cp1252 ne supporte pas les emojis)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — chemins & constantes
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent  # racine du projet
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_EVALUATION = BASE_DIR / "data" / "evaluation"

OUTPUT_FILE = DATA_PROCESSED / "hotpotqa_sprint3.json"

# Colonnes a garder — on drop volontairement "context" et "supporting_facts"
COLUMNS_TO_KEEP = ["id", "question", "type", "level"]

# Echantillonnage equilibre
N_PER_TYPE = 100  # 100 bridge + 100 comparison = 200 total


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Creation de l'arborescence data/
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 1] Creation de l'arborescence du projet...")

for folder in [DATA_RAW, DATA_PROCESSED, DATA_EVALUATION]:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"   [OK] {folder.relative_to(BASE_DIR)}")

print("   --> Arborescence prete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Telechargement HotpotQA (HuggingFace gere le cache automatiquement)
# ─────────────────────────────────────────────────────────────────────────────

print("[STEP 2] Telechargement du dataset HotpotQA (fullwiki / validation)...")
print(
    "   [INFO] Si deja en cache, HuggingFace l'utilise directement — pas de re-download.\n"
)

# HuggingFace met les fichiers en cache dans ~/.cache/huggingface/datasets/
dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")

print(f"   [OK] Dataset charge : {len(dataset)} exemples au total.")
print(f"   --> Colonnes disponibles : {dataset.column_names}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Schema Reduction : on drop les colonnes lourdes
# ─────────────────────────────────────────────────────────────────────────────

print(
    "[STEP 3] Schema reduction — suppression des colonnes lourdes (context, supporting_facts)..."
)

# On conserve uniquement ce dont on a besoin pour l'evaluation du Sprint 3
heavy_columns = [col for col in dataset.column_names if col not in COLUMNS_TO_KEEP]
dataset_light = dataset.remove_columns(heavy_columns)

print(f"   [OK] Colonnes supprimees : {heavy_columns}")
print(f"   --> Colonnes conservees : {dataset_light.column_names}")
print(f"   --> Taille du dataset allege : {len(dataset_light)} lignes\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Echantillonnage equilibre : 100 bridge + 100 comparison
# ─────────────────────────────────────────────────────────────────────────────

print(
    f"[STEP 4] Echantillonnage equilibre ({N_PER_TYPE} bridge + {N_PER_TYPE} comparison)..."
)

# Filtrage par type — HuggingFace Dataset.filter() est efficace et non destructif
bridge_samples = dataset_light.filter(lambda row: row["type"] == "bridge")
comparison_samples = dataset_light.filter(lambda row: row["type"] == "comparison")

print(f"   --> bridge disponibles    : {len(bridge_samples)}")
print(f"   --> comparison disponibles: {len(comparison_samples)}")

# Verification qu'on a assez de donnees
if len(bridge_samples) < N_PER_TYPE:
    raise ValueError(
        f"Pas assez de requetes bridge : {len(bridge_samples)} < {N_PER_TYPE}"
    )
if len(comparison_samples) < N_PER_TYPE:
    raise ValueError(
        f"Pas assez de requetes comparison : {len(comparison_samples)} < {N_PER_TYPE}"
    )

# Selection des N premiers exemples de chaque type
# On utilise select() plutot qu'un slice pour rester dans l'API HuggingFace
bridge_final = bridge_samples.select(range(N_PER_TYPE))
comparison_final = comparison_samples.select(range(N_PER_TYPE))

print(f"   [OK] {len(bridge_final)} bridge selectionnes")
print(f"   [OK] {len(comparison_final)} comparison selectionnes")

# Fusion des deux sous-ensembles en une seule liste Python
# On convertit directement en list[dict] pour la serialisation JSON
records_bridge: list[dict[str, str]] = bridge_final.to_list()
records_comparison: list[dict[str, str]] = comparison_final.to_list()
all_records = records_bridge + records_comparison

print(f"   --> Total final : {len(all_records)} requetes\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Sauvegarde en JSON indente (lisible et versionnable)
# ─────────────────────────────────────────────────────────────────────────────

print(f"[STEP 5] Sauvegarde dans {OUTPUT_FILE.relative_to(BASE_DIR)} ...")

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(all_records, f, ensure_ascii=False, indent=2)

# Verification post-sauvegarde
saved_size_kb = OUTPUT_FILE.stat().st_size / 1024
print(f"   [OK] Fichier sauvegarde ({saved_size_kb:.1f} Ko)")
print(f"   --> Chemin absolu : {OUTPUT_FILE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Resume final
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("[DONE] Preprocessing termine avec succes !")
print("   Dataset  : hotpot_qa / fullwiki / validation")
print(
    f"   Exemples : {len(all_records)} ({N_PER_TYPE} bridge + {N_PER_TYPE} comparison)"
)
print("   Fichier  : data/processed/hotpotqa_sprint3.json")
print("=" * 60)
print()
