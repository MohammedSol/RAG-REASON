"""
Construction du jeu annoté du Verifier — Sprint I5-C.

Assemble 50 exemples destinés à une annotation HUMAINE, seule mesure
indépendante de la qualité du Verifier : les métriques RAGAS du Sprint I5-B
ont pour juge le modèle qui a aussi généré les réponses.

COMPOSITION — 25 synthétiques + 25 réels
=========================================

**25 cas synthétiques.** Des réponses fondées, dans lesquelles une
affirmation absente des sources a été injectée délibérément. Le label est
acquis PAR CONSTRUCTION (`hallucinated`) : on sait ce qu'on a injecté. Ils
divisent par deux le temps d'annotation tout en fournissant un socle de
vérité terrain solide.

Trois familles d'injection, parce que les hallucinations réelles ne se
ressemblent pas :

    addition      un fait absent des sources est AJOUTÉ à une réponse
                  par ailleurs correcte (chiffre, date, détail inventé)
    contradiction un fait PRÉSENT dans les sources est remplacé par une
                  valeur fausse (date, nombre, lieu)
    substitution  une entité correcte est remplacée par une AUTRE entité
                  réelle mais fausse ici (personne, organisation, lieu)

**25 cas réels**, sans transformation, tels que produits par les campagnes
du Sprint I5-B : 20 du système complet et 5 du baseline, panachant réponses
substantielles et refus.

VERDICTS DU VERIFIER
--------------------
`run_full.json` porte déjà `is_grounded` pour ses 20 questions. En revanche :

* le BASELINE n'a pas de Verifier — `is_grounded` y vaut `null` partout ;
* les réponses TRANSFORMÉES n'ont évidemment jamais été vérifiées.

Le Verifier est donc exécuté ici sur ces 30 cas. C'est l'opération coûteuse
de ce script (quelques minutes par appel) : elle est INCRÉMENTALE et
reprenable, comme les campagnes du Sprint I5-B.

Usage :
    uv run python tests/evaluation/build_annotation_set.py
    uv run python tests/evaluation/build_annotation_set.py --dry-run
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "tests" / "evaluation" / "results"
OUTPUT = BASE_DIR / "tests" / "evaluation" / "verifier_annotation_set.json"
CACHE = BASE_DIR / "tests" / "evaluation" / "results" / "verifier_verdicts_cache.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
for noisy in ("LiteLLM", "httpx", "litellm", "opentelemetry"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
logger = logging.getLogger("jeu-annote")

_REFUSAL = re.compile(
    r"does not contain|cannot (answer|determine|provide)|no information|"
    r"not (mention|provide)",
    re.I,
)


class Injection(NamedTuple):
    """Une transformation contrôlée à appliquer à une réponse fondée.

    Attributes:
        system: `full` ou `naive` — campagne d'origine de la réponse.
        question_id: Identifiant HotpotQA de la question source.
        kind: `addition`, `contradiction` ou `substitution`.
        find: Fragment à remplacer. Vide pour une `addition` (on ajoute
            alors `replace` à la fin de la réponse).
        replace: Texte de remplacement, ou fragment à ajouter.
        rationale: Ce qui rend l'affirmation fausse ou absente des sources.
    """

    system: str
    question_id: str
    kind: str
    find: str
    replace: str
    rationale: str


# ─────────────────────────────────────────────────────────────────────────────
# Les 25 injections — rédigées à la main, réponse par réponse
# ─────────────────────────────────────────────────────────────────────────────
#
# Chaque injection est écrite à partir de la réponse RÉELLE et de ses sources.
# Une injection générée mécaniquement produirait un texte incongru qu'un
# annotateur repérerait à la forme plutôt qu'au fond, et le jeu ne mesurerait
# alors plus rien d'intéressant.
INJECTIONS: tuple[Injection, ...] = (
    # ── Handi-Snacks / Mondelez ──────────────────────────────────────────
    Injection(
        "full",
        "5a86ebac55429960ec39b6d6",
        "substitution",
        "Mondelez International.",
        "The Kraft Heinz Company.",
        "Les sources attribuent Handi-Snacks à Mondelez, pas à Kraft Heinz.",
    ),
    Injection(
        "naive",
        "5a86ebac55429960ec39b6d6",
        "addition",
        "",
        " The product line was launched in 1987 and generates $340 million "
        "in annual revenue.",
        "Ni la date de lancement ni le chiffre d'affaires ne figurent dans "
        "les sources.",
    ),
    # ── Carrefour / hypermarchés ─────────────────────────────────────────
    Injection(
        "full",
        "5ab84bf555429916710eb01f",
        "contradiction",
        "1,462 hypermarkets",
        "2,847 hypermarkets",
        "Les sources donnent 1 462 hypermarchés fin 2016.",
    ),
    Injection(
        "full",
        "5ab84bf555429916710eb01f",
        "addition",
        "",
        " The chain also operated 4,100 discount stores in Brazil that year.",
        "Aucune source ne mentionne de magasins discount au Brésil.",
    ),
    # ── Independent Spirit Awards ────────────────────────────────────────
    Injection(
        "full",
        "5ac3165c5542995ef918c10a",
        "substitution",
        "John Waters",
        "Quentin Tarantino",
        "Les sources désignent John Waters comme animateur de la 18e cérémonie.",
    ),
    Injection(
        "naive",
        "5ac3165c5542995ef918c10a",
        "addition",
        "",
        " The ceremony was broadcast live on HBO from the Santa Monica Pier.",
        "Ni le diffuseur ni le lieu ne figurent dans les sources.",
    ),
    Injection(
        "full",
        "5ac3165c5542995ef918c10a",
        "contradiction",
        "in 2002",
        "in 1998",
        "Les sources situent la 18e cérémonie en 2002.",
    ),
    # ── VCU ──────────────────────────────────────────────────────────────
    Injection(
        "full",
        "5adf37a95542995ec70e8f97",
        "contradiction",
        "founded in 1838",
        "founded in 1902",
        "Les sources donnent 1838 comme année de fondation.",
    ),
    Injection(
        "full",
        "5adf37a95542995ec70e8f97",
        "addition",
        "",
        " The university enrolled 41,300 students during the 2011–12 academic year.",
        "Aucun effectif étudiant ne figure dans les sources.",
    ),
    # ── L'Oiseau Blanc ───────────────────────────────────────────────────
    Injection(
        "full",
        "5ae0361155429925eb1afc2c",
        "substitution",
        "François Coli and Charles Nungesser",
        "Jean Mermoz and Antoine de Saint-Exupéry",
        "Les sources nomment Coli et Nungesser comme équipage de L'Oiseau Blanc.",
    ),
    Injection(
        "naive",
        "5ae0361155429925eb1afc2c",
        "addition",
        "",
        " Their aircraft was recovered off the coast of Newfoundland in 1961.",
        "Aucune source ne fait état d'une épave retrouvée.",
    ),
    # ── London Review of Books ───────────────────────────────────────────
    Injection(
        "full",
        "5ae2b770554299495565db0f",
        "contradiction",
        "March and April",
        "September and October",
        "Les sources situent le festival en mars et avril.",
    ),
    Injection(
        "naive",
        "5ae2b770554299495565db0f",
        "substitution",
        "London Review of Books",
        "Times Literary Supplement",
        "Les sources désignent la London Review of Books.",
    ),
    # ── Duke Energy / AMG ────────────────────────────────────────────────
    Injection(
        "full",
        "5abbd3ac55429931dba1458b",
        "contradiction",
        "Affiliated Managers Group is headquartered in Massachusetts",
        "Affiliated Managers Group is headquartered in Connecticut",
        "Les sources situent AMG dans le Massachusetts.",
    ),
    Injection(
        "naive",
        "5abbd3ac55429931dba1458b",
        "addition",
        "",
        " Affiliated Managers Group manages $831 billion in assets across "
        "40 affiliates.",
        "Aucun encours ni nombre d'affiliés ne figure dans les sources.",
    ),
    # ── Laleli / Esma Sultan ─────────────────────────────────────────────
    Injection(
        "full",
        "5adbf0a255429947ff17385a",
        "substitution",
        "Ortaköy neighborhood",
        "Beşiktaş neighborhood",
        "Les sources situent le manoir Esma Sultan à Ortaköy.",
    ),
    Injection(
        "full",
        "5adbf0a255429947ff17385a",
        "addition",
        "",
        " The Esma Sultan Mansion was destroyed by fire in 1975 and rebuilt in 2001.",
        "Aucun incendie ni reconstruction ne figure dans les sources.",
    ),
    Injection(
        "naive",
        "5adbf0a255429947ff17385a",
        "contradiction",
        "both in Istanbul, Turkey",
        "both in Ankara, Turkey",
        "Les sources situent les deux édifices à Istanbul.",
    ),
    # ── Agee / To Shoot an Elephant ──────────────────────────────────────
    Injection(
        "full",
        "5adc0c2b55429947ff1738db",
        "substitution",
        "the writer James Agee",
        "the photographer Walker Evans",
        "Les sources présentent Agee comme un documentaire sur l'écrivain James Agee.",
    ),
    Injection(
        "full",
        "5adc0c2b55429947ff1738db",
        "contradiction",
        "2008-2009 Gaza War",
        "1982 Lebanon War",
        "Les sources rattachent le film à la guerre de Gaza de 2008-2009.",
    ),
    Injection(
        "naive",
        "5adc0c2b55429947ff1738db",
        "addition",
        "",
        " To Shoot an Elephant won the Grand Jury Prize at Sundance in 2010.",
        "Aucune récompense ne figure dans les sources.",
    ),
    # ── Chrysalis / Look ─────────────────────────────────────────────────
    Injection(
        "full",
        "5ae058e855429945ae959331",
        "addition",
        "",
        " Chrysalis ceased publication in 1980 after eleven issues.",
        "Ni la date d'arrêt ni le nombre de numéros ne figurent dans les sources.",
    ),
    Injection(
        "full",
        "5ae058e855429945ae959331",
        "substitution",
        "a glossy high street fashion and celebrity weekly magazine",
        "a monthly literary review",
        "Les sources décrivent Look comme un hebdomadaire de mode et de célébrités.",
    ),
    # ── Am Rong / Ava DuVernay ───────────────────────────────────────────
    Injection(
        "naive",
        "5a76387d554299109176e6ba",
        "contradiction",
        "Ava DuVernay was born in 1972",
        "Ava DuVernay was born in 1968",
        "Les sources donnent 1972 comme année de naissance d'Ava DuVernay.",
    ),
    # ── S. Sylvan Simon / Danny Cannon ───────────────────────────────────
    Injection(
        "naive",
        "5ab611cc5542992aa134a411",
        "addition",
        "",
        " Both men also worked as cinematographers on more than twenty films.",
        "Aucune source ne leur attribue un travail de chef opérateur.",
    ),
)

# 5 questions du BASELINE retenues comme cas réels, en plus des 20 du système
# complet. Panachage volontaire : deux réponses substantielles et trois refus,
# pour que l'annotateur ne puisse pas déduire le label de la seule longueur.
NAIVE_REAL_IDS: tuple[str, ...] = (
    "5a76387d554299109176e6ba",  # substantielle — Am Rong / Ava DuVernay
    "5ab611cc5542992aa134a411",  # substantielle — Simon / Cannon
    "5aba7cfe554299232ef4a2fd",  # refus
    "5abf44025542993fe9a41def",  # refus
    "5ae25ed15542992decbdccd2",  # refus
)


def load_campaign(system: str) -> dict[str, dict[str, Any]]:
    """Résultats d'une campagne, indexés par identifiant de question."""
    payload = json.loads((RESULTS_DIR / f"run_{system}.json").read_text("utf-8"))
    return {r["id"]: r for r in payload["results"]}


def apply_injection(answer: str, injection: Injection) -> str:
    """Applique une injection à une réponse.

    Raises:
        SystemExit: Si le fragment à remplacer est absent de la réponse —
            l'injection serait alors silencieusement sans effet, et le cas
            porterait un label `hallucinated` sur un texte intact.
    """
    if injection.kind == "addition":
        return answer.rstrip() + injection.replace
    if injection.find not in answer:
        raise SystemExit(
            f"injection {injection.system}/{injection.question_id[:12]} "
            f"({injection.kind}) : fragment introuvable dans la réponse.\n"
            f"  cherché : {injection.find!r}\n"
            f"  réponse : {answer!r}"
        )
    return answer.replace(injection.find, injection.replace, 1)


def load_cache() -> dict[str, bool]:
    """Verdicts du Verifier déjà calculés, indexés par clé d'entrée."""
    if not CACHE.is_file():
        return {}
    try:
        cached: dict[str, bool] = json.loads(CACHE.read_text(encoding="utf-8"))
        return cached
    except (OSError, ValueError):
        return {}


def save_cache(cache: dict[str, bool]) -> None:
    """Enregistre les verdicts — appelé après CHAQUE appel du Verifier."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def verify(answer: str, contexts: list[str], sources: list[str]) -> bool | None:
    """Exécute le Verifier réel sur une réponse et ses sources.

    Returns:
        Le `is_grounded` produit, ou `None` si l'appel échoue — l'entrée sera
        alors écartée du calcul de précision/rappel plutôt que comptée à tort.
    """
    from reasoning.contracts.action_interface import RetrievedChunk
    from reasoning.verifier import Verifier

    chunks = [
        RetrievedChunk(
            chunk_id=str(index),
            content=content,
            source=sources[index] if index < len(sources) else "inconnu",
            relevance_score=0.5,
        )
        for index, content in enumerate(contexts)
    ]
    try:
        return bool(Verifier().verify(answer, chunks).is_grounded)
    except Exception as exc:  # noqa: BLE001 — une entrée perdue, pas le jeu
        logger.error("Verifier en échec (%s: %s).", type(exc).__name__, exc)
        return None


def build() -> list[dict[str, Any]]:
    """Assemble les 50 entrées, en exécutant le Verifier là où il le faut."""
    full = load_campaign("full")
    naive = load_campaign("naive")
    campaigns = {"full": full, "naive": naive}
    cache = load_cache()

    entries: list[dict[str, Any]] = []

    # ── 25 cas réels ─────────────────────────────────────────────────────
    for record in full.values():
        entries.append(
            {
                "id": f"real-full-{record['id'][:12]}",
                "source_system": "full",
                "source_question_id": record["id"],
                "question": record["question"],
                "answer": record["answer"],
                "sources": list(record["contexts"]),
                "source_files": list(record["sources"]),
                "verifier_verdict": record.get("is_grounded"),
                "origin": "real",
                "transformation": None,
                "expected_label": None,
                "human_label": "",
            }
        )

    for qid in NAIVE_REAL_IDS:
        record = naive[qid]
        entries.append(
            {
                "id": f"real-naive-{qid[:12]}",
                "source_system": "naive",
                "source_question_id": qid,
                "question": record["question"],
                "answer": record["answer"],
                "sources": list(record["contexts"]),
                "source_files": list(record["sources"]),
                "verifier_verdict": None,  # calculé plus bas
                "origin": "real",
                "transformation": None,
                "expected_label": None,
                "human_label": "",
            }
        )

    # ── 25 cas synthétiques ──────────────────────────────────────────────
    for index, injection in enumerate(INJECTIONS, start=1):
        record = campaigns[injection.system][injection.question_id]
        transformed = apply_injection(record["answer"], injection)
        if transformed == record["answer"]:
            raise SystemExit(
                f"injection {index} sans effet sur la réponse — vérifier "
                f"{injection.question_id[:12]}"
            )
        entries.append(
            {
                "id": f"synth-{index:02d}-{injection.kind[:4]}",
                "source_system": injection.system,
                "source_question_id": injection.question_id,
                "question": record["question"],
                "answer": transformed,
                "sources": list(record["contexts"]),
                "source_files": list(record["sources"]),
                "verifier_verdict": None,  # calculé plus bas
                "origin": "synthetic",
                "transformation": {
                    "kind": injection.kind,
                    "original_answer": record["answer"],
                    "find": injection.find or None,
                    "replace": injection.replace,
                    "rationale": injection.rationale,
                },
                # Label acquis par construction : on a injecté une affirmation
                # absente des sources. Conservé DISTINCT de `human_label`,
                # pour que la concordance entre les deux reste mesurable.
                "expected_label": "hallucinated",
                "human_label": "",
            }
        )

    # ── Verdicts manquants : exécution du Verifier ───────────────────────
    todo = [e for e in entries if e["verifier_verdict"] is None]
    logger.info(
        "%d entrées, %d verdicts à calculer (%d déjà en cache).",
        len(entries),
        len(todo),
        sum(1 for e in todo if e["id"] in cache),
    )

    started = time.perf_counter()
    for position, entry in enumerate(todo, start=1):
        if entry["id"] in cache:
            entry["verifier_verdict"] = cache[entry["id"]]
            continue
        call_started = time.perf_counter()
        verdict = verify(entry["answer"], entry["sources"], entry["source_files"])
        entry["verifier_verdict"] = verdict
        if verdict is not None:
            cache[entry["id"]] = verdict
            save_cache(cache)  # ← après CHAQUE appel
        logger.info(
            "[%d/%d] %s : is_grounded=%s | %.0f s | écoulé %.0f min",
            position,
            len(todo),
            entry["id"],
            verdict,
            time.perf_counter() - call_started,
            (time.perf_counter() - started) / 60,
        )

    return entries


def main() -> None:
    """Point d'entrée : construit, valide et écrit le jeu annoté."""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        logger.info("Mode --dry-run : injections vérifiées, Verifier NON appelé.")
        full, naive = load_campaign("full"), load_campaign("naive")
        campaigns = {"full": full, "naive": naive}
        for index, injection in enumerate(INJECTIONS, start=1):
            record = campaigns[injection.system][injection.question_id]
            transformed = apply_injection(record["answer"], injection)
            print(f"\n  [{index:02d}] {injection.kind} — {injection.question_id[:12]}")
            print(f"       avant : {record['answer'][:100]}")
            print(f"       après : {transformed[:100]}")
        print(f"\n  {len(INJECTIONS)} injections applicables.\n")
        return

    entries = build()
    OUTPUT.write_text(
        json.dumps(
            {
                "version": 1,
                "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "n": len(entries),
                "composition": {
                    "real": sum(1 for e in entries if e["origin"] == "real"),
                    "synthetic": sum(1 for e in entries if e["origin"] == "synthetic"),
                },
                "injection_kinds": {
                    kind: sum(1 for i in INJECTIONS if i.kind == kind)
                    for kind in ("addition", "contradiction", "substitution")
                },
                "annotation_protocol": (
                    "`human_label` vaut 'grounded' ou 'hallucinated'. Il est "
                    "rempli par un annotateur humain via l'onglet Annotation du "
                    "dashboard, SANS voir `verifier_verdict` ni "
                    "`expected_label` — les afficher créerait un biais "
                    "d'ancrage qui invaliderait la mesure."
                ),
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    real = [e for e in entries if e["origin"] == "real"]
    synth = [e for e in entries if e["origin"] == "synthetic"]
    unresolved = [e for e in entries if e["verifier_verdict"] is None]
    refusals = sum(1 for e in real if _REFUSAL.search(e["answer"]))

    print("\n" + "=" * 72)
    print(f"  entrées             : {len(entries)}")
    print(
        f"  réels               : {len(real)} "
        f"({refusals} refus, {len(real) - refusals} substantielles)"
    )
    print(f"  synthétiques        : {len(synth)}")
    for kind in ("addition", "contradiction", "substitution"):
        print(f"      {kind:14} : {sum(1 for i in INJECTIONS if i.kind == kind)}")
    print(f"  verdicts manquants  : {len(unresolved)}")
    print(f"  human_label remplis : {sum(1 for e in entries if e['human_label'])}")
    print(f"  écrit               : {OUTPUT.relative_to(BASE_DIR)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
