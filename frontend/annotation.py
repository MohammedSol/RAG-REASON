"""
Onglet d'annotation du Verifier — Sprint I5-C.

Interface d'annotation des 50 exemples de
`tests/evaluation/verifier_annotation_set.json`. Elle fournit une mesure
INDÉPENDANTE de la qualité du Verifier : les métriques RAGAS du Sprint I5-B
ont pour juge le modèle qui a aussi généré les réponses.

DEUX RÈGLES DE CONCEPTION NON NÉGOCIABLES
==========================================

**Le verdict du Verifier n'est jamais affiché pendant l'annotation.** Le
montrer produirait un biais d'ancrage : l'annotateur validerait la machine
au lieu de juger la réponse, et la mesure de précision/rappel ne vaudrait
plus rien. Même règle pour `expected_label` des cas synthétiques — sans
quoi ces 25 cas seraient annotés de confiance, et la concordance entre
label attendu et label humain, qui sert justement à détecter une injection
ratée, ne mesurerait plus rien.

**Sauvegarde après chaque clic.** Le fichier est réécrit à chaque décision.
Une fermeture accidentelle d'onglet ne coûte rien.

NOTE D'IMPLÉMENTATION — POURQUOI DES CALLBACKS `on_click`
----------------------------------------------------------
Une première version appelait `st.rerun()` dans le corps des boutons et
mettait en forme la réponse avec `unsafe_allow_html`. Cette combinaison
faisait planter l'interface :

    NotFoundError: Failed to execute 'removeChild' on 'Node'

`st.rerun()` interrompt le rendu en cours pour en relancer un autre, pendant
que React manipule encore l'arbre DOM ; les nœuds HTML bruts injectés à la
main ne sont alors plus là où React les attend.

Deux corrections, appliquées ensemble :

* les boutons passent par `on_click=` — le callback s'exécute AVANT le
  nouveau rendu, si bien que toute la page se redessine d'un coup avec le
  bon exemple, sans `st.rerun()` ;
* la mise en forme n'utilise plus que des composants natifs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATION_SET = BASE_DIR / "tests" / "evaluation" / "verifier_annotation_set.json"

GROUNDED = "grounded"
HALLUCINATED = "hallucinated"

# Clés d'état Streamlit, préfixées pour ne pas heurter les autres onglets.
_CURSOR = "annot_cursor"


def _load() -> dict[str, Any] | None:
    """Charge le jeu d'annotation, ou None s'il n'a pas encore été construit."""
    if not ANNOTATION_SET.is_file():
        return None
    payload: dict[str, Any] = json.loads(ANNOTATION_SET.read_text(encoding="utf-8"))
    return payload


def _save(payload: dict[str, Any]) -> None:
    """Réécrit le jeu sur disque. Appelé après CHAQUE décision."""
    ANNOTATION_SET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _first_unlabelled(entries: list[dict[str, Any]]) -> int:
    """Indice du premier exemple non annoté, ou 0 si tout l'est."""
    for index, entry in enumerate(entries):
        if not entry.get("human_label"):
            return index
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — exécutés AVANT le nouveau rendu, jamais `st.rerun()`
# ─────────────────────────────────────────────────────────────────────────────


def _decide(payload: dict[str, Any], index: int, label: str) -> None:
    """Enregistre une décision et avance d'un exemple."""
    entries = payload["entries"]
    entries[index]["human_label"] = label
    _save(payload)
    st.session_state[_CURSOR] = min(index + 1, len(entries) - 1)


def _go(index: int) -> None:
    """Déplace le curseur sur un exemple donné."""
    st.session_state[_CURSOR] = index


def render_tab_annotation() -> None:
    """Rend l'onglet d'annotation."""
    st.header("🏷️ Annotation du Verifier")

    payload = _load()
    if payload is None:
        st.warning(
            "Jeu d'annotation absent. Le construire par :\n\n"
            "```bash\n"
            "uv run python tests/evaluation/build_annotation_set.py\n"
            "```"
        )
        return

    entries: list[dict[str, Any]] = payload["entries"]
    total = len(entries)
    done = sum(1 for e in entries if e.get("human_label"))

    st.caption(
        "Tout ce que dit la réponse est-il écrit dans les sources ? "
        "Le verdict du Verifier est volontairement masqué : l'afficher "
        "biaiserait le jugement et invaliderait la mesure."
    )
    st.progress(done / total, text=f"{done} / {total} annotés")

    if _CURSOR not in st.session_state:
        st.session_state[_CURSOR] = _first_unlabelled(entries)
    index = int(st.session_state[_CURSOR])
    entry = entries[index]

    # ── Navigation ───────────────────────────────────────────────────────
    nav_prev, nav_pos, nav_next, nav_skip = st.columns([1, 2, 1, 2])
    with nav_prev:
        st.button(
            "← Précédent",
            use_container_width=True,
            disabled=index == 0,
            on_click=_go,
            args=(index - 1,),
            key="annot_prev",
        )
    with nav_pos:
        st.markdown(f"**Exemple {index + 1} / {total}**")
    with nav_next:
        st.button(
            "Suivant →",
            use_container_width=True,
            disabled=index >= total - 1,
            on_click=_go,
            args=(index + 1,),
            key="annot_next",
        )
    with nav_skip:
        st.button(
            "⏭️ Premier non annoté",
            use_container_width=True,
            on_click=_go,
            args=(_first_unlabelled(entries),),
            key="annot_skip",
        )

    if entry.get("human_label"):
        libelle = "Fondée" if entry["human_label"] == GROUNDED else "Hallucinée"
        st.info(f"Déjà annoté : **{libelle}**. Cliquer à nouveau remplace la réponse.")

    st.divider()
    st.subheader(f"❓ {entry['question']}")

    # ── Réponse et sources, côte à côte ──────────────────────────────────
    col_answer, col_sources = st.columns([1, 1])
    with col_answer:
        st.markdown("#### Réponse à juger")
        with st.container(border=True):
            st.write(entry["answer"])
    with col_sources:
        st.markdown(f"#### Sources ({len(entry['sources'])})")
        files = entry.get("source_files") or []
        for position, text in enumerate(entry["sources"]):
            name = (
                files[position] if position < len(files) else f"passage {position + 1}"
            )
            with st.expander(f"📄 {name}", expanded=position < 2):
                st.write(text)

    st.divider()

    # ── Décision ─────────────────────────────────────────────────────────
    st.markdown("#### Votre jugement")
    choice_grounded, choice_halluc = st.columns(2)
    with choice_grounded:
        st.button(
            "✅ Fondée",
            use_container_width=True,
            type="primary",
            help="Toutes les affirmations sont soutenues par les sources.",
            on_click=_decide,
            args=(payload, index, GROUNDED),
            key="annot_grounded",
        )
    with choice_halluc:
        st.button(
            "❌ Hallucinée",
            use_container_width=True,
            help="Au moins une affirmation n'est pas soutenue par les sources.",
            on_click=_decide,
            args=(payload, index, HALLUCINATED),
            key="annot_hallucinated",
        )

    st.caption(
        "Une réponse qui refuse de répondre (« the context does not "
        "contain… ») est **fondée** si ce refus est exact : elle n'avance "
        "aucune affirmation que les sources ne soutiennent pas."
    )

    if done == total:
        st.success(
            f"Les {total} exemples sont annotés. Calculer précision et rappel :\n\n"
            "```bash\n"
            "uv run python tests/evaluation/verifier_metrics.py\n"
            "```"
        )


__all__ = ["render_tab_annotation"]
