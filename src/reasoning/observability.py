"""
Traçabilité des appels LLM — module REASONING (Sprint I5-A).

Le cahier des charges exige la traçabilité complète des sorties LLM
intermédiaires. Ce module l'apporte SANS toucher aux composants gelés
(`analyzer/`, `planner/`, `critic/`, `verifier/`) : il instrumente le point
de passage commun à tous, `litellm.completion`.

Trois responsabilités, toutes optionnelles et sans effet si l'environnement
n'est pas configuré :

1. **Activer le callback Langfuse de LiteLLM** quand les clés sont présentes.
2. **Regrouper les appels d'une même requête** sous un identifiant commun,
   propagé par `contextvars` — les appels Analyzer / Planner / Critic /
   Verifier / synthèse d'une même question se retrouvent ensemble.
3. **Compter les appels LLM** réellement émis, pour le plafond configurable
   appliqué par le nœud `critique` (`nodes.py`).

AUCUNE DÉPENDANCE DURE. Si `langfuse` n'est pas installé, ou si les clés
manquent, `configure_langfuse()` retourne `False` et le pipeline fonctionne
exactement comme avant. Le comptage, lui, reste actif : il ne dépend pas de
Langfuse.

CHOIX DU CALLBACK — mesuré, pas supposé
---------------------------------------
LiteLLM expose deux intégrations Langfuse. Comparées sur 3 appels réels
chacune, avec les clés du projet :

    litellm.success_callback = ["langfuse"]   -> 0 trace remontée
        échoue à l'initialisation :
        AttributeError: module 'langfuse' has no attribute 'version'
        (l'intégration vise l'API du SDK v2, absente du SDK v4)

    litellm.callbacks = ["langfuse_otel"]     -> 3 traces, 6 observations

C'est donc `langfuse_otel` qui est utilisé, malgré la formulation initiale de
la mission. Le SDK Langfuse installé est en v4, bâti sur OpenTelemetry.

REGROUPEMENT : PAR SESSION, PAS PAR TRACE
------------------------------------------
Chaque appel LiteLLM crée sa propre *trace* Langfuse. Le regroupement se fait
par `session_id`, que ce module fixe à l'identifiant de requête. Dans
l'interface Langfuse, tous les appels d'une même question apparaissent donc
sous une même session — ce qui est le comportement recherché.

INSTRUMENTATION — pourquoi un remplacement de symbole
------------------------------------------------------
LiteLLM lit `trace_id` et `session_id` dans le `metadata` de CHAQUE appel
(`litellm/integrations/langfuse/langfuse.py`). Il n'existe aucun point
d'injection global. Or les cinq composants font
`from litellm import completion` : ils capturent la fonction à l'import, si
bien que remplacer `litellm.completion` après coup n'aurait aucun effet sur
eux.

`instrument()` remplace donc le symbole `completion` DANS CHAQUE MODULE
appelant. Aucun fichier gelé n'est modifié : c'est un remplacement à
l'exécution, réversible par `uninstrument()`, et exactement le mécanisme
qu'utilise déjà `unittest.mock.patch` dans la suite de tests.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Modules qui appellent `completion` et doivent être instrumentés. Liste
# EXPLICITE : un composant ajouté au projet sans être inscrit ici ne serait
# ni tracé ni compté. `instrument()` retourne les modules effectivement
# instrumentés, pour que l'appelant puisse le vérifier.
_INSTRUMENTED_MODULES: tuple[str, ...] = (
    "reasoning.analyzer.analyzer",
    "reasoning.planner.planner",
    "reasoning.critic.critic",
    "reasoning.verifier.verifier",
    "reasoning.graph.nodes",
)

# Identifiant de la requête en cours — propagé à tous les appels LLM qu'elle
# déclenche. `ContextVar` et non variable globale : correct sous asyncio, où
# plusieurs requêtes peuvent être traitées en parallèle.
_TRACE_ID: ContextVar[str | None] = ContextVar("reasoning_trace_id", default=None)


class _CallCounter:
    """Compteur d'appels LLM, partagé par référence entre contextes.

    POURQUOI UN OBJET MUTABLE ET NON UN `ContextVar[int]`. Chaque
    `asyncio.Task` reçoit une COPIE du contexte : un `ContextVar.set()`
    exécuté dans une tâche fille reste invisible du parent. Comme LangGraph
    exécute chaque nœud dans sa propre tâche, un compteur entier stocké
    directement dans un `ContextVar` remontait toujours 0 — constaté au
    Sprint I5-A, où la mesure affichait 0 appel alors que Langfuse en avait
    tracé 8.

    La copie de contexte duplique la LIAISON, pas l'objet lié. En stockant
    une instance mutable, toutes les tâches incrémentent le même compteur.
    """

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: int = 0


# Défaut `None`, et non une instance : un compteur mutable partagé par défaut
# serait unique à tout le processus et accumulerait sans fin les appels émis
# hors requête. `None` traduit exactement l'intention — on ne compte que dans
# un bloc `trace_query`, c'est-à-dire dans le cadre d'UNE requête.
_LLM_CALLS: ContextVar[_CallCounter | None] = ContextVar(
    "reasoning_llm_calls", default=None
)

# Nom du symbole remplacé. Constante plutôt que littéral aux points d'appel :
# `setattr(module, "completion", ...)` sur un littéral déclenche la règle ruff
# B010, dont la correction automatique produit `module.completion = ...` —
# expression que mypy rejette sur un `ModuleType`. Passer par une constante
# satisfait les deux outils sans `noqa`.
_COMPLETION_ATTR: str = "completion"

_originals: dict[str, Callable[..., Any]] = {}
_langfuse_active: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Activation de Langfuse
# ─────────────────────────────────────────────────────────────────────────────


def configure_langfuse() -> bool:
    """Active la traçabilité Langfuse si l'environnement le permet.

    Idempotent : plusieurs appels ne créent qu'une activation.

    Returns:
        True si le callback est activé, False sinon — clés absentes, SDK non
        installé, ou erreur d'initialisation. Dans tous les cas de `False`, le
        pipeline reste pleinement fonctionnel.
    """
    global _langfuse_active
    if _langfuse_active:
        return True

    public = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public or not secret:
        logger.info(
            "Langfuse désactivé : LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            "absentes. Le pipeline fonctionne normalement, sans traçabilité."
        )
        return False

    # Le projet nomme cette variable LANGFUSE_BASE_URL ; LiteLLM et le SDK
    # Langfuse lisent LANGFUSE_HOST. On fait le pont sans écraser une valeur
    # déjà posée explicitement.
    base_url = os.getenv("LANGFUSE_BASE_URL", "").strip()
    if base_url and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = base_url

    try:
        import litellm
    except ImportError:  # pragma: no cover — litellm est une dépendance dure
        logger.warning("Langfuse désactivé : litellm introuvable.")
        return False

    try:
        import langfuse  # noqa: F401
    except ImportError:
        logger.warning(
            "Langfuse désactivé : le paquet `langfuse` n'est pas installé. "
            "Installer avec `uv sync --extra observability`."
        )
        return False

    existing = list(getattr(litellm, "callbacks", []) or [])
    if "langfuse_otel" not in existing:
        litellm.callbacks = [*existing, "langfuse_otel"]

    _langfuse_active = True
    logger.info(
        "Langfuse activé (callback langfuse_otel) sur %s.",
        os.getenv("LANGFUSE_HOST", "hôte par défaut"),
    )
    return True


def is_langfuse_active() -> bool:
    """Indique si le callback Langfuse a été activé."""
    return _langfuse_active


# ─────────────────────────────────────────────────────────────────────────────
# Instrumentation de `completion`
# ─────────────────────────────────────────────────────────────────────────────


def _wrap(original: Callable[..., Any]) -> Callable[..., Any]:
    """Enveloppe `completion` : compte l'appel et y attache la trace courante."""

    def _instrumented(*args: Any, **kwargs: Any) -> Any:
        counter = _LLM_CALLS.get()
        if counter is not None:
            counter.value += 1

        trace_id = _TRACE_ID.get()
        if trace_id is not None:
            metadata = dict(kwargs.get("metadata") or {})
            # `session_id` est ce qui REGROUPE les appels dans Langfuse ;
            # `trace_id` reste utile aux intégrations qui l'exploitent.
            metadata.setdefault("session_id", trace_id)
            metadata.setdefault("trace_id", trace_id)
            metadata.setdefault("tags", ["rag-reason"])
            kwargs["metadata"] = metadata

        return original(*args, **kwargs)

    _instrumented.__name__ = getattr(original, "__name__", "completion")
    _instrumented.__doc__ = getattr(original, "__doc__", None)
    return _instrumented


def instrument() -> list[str]:
    """Remplace `completion` dans chaque module appelant.

    Idempotent : un module déjà instrumenté n'est pas ré-enveloppé.

    Returns:
        Les noms des modules effectivement instrumentés.
    """
    import importlib

    done: list[str] = []
    for name in _INSTRUMENTED_MODULES:
        if name in _originals:
            done.append(name)
            continue
        try:
            module = importlib.import_module(name)
        except ImportError as exc:  # pragma: no cover — arborescence figée
            logger.warning("Instrumentation impossible pour %s : %s", name, exc)
            continue
        original = getattr(module, _COMPLETION_ATTR, None)
        if original is None:
            logger.warning("Module %s : aucun symbole `completion` à envelopper.", name)
            continue
        _originals[name] = original
        setattr(module, _COMPLETION_ATTR, _wrap(original))
        done.append(name)

    logger.info("Appels LLM instrumentés dans %d module(s) : %s", len(done), done)
    return done


def uninstrument() -> None:
    """Restaure les `completion` d'origine — utile aux tests."""
    import importlib

    for name, original in list(_originals.items()):
        try:
            setattr(importlib.import_module(name), _COMPLETION_ATTR, original)
        except ImportError:  # pragma: no cover
            pass
        del _originals[name]


# ─────────────────────────────────────────────────────────────────────────────
# Contexte de requête et comptage
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def trace_query(query_id: str) -> Iterator[None]:
    """Regroupe tous les appels LLM du bloc sous un même identifiant.

    Remet aussi le compteur d'appels à zéro à l'entrée : chaque requête part
    d'un budget neuf.

    Args:
        query_id: Identifiant de la requête — `plan_id` ou `query_id`.
    """
    token_trace = _TRACE_ID.set(query_id)
    token_calls = _LLM_CALLS.set(_CallCounter())
    try:
        yield
    finally:
        _TRACE_ID.reset(token_trace)
        _LLM_CALLS.reset(token_calls)


def current_trace_id() -> str | None:
    """Identifiant de trace de la requête en cours, ou None hors contexte."""
    return _TRACE_ID.get()


def llm_call_count() -> int:
    """Nombre d'appels LLM émis depuis l'entrée dans `trace_query`.

    Retourne 0 si l'instrumentation n'est pas active — le plafond ne se
    déclenche alors jamais, ce qui est le comportement voulu : ne pas
    instrumenter ne doit pas brider le pipeline.
    """
    counter = _LLM_CALLS.get()
    return counter.value if counter is not None else 0


def reset_llm_call_count() -> None:
    """Remet le compteur à zéro sans changer de contexte de trace."""
    counter = _LLM_CALLS.get()
    if counter is not None:
        counter.value = 0


def flush() -> None:
    """Force l'envoi des traces en attente. Sans effet si Langfuse est inactif.

    L'export Langfuse est asynchrone : sans ce vidage explicite, un script
    court peut se terminer avant que les traces ne soient parties.
    """
    if not _langfuse_active:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:  # noqa: BLE001 — l'observabilité ne casse rien
        logger.warning("Vidage Langfuse impossible (%s: %s).", type(exc).__name__, exc)


__all__ = [
    "configure_langfuse",
    "current_trace_id",
    "flush",
    "instrument",
    "is_langfuse_active",
    "llm_call_count",
    "reset_llm_call_count",
    "trace_query",
    "uninstrument",
]
