# mypy: ignore-errors
"""
RAG-REASON — Dashboard de démonstration du module REASONING.

Ce tableau de bord expose le moteur de raisonnement composant par composant,
puis le pipeline complet orchestré par LangGraph.

Lancement :
    uv run streamlit run frontend/app.py

Note d'architecture : le paquet `rag-reason` étant installé en mode editable,
le paquet `reasoning` est importable directement — aucune manipulation de
`sys.path` n'est nécessaire (contrairement à la version Sprint 3).

Périmètre : ce fichier consomme les composants de `src/reasoning/` en LECTURE
SEULE, via leur API publique. Il ne les modifie ni ne les contourne.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import graphviz
import streamlit as st

from reasoning.analyzer import QueryAnalyzer
from reasoning.contracts.action_interface import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from reasoning.contracts.internal_models import ExecutionPlan, PlanStep, QueryType
from reasoning.critic import Critic
from reasoning.graph.graph import build_graph
from reasoning.graph.policy import (
    ROUTE_CLARIFY,
    ROUTE_END,
    ROUTE_GENERATE_ANSWER,
    ROUTE_PLAN,
    ROUTE_RETRIEVE,
)
from reasoning.graph.state import build_initial_state
from reasoning.planner import Planner
from reasoning.verifier import Verifier

# ═════════════════════════════════════════════════════════════════════════════
# Constantes de présentation
# ═════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent

#: Avertissement réutilisé partout où une donnée de retrieval est simulée.
SIMULATION_NOTICE = (
    "⚠️ **Données de retrieval simulées.** Le module ACTION (recherche "
    "documentaire réelle, développé en binôme) n'est pas encore intégré. "
    "Les extraits ci-dessous sont fictifs et éditables : ils servent à "
    "démontrer le comportement du composant, pas la qualité d'un retrieval."
)

#: Requêtes d'exemple prêtes à l'emploi (démonstration sans réflexion préalable).
EXAMPLE_QUERIES: dict[str, str] = {
    "SIMPLE — fait unique": "When was OpenAI founded?",
    "MULTI_HOP — chaîne de raisonnement": (
        "Who is the CEO of the company that created GPT-4, and when was that "
        "company founded?"
    ),
    "COMPARATIVE — mise en parallèle": (
        "What is the difference between BM25 and a dense retriever?"
    ),
    "AMBIGUOUS — clarification requise": "Explique-moi les réseaux.",
}

#: Chunks simulés par défaut, cohérents avec les fixtures de test.
DEFAULT_CHUNKS: list[dict[str, Any]] = [
    {
        "chunk_id": "openai-history-001",
        "source": "openai_company_history.pdf",
        "content": (
            "OpenAI was founded in December 2015 by Elon Musk, Sam Altman, "
            "Greg Brockman, Ilya Sutskever, Wojciech Zaremba, and John "
            "Schulman. The company is headquartered in San Francisco, "
            "California. Sam Altman currently serves as its CEO."
        ),
        "relevance_score": 0.94,
    },
    {
        "chunk_id": "gpt4-report-002",
        "source": "gpt4_technical_report.pdf",
        "content": (
            "GPT-4 was released by OpenAI in March 2023. It is a large "
            "multimodal model capable of processing both text and image "
            "inputs, and it exhibits human-level performance on several "
            "professional and academic benchmarks."
        ),
        "relevance_score": 0.89,
    },
]

#: Réponse par défaut du Verifier — une affirmation FONDÉE + une HALLUCINATION.
DEFAULT_VERIFIER_ANSWER = (
    "OpenAI was founded in December 2015 and is headquartered in San "
    "Francisco. GPT-4 was trained on exactly 1.8 trillion parameters using "
    "quantum processors manufactured in Antarctica."
)

#: Nœuds du graphe et leur libellé d'affichage.
NODE_LABELS: dict[str, str] = {
    "analyze_query": "analyze_query\n(Query Analyzer)",
    "clarify": "clarify\n(demande de précision)",
    "plan": "plan\n(Planner)",
    "retrieve": "retrieve\n(→ module ACTION)",
    "critique": "critique\n(Critic)",
    "generate_answer": "generate_answer\n(synthèse LLM)",
    "verify": "verify\n(Verifier)",
    "END": "END",
}

#: Arêtes canoniques du graphe compilé (cf. docs/graph_spec.md).
GRAPH_EDGES: list[tuple[str, str, str]] = [
    ("analyze_query", "plan", "budget > 0"),
    ("analyze_query", "clarify", "budget = 0"),
    ("clarify", "END", ""),
    ("plan", "retrieve", ""),
    ("retrieve", "critique", ""),
    ("critique", "retrieve", "contexte insuffisant"),
    ("critique", "generate_answer", "contexte suffisant"),
    ("generate_answer", "verify", ""),
    ("verify", "END", "fondé"),
    ("verify", "plan", "non fondé"),
]

COMPONENT_TABLE: list[dict[str, str]] = [
    {
        "Composant": "Query Analyzer",
        "Rôle": "Classifie la requête et alloue le budget de raisonnement",
        "Sprint": "2",
        "Statut": "✅ Validé",
    },
    {
        "Composant": "Planner",
        "Rôle": "Décompose la requête en un DAG de sous-questions atomiques",
        "Sprint": "3",
        "Statut": "✅ Validé",
    },
    {
        "Composant": "Critic",
        "Rôle": "Juge si le contexte récupéré suffit à répondre à une étape",
        "Sprint": "4",
        "Statut": "✅ Validé",
    },
    {
        "Composant": "Verifier",
        "Rôle": "Vérifie que chaque affirmation est traçable dans les sources",
        "Sprint": "5",
        "Statut": "✅ Validé",
    },
    {
        "Composant": "Orchestration LangGraph",
        "Rôle": "Assemble les composants et pilote les boucles de rétroaction",
        "Sprint": "6",
        "Statut": "✅ Validé",
    },
    {
        "Composant": "Module ACTION (astraexec)",
        "Rôle": "Recherche documentaire réelle — développé en binôme",
        "Sprint": "—",
        "Statut": "⏳ Non intégré",
    },
    {
        "Composant": "Évaluation RAGAS",
        "Rôle": "Mesure objective de la qualité sur un corpus de référence",
        "Sprint": "7",
        "Statut": "⏳ En attente du module ACTION",
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# Helpers factorisés
# ═════════════════════════════════════════════════════════════════════════════


@st.cache_resource
def get_analyzer() -> QueryAnalyzer:
    """Instance QueryAnalyzer partagée entre les reruns Streamlit."""
    return QueryAnalyzer()


@st.cache_resource
def get_planner() -> Planner:
    """Instance Planner partagée entre les reruns Streamlit."""
    return Planner()


@st.cache_resource
def get_critic() -> Critic:
    """Instance Critic partagée entre les reruns Streamlit."""
    return Critic()


@st.cache_resource
def get_verifier() -> Verifier:
    """Instance Verifier partagée entre les reruns Streamlit."""
    return Verifier()


def run_async(coro: Any) -> Any:
    """Exécute une coroutine depuis le thread synchrone de Streamlit."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Appelle `func` en mesurant sa latence.

    Returns:
        Tuple `(résultat, latence_en_ms)`.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


def render_error(context: str, exc: Exception) -> None:
    """Affiche une erreur lisible plutôt qu'une traceback brute."""
    st.error(
        f"**{context}**\n\n"
        f"`{type(exc).__name__}` : {exc}\n\n"
        "Vérifiez qu'Ollama est démarré (`ollama serve`) et que le modèle "
        "configuré est disponible (`ollama pull qwen2.5:7b`)."
    )


def render_simulation_notice() -> None:
    """Affiche l'avertissement obligatoire sur les données simulées."""
    st.warning(SIMULATION_NOTICE)


def chunks_to_contract(raw_chunks: list[dict[str, Any]]) -> list[RetrievedChunk]:
    """Convertit les chunks éditables du formulaire en objets du contrat."""
    return [
        RetrievedChunk(
            chunk_id=chunk["chunk_id"],
            content=chunk["content"],
            source=chunk["source"],
            relevance_score=float(chunk.get("relevance_score", 0.75)),
        )
        for chunk in raw_chunks
        if chunk.get("content", "").strip()
    ]


def chunk_editor(prefix: str, defaults: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Formulaire d'édition de chunks simulés, réutilisé par plusieurs onglets.

    Args:
        prefix: Préfixe unique des clés de widgets (isolation entre onglets).
        defaults: Chunks pré-remplis.

    Returns:
        La liste des chunks tels qu'édités par l'utilisateur.
    """
    edited: list[dict[str, Any]] = []
    for i, chunk in enumerate(defaults):
        with st.expander(
            f"📄 Chunk simulé {i + 1} — `{chunk['chunk_id']}`", expanded=(i == 0)
        ):
            col_id, col_src = st.columns(2)
            chunk_id = col_id.text_input(
                "chunk_id", value=chunk["chunk_id"], key=f"{prefix}_id_{i}"
            )
            source = col_src.text_input(
                "source", value=chunk["source"], key=f"{prefix}_src_{i}"
            )
            content = st.text_area(
                "content",
                value=chunk["content"],
                height=120,
                key=f"{prefix}_content_{i}",
            )
            score = st.slider(
                "relevance_score",
                0.0,
                1.0,
                float(chunk.get("relevance_score", 0.75)),
                0.01,
                key=f"{prefix}_score_{i}",
            )
        edited.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "content": content,
                "relevance_score": score,
            }
        )
    return edited


class SimulatedRetrievalClient:
    """Double de test du module ACTION, injecté dans le nœud `retrieve`.

    Respecte le protocole `RetrievalClient` (`src/reasoning/action_client.py`)
    sans dépendance réseau : le module ACTION n'étant pas branché, c'est le
    seul moyen d'exécuter le graphe de bout en bout.
    """

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.call_count = 0

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Retourne systématiquement le même jeu de chunks simulés."""
        self.call_count += 1
        return RetrievalResponse(
            query_id=request.query_id,
            chunks=self._chunks,
            retrieval_score=0.90 if self._chunks else None,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Rendus Graphviz
# ═════════════════════════════════════════════════════════════════════════════


def build_architecture_digraph() -> graphviz.Digraph:
    """Schéma statique de l'architecture complète du module REASONING."""
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", bgcolor="transparent")
    dot.attr("node", fontname="sans-serif", style="rounded,filled", shape="box")

    dot.node("query", "Requête utilisateur", fillcolor="#e8eaf6", shape="ellipse")
    dot.node("analyze_query", NODE_LABELS["analyze_query"], fillcolor="#e3f2fd")
    dot.node("plan", NODE_LABELS["plan"], fillcolor="#e3f2fd")
    dot.node("retrieve", NODE_LABELS["retrieve"], fillcolor="#fff3e0")
    dot.node("critique", NODE_LABELS["critique"], fillcolor="#e3f2fd")
    dot.node("generate_answer", NODE_LABELS["generate_answer"], fillcolor="#e3f2fd")
    dot.node("verify", NODE_LABELS["verify"], fillcolor="#e3f2fd")
    dot.node("clarify", NODE_LABELS["clarify"], fillcolor="#fffde7")
    dot.node("END", "END\n(VerificationResult)", fillcolor="#e8f5e9", shape="ellipse")

    dot.edge("query", "analyze_query")
    for src, dst, label in GRAPH_EDGES:
        is_loop = (src, dst) in {("critique", "retrieve"), ("verify", "plan")}
        dot.edge(
            src,
            dst,
            label=label,
            color="#c62828" if is_loop else "#546e7a",
            style="dashed" if is_loop else "solid",
            fontsize="9",
            constraint="false" if is_loop else "true",
        )

    # Frontière avec le module ACTION, matérialisée explicitement.
    dot.node(
        "action",
        "MODULE ACTION (astraexec)\nRecherche documentaire\n⏳ non intégré",
        fillcolor="#fbe9e7",
        style="rounded,filled,dashed",
        color="#bf360c",
    )
    dot.edge(
        "retrieve",
        "action",
        label="RetrievalRequest / RetrievalResponse (JSON)",
        color="#bf360c",
        style="dashed",
        fontsize="9",
        dir="both",
    )
    return dot


def build_plan_digraph(plan: ExecutionPlan) -> graphviz.Digraph:
    """DAG des étapes d'un ExecutionPlan, avec leurs dépendances."""
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", bgcolor="transparent")
    dot.attr("node", fontname="sans-serif", shape="box", style="rounded,filled")
    for step in plan.steps:
        wrapped = step.sub_query
        if len(wrapped) > 60:
            wrapped = wrapped[:60] + "…"
        dot.node(step.step_id, f"[{step.step_id}]\n{wrapped}", fillcolor="#e3f2fd")
        for dep in step.depends_on:
            dot.edge(dep, step.step_id, color="#546e7a")
    return dot


def build_execution_digraph(
    visited_nodes: list[str], traversed_edges: set[tuple[str, str]]
) -> graphviz.Digraph:
    """Graphe complet avec mise en évidence du parcours réellement emprunté."""
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", bgcolor="transparent")
    dot.attr("node", fontname="sans-serif", shape="box", style="rounded,filled")

    visits = {node: visited_nodes.count(node) for node in set(visited_nodes)}

    for node, label in NODE_LABELS.items():
        visited = node in visits
        suffix = f"\n×{visits[node]}" if visits.get(node, 0) > 1 else ""
        dot.node(
            node,
            label + suffix,
            fillcolor="#c8e6c9" if visited else "#f5f5f5",
            color="#2e7d32" if visited else "#bdbdbd",
            fontcolor="#1b5e20" if visited else "#9e9e9e",
            penwidth="2" if visited else "1",
        )

    for src, dst, label in GRAPH_EDGES:
        taken = (src, dst) in traversed_edges
        dot.edge(
            src,
            dst,
            label=label,
            color="#2e7d32" if taken else "#e0e0e0",
            fontcolor="#1b5e20" if taken else "#bdbdbd",
            penwidth="3" if taken else "1",
            fontsize="9",
            constraint="false"
            if (src, dst) in {("critique", "retrieve"), ("verify", "plan")}
            else "true",
        )
    return dot


# ═════════════════════════════════════════════════════════════════════════════
# Exécution du pipeline avec trace
# ═════════════════════════════════════════════════════════════════════════════


def describe_transition(node: str, update: dict[str, Any]) -> str:
    """Formule en clair le verdict ayant déclenché la transition sortante."""
    route = update.get("next_route", "")
    agent_state = update.get("agent_state")

    if node == "analyze_query" and agent_state is not None:
        analysis = agent_state.analysis
        if route == ROUTE_CLARIFY:
            return "reasoning_budget = 0 (AMBIGUOUS) → clarify, aucun retrieval"
        return (
            f"query_type = {analysis.query_type.value}, "
            f"reasoning_budget = {analysis.reasoning_budget} → plan"
        )

    if node == "critique" and agent_state is not None:
        evaluation = agent_state.evaluations[-1] if agent_state.evaluations else None
        if evaluation is None:
            return "→ " + route
        verdict = (
            f"is_sufficient={evaluation.is_sufficient}, "
            f"relevance_score={evaluation.relevance_score:.2f}"
        )
        if route == ROUTE_RETRIEVE:
            return f"{verdict} → retrieve (nouvelle tentative ou étape suivante)"
        if route == ROUTE_GENERATE_ANSWER:
            return f"{verdict} → generate_answer"
        return verdict

    if node == "verify" and agent_state is not None:
        verification = agent_state.verification
        if verification is None:
            return "→ " + route
        verdict = (
            f"is_grounded={verification.is_grounded}, "
            f"faithfulness_score={verification.faithfulness_score:.2f}"
        )
        if route == ROUTE_END:
            return f"{verdict} → END"
        if route == ROUTE_PLAN:
            return f"{verdict} → plan (re-planification)"
        return verdict

    if node == "plan" and update.get("pending_step_ids") is not None:
        steps = update["pending_step_ids"]
        return f"plan construit — {len(steps)} étape(s) : {', '.join(steps)}"

    if node == "retrieve":
        response = update.get("last_retrieval_response")
        count = len(response.chunks) if response is not None else 0
        return f"{count} chunk(s) simulé(s) récupéré(s) → critique"

    if node == "generate_answer":
        return "réponse candidate générée → verify"

    if node == "clarify":
        return "demande de clarification produite → END"

    return "→ " + (route or "suite du graphe")


async def _stream_pipeline(
    query: str, client: SimulatedRetrievalClient
) -> dict[str, Any]:
    """Exécute le graphe en streaming et collecte la trace des nœuds."""
    graph = build_graph(retrieval_client=client)

    trace: list[dict[str, Any]] = []
    merged_state: dict[str, Any] = {}
    last_ts = time.perf_counter()

    async for chunk in graph.astream(build_initial_state(query)):
        for node, update in chunk.items():
            now = time.perf_counter()
            if isinstance(update, dict):
                merged_state.update(update)
                trace.append(
                    {
                        "node": node,
                        "reason": describe_transition(node, update),
                        "latency_ms": (now - last_ts) * 1000,
                    }
                )
            last_ts = now

    return {"trace": trace, "state": merged_state}


def execute_pipeline(query: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
    """Exécute le pipeline complet et retourne trace, état final et latence."""
    client = SimulatedRetrievalClient(chunks)
    start = time.perf_counter()
    result = run_async(_stream_pipeline(query, client))
    result["total_ms"] = (time.perf_counter() - start) * 1000
    result["retrieval_calls"] = client.call_count

    visited = [entry["node"] for entry in result["trace"]]
    edges: set[tuple[str, str]] = set()
    for previous, current in zip(visited, visited[1:], strict=False):
        edges.add((previous, current))
    if visited:
        last = visited[-1]
        if last in {"verify", "clarify"}:
            edges.add((last, "END"))
        visited = [*visited, "END"]
    result["visited_nodes"] = visited
    result["traversed_edges"] = edges
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Exécution de commandes qualité (repris du dashboard Sprint 3)
# ═════════════════════════════════════════════════════════════════════════════


def verdict_pytest(output: str, return_code: int) -> tuple[str, str]:
    """Qualifie une sortie pytest de façon STRICTE.

    Vert UNIQUEMENT si la suite est intégralement verte. Tout échec, toute
    erreur de collecte, tout code de retour non nul est signalé en rouge :
    un dashboard qui affiche vert quoi qu'il arrive ne prouve plus rien.

    Note : une tolérance existait pour `test_analyzer_default_params`, seul
    échec historique de la suite. Elle a été retirée une fois cet échec
    résolu — la maintenir aurait rouvert une faille de masquage, en affichant
    vert si ce test venait à échouer de nouveau.
    """
    failed = re.findall(r"^FAILED\s+(\S+)", output, re.MULTILINE)
    passed_match = re.search(r"(\d+) passed", output)
    n_passed = passed_match.group(1) if passed_match else "?"

    if return_code == 0 and not failed:
        return "success", f"✅ Succès — {n_passed} tests passés, aucun échec."

    if failed:
        details = "\n".join(f"- `{name}`" for name in failed)
        return "error", (
            f"❌ **Régression — {len(failed)} échec(s)** ({n_passed} tests "
            f"passés) :\n\n{details}"
        )

    return "error", (
        f"❌ **Échec (code de retour {return_code})** — aucun test en échec "
        "identifié : erreur de collecte ou d'exécution probable. "
        "Consultez la sortie ci-dessus."
    )


def verdict_ruff(output: str, return_code: int) -> tuple[str, str]:
    """Qualifie une sortie Ruff de façon STRICTE.

    Vert uniquement si Ruff ne remonte rien, ou si la totalité des erreurs
    sont des `I001` (tri d'imports) — seul défaut connu et sans effet
    fonctionnel. Tout autre code de règle est une erreur de fond.
    """
    if return_code == 0:
        return "success", "✅ Succès — aucune erreur Ruff."

    codes = set(re.findall(r"^([A-Z]{1,4}\d{3,4})\b", output, re.MULTILINE))

    if codes and codes == {"I001"}:
        count_match = re.search(r"Found (\d+) error", output)
        count = count_match.group(1) if count_match else "?"
        return "success", (
            f"✅ Aucune erreur de fond — {count} erreurs, **toutes `I001`** "
            "(tri d'imports). Cause connue : divergence de version entre le "
            "hook `pre-commit` et le Ruff du projet. Sans effet sur le "
            "comportement du code."
        )

    other = sorted(codes - {"I001"})
    return "error", (
        "❌ **Erreurs Ruff de fond détectées** : "
        + (", ".join(f"`{code}`" for code in other) if other else "voir sortie")
    )


def run_live_command(
    cmd_list: list[str],
    title: str,
    verdict: Any = None,
) -> None:
    """Exécute une commande en sous-processus et streame sa sortie.

    Args:
        cmd_list: La commande et ses arguments.
        title: Libellé affiché au-dessus de la sortie.
        verdict: Fonction `(sortie, code_retour) -> (statut, message)`
            qualifiant le résultat. Si `None`, seul le code de retour est
            interprété.
    """
    st.markdown(f"**{title}** — `{' '.join(cmd_list)}`")
    output_placeholder = st.empty()
    output_text = ""

    try:
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BASE_DIR),
        )
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                output_text += line
                output_placeholder.code(output_text, language="bash")
            process.stdout.close()

        return_code = process.wait()

        if verdict is not None:
            status, message = verdict(output_text, return_code)
        elif return_code == 0:
            status, message = "success", "✅ Succès (code de retour 0)."
        else:
            status, message = "error", f"❌ Échec (code de retour {return_code})."

        {"success": st.success, "error": st.error, "warning": st.warning}[status](
            message
        )
    except Exception as exc:  # noqa: BLE001
        render_error("Impossible d'exécuter la commande", exc)


@st.cache_data(ttl=60)
def load_metrics() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Charge les résultats d'évaluation JSON s'ils existent."""
    analyzer_path = BASE_DIR / "data" / "evaluation" / "analyzer_results.json"
    planner_path = BASE_DIR / "data" / "evaluation" / "planner_results.json"

    analyzer_data, planner_data = None, None
    if analyzer_path.exists():
        with open(analyzer_path, encoding="utf-8") as handle:
            analyzer_data = json.load(handle)
    if planner_path.exists():
        with open(planner_path, encoding="utf-8") as handle:
            planner_data = json.load(handle)
    return analyzer_data, planner_data


# ═════════════════════════════════════════════════════════════════════════════
# Onglet 1 — Vue d'ensemble
# ═════════════════════════════════════════════════════════════════════════════


def render_tab_overview() -> None:
    """Architecture globale, rôle des composants, principes de conception."""
    st.header("Architecture du module REASONING")
    st.markdown(
        "RAG-REASON est le **cerveau** d'un agent RAG multi-sauts : il décide, "
        "planifie, critique et vérifie — sans jamais accéder directement à la "
        "base vectorielle. L'accès aux documents est délégué au module ACTION "
        "par contrat JSON."
    )

    st.graphviz_chart(build_architecture_digraph(), use_container_width=True)

    st.subheader("Composants")
    st.dataframe(COMPONENT_TABLE, use_container_width=True, hide_index=True)

    st.subheader("Principes d'architecture")
    col_left, col_right = st.columns(2)
    with col_left:
        st.info(
            "**Composants purs et sans état**\n\n"
            "Analyzer, Planner, Critic et Verifier *jugent* et retournent un "
            "verdict. Ils ne modifient jamais l'état global, ne décident "
            "jamais du routage et ne comptent jamais leurs propres "
            "itérations. Seul l'orchestrateur décide du flux."
        )
        st.info(
            "**Format TOON**\n\n"
            "Toute sortie LLM structurée est produite en TOON "
            "(`clé :: valeur`, blocs `<<< >>>`) et parsée par "
            "`shared/toon_utils.py`. Aucun parsing JSON ni regex maison."
        )
    with col_right:
        st.info(
            "**Logique découplée du framework**\n\n"
            "La logique de routage vit dans `graph/policy.py`, en Python pur, "
            "sans aucun import de LangGraph. Un test automatisé verrouille "
            "cette contrainte. LangGraph n'est qu'un adaptateur d'exécution."
        )
        st.info(
            "**Repli fail-closed**\n\n"
            "En cas d'échec LLM ou de parsing, chaque composant retourne un "
            "verdict négatif prudent — jamais une heuristique de substitution "
            "silencieuse qui masquerait la panne."
        )

    st.subheader("Frontière REASONING / ACTION")
    st.warning(
        "Ce dépôt implémente **uniquement le module REASONING**. La recherche "
        "documentaire (module ACTION, dépôt `astraexec`) est développée en "
        "binôme et **n'est pas encore branchée**. La communication se fait "
        "exclusivement par JSON via les contrats Pydantic figés "
        "`RetrievalRequest` / `RetrievalResponse`. Dans ce dashboard, tout "
        "retrieval est donc simulé par un double de test injecté."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Onglet 2 — Composants (démo pas à pas)
# ═════════════════════════════════════════════════════════════════════════════


def _section_analyzer() -> None:
    """Démo isolée du Query Analyzer."""
    st.subheader("1️⃣ Query Analyzer — classification et budget")
    st.caption(
        "Premier nœud du graphe. Classifie la requête et alloue le "
        "`reasoning_budget` qui bornera toute l'exécution."
    )

    example = st.selectbox(
        "Exemple pré-rempli",
        list(EXAMPLE_QUERIES),
        key="analyzer_example",
    )
    query = st.text_area(
        "Requête à analyser",
        value=EXAMPLE_QUERIES[example],
        height=80,
        key="analyzer_query",
    )

    if st.button("Analyser la requête", type="primary", key="analyzer_run"):
        if not query.strip():
            st.warning("Saisissez une requête non vide.")
            return
        try:
            with st.spinner("Classification en cours (appel LLM)…"):
                result, latency = timed_call(get_analyzer().analyze, query)
        except Exception as exc:  # noqa: BLE001
            render_error("Échec de l'analyse", exc)
            return

        st.session_state["analysis"] = result
        st.session_state["analysis_query"] = query

    analysis = st.session_state.get("analysis")
    if analysis is None:
        return

    st.success("Analyse disponible — réutilisable par le Planner ci-dessous.")
    col1, col2, col3 = st.columns(3)
    col1.metric("query_type", analysis.query_type.value)
    col2.metric("reasoning_budget", analysis.reasoning_budget)
    col3.metric("confidence", f"{analysis.confidence:.2f}")

    st.write(
        "**Entités détectées :** "
        + (", ".join(analysis.detected_entities) or "_aucune_")
    )

    # confidence == 0.55 est la signature du repli heuristique (analyzer_spec.md §5.2)
    if abs(analysis.confidence - 0.55) < 1e-9:
        st.warning(
            "🔄 **Repli heuristique activé** — le LLM n'a pas produit de "
            "classification exploitable ; la classification provient des "
            "règles Python de secours (confidence forcée à 0.55)."
        )
    else:
        st.caption(
            "Classification produite par le LLM (ou le pré-classificateur regex)."
        )


def _section_planner() -> None:
    """Démo isolée du Planner, réutilisant l'AnalysisResult courant."""
    st.subheader("2️⃣ Planner — décomposition en DAG")
    st.caption(
        "Transforme la requête en graphe acyclique de sous-questions. "
        "Court-circuit Python pour les requêtes SIMPLE (aucun appel LLM)."
    )

    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.info("Lancez d'abord une analyse ci-dessus — le Planner la consomme.")
        return

    query = st.session_state.get("analysis_query", "")
    st.caption(
        f"Analyse réutilisée : `{analysis.query_type.value}`, budget "
        f"`{analysis.reasoning_budget}` — requête : _{query}_"
    )

    if st.button("Décomposer la requête", type="primary", key="planner_run"):
        try:
            spinner_text = (
                "Court-circuit SIMPLE (aucun appel LLM)…"
                if analysis.query_type == QueryType.SIMPLE
                else "Décomposition en cours (appel LLM)…"
            )
            with st.spinner(spinner_text):
                plan, latency = timed_call(get_planner().decompose, query, analysis)
        except Exception as exc:  # noqa: BLE001
            render_error("Échec de la décomposition", exc)
            return

        st.session_state["plan"] = plan
        st.session_state["plan_latency"] = latency

    plan = st.session_state.get("plan")
    if plan is None:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Étapes", len(plan.steps))
    col2.metric("plan_id", plan.plan_id)
    col3.metric("Latence", f"{st.session_state.get('plan_latency', 0):.0f} ms")

    if analysis.query_type == QueryType.SIMPLE:
        st.success(
            "✅ Court-circuit SIMPLE : plan mono-étape construit sans appel LLM."
        )

    st.graphviz_chart(build_plan_digraph(plan), use_container_width=True)
    for step in plan.steps:
        deps = ", ".join(step.depends_on) or "aucune"
        st.markdown(
            f"- **`{step.step_id}`** — {step.sub_query}  \n  _dépend de : {deps}_"
        )


def _section_critic() -> None:
    """Démo isolée du Critic, étape par étape, sur chunks simulés."""
    st.subheader("3️⃣ Critic — suffisance du contexte")
    st.caption(
        "Évalue si le contexte récupéré permet de répondre à une sous-question. "
        "Produit un feedback actionnable destiné au Planner."
    )
    render_simulation_notice()

    plan = st.session_state.get("plan")
    steps: list[PlanStep] = (
        list(plan.steps)
        if plan is not None
        else [
            PlanStep(
                step_id="step_1",
                sub_query="What year was the company that created GPT-4 founded?",
            )
        ]
    )
    if plan is None:
        st.info(
            "Aucun plan en session : une étape de démonstration autonome est "
            "utilisée. Exécutez le Planner ci-dessus pour évaluer un vrai plan."
        )

    critic = get_critic()
    st.caption(
        f"Seuil de suffisance : `{critic.sufficiency_threshold}` · "
        f"max_retries : `{critic.max_retries}`"
    )

    chunks = chunk_editor("critic", DEFAULT_CHUNKS)

    for step in steps:
        with st.expander(f"🔎 `{step.step_id}` — {step.sub_query}", expanded=True):
            if st.button("Évaluer cette étape", key=f"critic_run_{step.step_id}"):
                try:
                    response = RetrievalResponse(
                        query_id=plan.plan_id if plan is not None else "demo-query",
                        chunks=chunks_to_contract(chunks),
                        retrieval_score=0.75,
                    )
                    with st.spinner("Évaluation du contexte (appel LLM)…"):
                        evaluation, latency = timed_call(
                            critic.evaluate, step, response
                        )
                except Exception as exc:  # noqa: BLE001
                    render_error("Échec de l'évaluation", exc)
                    continue

                if evaluation.is_sufficient:
                    st.success("✅ Contexte **suffisant**")
                else:
                    st.error("❌ Contexte **insuffisant**")

                col1, col2 = st.columns(2)
                col1.metric(
                    "relevance_score",
                    f"{evaluation.relevance_score:.2f}",
                    delta=f"seuil {critic.sufficiency_threshold:.2f}",
                    delta_color="off",
                )
                col2.metric("Latence", f"{latency:.0f} ms")
                st.progress(min(evaluation.relevance_score, 1.0))

                if evaluation.feedback:
                    st.markdown(f"**Feedback pour le Planner :** {evaluation.feedback}")
                if evaluation.missing_aspects:
                    st.markdown("**Aspects manquants :**")
                    for aspect in evaluation.missing_aspects:
                        st.markdown(f"- {aspect}")


def _section_verifier() -> None:
    """Démo isolée du Verifier, avec démonstration de l'invariant final_answer."""
    st.subheader("4️⃣ Verifier — fidélité aux sources (groundedness)")
    st.caption(
        "Décompose la réponse en affirmations atomiques et vérifie que chacune "
        "est traçable dans les sources."
    )
    render_simulation_notice()

    verifier = get_verifier()
    st.caption(f"Seuil de fidélité : `{verifier.faithfulness_threshold}`")

    answer = st.text_area(
        "Réponse candidate à vérifier "
        "(l'exemple contient volontairement une affirmation fondée ET une hallucination)",
        value=DEFAULT_VERIFIER_ANSWER,
        height=110,
        key="verifier_answer",
    )
    chunks = chunk_editor("verifier", DEFAULT_CHUNKS)

    if st.button("Vérifier la réponse", type="primary", key="verifier_run"):
        try:
            with st.spinner("Vérification claim par claim (appel LLM)…"):
                result, latency = timed_call(
                    verifier.verify, answer, chunks_to_contract(chunks)
                )
        except Exception as exc:  # noqa: BLE001
            render_error("Échec de la vérification", exc)
            return

        if result.is_grounded:
            st.success("✅ Réponse **fondée** sur les sources")
        else:
            st.error("❌ Réponse **non fondée** — affirmations non traçables détectées")

        col1, col2 = st.columns(2)
        col1.metric(
            "faithfulness_score",
            f"{result.faithfulness_score:.2f}",
            delta=f"seuil {verifier.faithfulness_threshold:.2f}",
            delta_color="off",
        )
        col2.metric("Latence", f"{latency:.0f} ms")
        st.progress(min(result.faithfulness_score, 1.0))

        if result.unsupported_claims:
            st.markdown("**Affirmations non supportées :**")
            for claim in result.unsupported_claims:
                st.markdown(f"- ❌ {claim}")
        else:
            st.markdown("_Aucune affirmation non supportée._")

        # Invariant de conception à démontrer explicitement (verifier_spec.md §7.1)
        st.divider()
        if result.final_answer == answer:
            st.success(
                "🔒 **Invariant vérifié : `final_answer == answer`.** Le Verifier "
                "*juge* la réponse, il ne la modifie jamais — ni troncature, ni "
                "reformulation, ni annotation. La décision de re-générer ou "
                "d'avertir l'utilisateur appartient à l'orchestrateur."
            )
        else:
            st.error("Invariant violé : final_answer diffère de answer.")


def render_tab_components() -> None:
    """Onglet de démonstration composant par composant."""
    st.header("Démonstration pas à pas")
    st.markdown(
        "Chaque composant est exécutable **indépendamment**. Analyzer et "
        "Planner s'enchaînent (le second consomme l'`AnalysisResult` du "
        "premier) ; Critic et Verifier sont testables isolément."
    )
    st.divider()
    _section_analyzer()
    st.divider()
    _section_planner()
    st.divider()
    _section_critic()
    st.divider()
    _section_verifier()


# ═════════════════════════════════════════════════════════════════════════════
# Onglet 3 — Pipeline orchestré
# ═════════════════════════════════════════════════════════════════════════════


def render_tab_pipeline() -> None:
    """Exécution réelle du graphe LangGraph, avec trace et budget."""
    st.header("Pipeline orchestré de bout en bout (LangGraph)")
    st.markdown(
        "Exécution **réelle** du graphe compilé : Analyzer, Planner, Critic et "
        "Verifier sont les vrais composants et appellent réellement le LLM. "
        "Seul le nœud `retrieve` est simulé."
    )
    render_simulation_notice()
    st.info(
        "⏱️ **Durée attendue : 1 à 3 minutes.** Le pipeline enchaîne plusieurs "
        "appels au modèle 7B local. La latence de chaque nœud est affichée à "
        "l'issue de l'exécution."
    )

    example = st.selectbox(
        "Exemple pré-rempli", list(EXAMPLE_QUERIES), key="pipeline_example"
    )
    query = st.text_area(
        "Requête à traiter",
        value=EXAMPLE_QUERIES[example],
        height=80,
        key="pipeline_query",
    )

    with st.expander("📄 Chunks simulés injectés dans le nœud `retrieve`"):
        chunks = chunk_editor("pipeline", DEFAULT_CHUNKS)

    if st.button("▶️ Exécuter le pipeline complet", type="primary", key="pipeline_run"):
        if not query.strip():
            st.warning("Saisissez une requête non vide.")
            return
        try:
            with st.spinner("Exécution du graphe — plusieurs appels LLM en cours…"):
                result = execute_pipeline(query, chunks_to_contract(chunks))
        except Exception as exc:  # noqa: BLE001
            render_error("Échec de l'exécution du pipeline", exc)
            return
        st.session_state["pipeline_result"] = result

    result = st.session_state.get("pipeline_result")
    if result is None:
        return

    state = result["state"]
    agent_state = state.get("agent_state")

    # ── Budget global ────────────────────────────────────────────────────────
    st.subheader("Budget de raisonnement")
    st.caption(
        "Compteur global unique : `feedback_loop_count` est incrémenté à "
        "chaque passage par `critique` **ou** `verify`, et plafonné par le "
        "`reasoning_budget` alloué par l'Analyzer (docs/graph_spec.md §3)."
    )
    budget = agent_state.analysis.reasoning_budget if agent_state.analysis else 0
    consumed = agent_state.feedback_loop_count if agent_state else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Budget alloué", budget)
    col2.metric("Budget consommé", consumed)
    col3.metric("Appels retrieve", result["retrieval_calls"])
    if budget > 0:
        st.progress(min(consumed / budget, 1.0), text=f"{consumed} / {budget}")
    if consumed >= budget > 0:
        st.warning(
            "🛑 Budget épuisé — la garde anti-boucle globale a forcé la sortie. "
            "C'est le mécanisme qui garantit la terminaison du graphe."
        )

    # ── Trace d'exécution ────────────────────────────────────────────────────
    st.subheader("Trace d'exécution")
    st.caption("Nœuds réellement traversés, dans l'ordre, avec le verdict déclencheur.")
    for i, entry in enumerate(result["trace"], start=1):
        st.markdown(
            f"**{i}. `{entry['node']}`** — {entry['reason']}  \n"
            f"<span style='color:#78909c;font-size:0.85em'>latence : "
            f"{entry['latency_ms']:.0f} ms</span>",
            unsafe_allow_html=True,
        )
    st.metric("Latence totale du pipeline", f"{result['total_ms'] / 1000:.1f} s")

    # ── Graphe parcouru ──────────────────────────────────────────────────────
    st.subheader("Chemin emprunté dans le graphe")
    st.caption("Vert = nœuds et arêtes réellement parcourus · Gris = non empruntés.")
    st.graphviz_chart(
        build_execution_digraph(result["visited_nodes"], result["traversed_edges"]),
        use_container_width=True,
    )

    # ── Résultat final ───────────────────────────────────────────────────────
    st.subheader("Résultat final — VerificationResult")
    verification = agent_state.verification if agent_state else None
    if verification is None:
        st.warning("Le graphe s'est terminé sans produire de VerificationResult.")
        return

    if verification.is_grounded:
        st.success("✅ Réponse finale **fondée** sur les sources")
    else:
        st.error("❌ Réponse finale **non fondée**")

    col1, col2 = st.columns(2)
    col1.metric("faithfulness_score", f"{verification.faithfulness_score:.2f}")
    col2.metric("is_grounded", str(verification.is_grounded))

    st.markdown("**Réponse finale :**")
    st.info(verification.final_answer)

    if verification.unsupported_claims:
        st.markdown("**Affirmations non supportées :**")
        for claim in verification.unsupported_claims:
            st.markdown(f"- ❌ {claim}")

    if agent_state and agent_state.evaluations:
        with st.expander("🔎 Détail des évaluations du Critic"):
            for evaluation in agent_state.evaluations:
                st.markdown(
                    f"- `{evaluation.step_id}` — is_sufficient="
                    f"**{evaluation.is_sufficient}**, score="
                    f"{evaluation.relevance_score:.2f}"
                    + (f" · _{evaluation.feedback}_" if evaluation.feedback else "")
                )


# ═════════════════════════════════════════════════════════════════════════════
# Onglet 4 — Qualité & Tests
# ═════════════════════════════════════════════════════════════════════════════

QUALITY_TABLE: list[dict[str, str]] = [
    {
        "Sprint": "1",
        "Périmètre": "Contrats Pydantic",
        "Tests": "48",
        "Couverture": "100 %",
        "Statut": "✅",
    },
    {
        "Sprint": "2",
        "Périmètre": "Query Analyzer",
        "Tests": "30",
        "Couverture": "91 %",
        "Statut": "✅",
    },
    {
        "Sprint": "3",
        "Périmètre": "Planner",
        "Tests": "4",
        "Couverture": "—",
        "Statut": "✅",
    },
    {
        "Sprint": "4",
        "Périmètre": "Critic",
        "Tests": "24",
        "Couverture": "—",
        "Statut": "✅",
    },
    {
        "Sprint": "5",
        "Périmètre": "Verifier",
        "Tests": "22 + 9 intégration",
        "Couverture": "96 %",
        "Statut": "✅",
    },
    {
        "Sprint": "6",
        "Périmètre": "Orchestration LangGraph",
        "Tests": "33",
        "Couverture": "91 %",
        "Statut": "✅",
    },
    {
        "Sprint": "—",
        "Périmètre": "TOON (shared)",
        "Tests": "71",
        "Couverture": "—",
        "Statut": "✅",
    },
]


def render_tab_quality() -> None:
    """Tableau qualité statique et exécution live des suites de tests."""
    st.header("Qualité et tests")
    st.markdown(
        "**232 tests passants — suite intégralement verte**, **91 % de "
        "couverture**, `mypy --strict` sans erreur sur les sources **et** les "
        "tests, `ruff` conforme."
    )
    st.caption(
        'La couverture est mesurée sur `tests/ -m "not integration"`, qui '
        "inclut `tests/integration/test_graph.py` — celui-ci n'exige pas "
        "Ollama et couvre l'orchestration. La restreindre à `tests/unit/` "
        "sous-estimerait le graphe (86 % → 20 % sur `nodes.py`)."
    )
    st.dataframe(QUALITY_TABLE, use_container_width=True, hide_index=True)

    st.success(
        "✅ **Aucun échec — la suite est intégralement verte.** "
        "`test_analyzer_default_params`, seul échec historique, est résolu : "
        "il assertait `timeout == 15.0` depuis le commit initial du Sprint 3 "
        "alors que `QueryAnalyzer` applique `20.0` depuis ce même commit — il "
        "n'avait donc **jamais** pu passer, et sa justification renvoyait à un "
        "document absent du dépôt. Après arbitrage, c'est l'**assertion** qui "
        "a été corrigée : les 20 s couvrent le démarrage à froid mesuré à "
        "9,6 s et laissent une marge au-dessus de la médiane de 4,0 s du "
        "chemin LLM. Le code source n'a pas été modifié."
    )

    st.subheader("Exécution en direct")
    st.caption(
        "Les commandes s'exécutent dans un sous-processus isolé, à la racine "
        "du dépôt. La sortie s'affiche au fil de l'eau."
    )

    commands: list[tuple[str, list[str], str, Any]] = [
        (
            "🧪 Suite rapide (hors intégration)",
            ["uv", "run", "pytest", "tests/", "-q", "-m", "not integration"],
            "Suite complète hors tests nécessitant Ollama",
            verdict_pytest,
        ),
        (
            "🔍 Tests du Verifier",
            ["uv", "run", "pytest", "tests/unit/test_verifier.py", "-v"],
            "Sprint 5 — Verifier",
            verdict_pytest,
        ),
        (
            "🔀 Orchestration (policy + graphe)",
            [
                "uv",
                "run",
                "pytest",
                "tests/unit/test_policy.py",
                "tests/integration/test_graph.py",
                "-v",
            ],
            "Sprint 6 — routage et graphe",
            verdict_pytest,
        ),
        (
            "📊 Couverture de code",
            [
                "uv",
                "run",
                "pytest",
                "--cov=reasoning",
                "--cov-report=term-missing",
                "tests/",
                "-m",
                "not integration",
            ],
            "Couverture sur toute la suite exécutable sans Ollama",
            verdict_pytest,
        ),
        (
            "🧹 Ruff",
            ["uv", "run", "ruff", "check", "src/", "tests/"],
            "Analyse statique",
            verdict_ruff,
        ),
        (
            "🔎 Mypy strict",
            ["uv", "run", "mypy", "--strict", "src/", "tests/"],
            "Typage strict (sources + tests)",
            None,
        ),
    ]

    for label, cmd, title, verdict in commands:
        if st.button(label, key=f"cmd_{label}", use_container_width=True):
            run_live_command(cmd, title, verdict)


# ═════════════════════════════════════════════════════════════════════════════
# Onglet 5 — Métriques d'évaluation
# ═════════════════════════════════════════════════════════════════════════════


def render_tab_metrics() -> None:
    """Métriques d'évaluation historiques et statut de l'évaluation RAGAS."""
    st.header("Métriques d'évaluation")

    st.warning(
        "⏳ **Évaluation RAGAS (Sprint 7) en attente.** Les métriques RAGAS "
        "(`faithfulness`, `answer_relevancy`, `context_precision`, "
        "`context_recall`) nécessitent un **corpus documentaire réel** et donc "
        "l'intégration du module ACTION. Produire un baseline sur des chunks "
        "simulés n'aurait aucune valeur scientifique : les scores mesureraient "
        "la qualité des données inventées, pas celle du système. "
        "Les métriques ci-dessous proviennent des évaluations menées aux "
        "sprints 2 et 3 sur le dataset HotpotQA."
    )

    analyzer_data, planner_data = load_metrics()

    if not analyzer_data and not planner_data:
        st.info(
            "Aucun résultat d'évaluation trouvé. Générez-les avec :\n\n"
            "`uv run python scripts/02_evaluate_analyzer.py`\n\n"
            "`uv run python scripts/03_evaluate_planner.py`"
        )
        return

    col_analyzer, col_planner = st.columns(2)

    with col_analyzer:
        st.subheader("🔍 Query Analyzer")
        if analyzer_data and "summary" in analyzer_data:
            summary = analyzer_data["summary"]
            st.metric("Requêtes évaluées", summary.get("total", "N/A"))
            st.metric("Accuracy globale", f"{summary.get('accuracy_pct', 0)} %")
            st.metric(
                "Accuracy bridge (MULTI_HOP)",
                f"{summary.get('bridge_accuracy_pct', 0)} %",
            )
            st.metric(
                "Accuracy comparison (COMPARATIVE)",
                f"{summary.get('comparison_accuracy_pct', 0)} %",
            )

            by_path = summary.get("accuracy_by_path")
            latency = summary.get("latency")
            if by_path:
                st.markdown("**Ventilation par chemin de décision**")
                st.caption(
                    "L'Analyzer a trois chemins. Les agréger masquait deux "
                    "régimes très différents : le pré-classificateur regex "
                    "répond instantanément, le LLM prend plusieurs secondes."
                )
                labels = {
                    "regex": "Pré-classificateur regex",
                    "llm": "LLM",
                    "heuristic_fallback": "Repli heuristique",
                }
                rows = []
                for key, label in labels.items():
                    stats = by_path.get(key, {})
                    if not stats.get("n"):
                        continue
                    lat = (latency or {}).get("by_path", {}).get(key, {})
                    rows.append(
                        {
                            "Chemin": label,
                            "Volume": f"{stats['n']} ({stats['share_pct']:.1f} %)",
                            "Accuracy": f"{stats['accuracy_pct']:.1f} %",
                            "Latence méd.": f"{lat.get('median_ms', 0):.0f} ms",
                        }
                    )
                if rows:
                    # st.table plutôt que st.dataframe : rendu DOM statique,
                    # lisible dans une colonne étroite et sans virtualisation
                    # (le grid interactif se replie et devient illisible ici).
                    st.table(rows)

            if latency:
                warm = latency.get("warm_all", {})
                st.metric(
                    "Latence médiane (à chaud, tous chemins)",
                    f"{warm.get('median_ms', 0):.0f} ms",
                    delta=f"moyenne {warm.get('mean_ms', 0):.0f} ms",
                    delta_color="off",
                )
                st.caption(
                    f"Démarrage à froid isolé : {latency.get('cold_start_ms', 0):.0f} ms "
                    "(chargement du modèle Ollama, exclu des statistiques). "
                    f"Ancienne moyenne globale diluée : "
                    f"{summary.get('avg_latency_ms', 0):.0f} ms."
                )
        else:
            st.info("Résultats Analyzer indisponibles.")

    with col_planner:
        st.subheader("📐 Planner (génération de DAG)")
        if planner_data and "summary" in planner_data:
            summary = planner_data["summary"]
            st.metric("Requêtes évaluées", summary.get("total", "N/A"))
            st.metric(
                "Validité TOON (sortie LLM brute)",
                f"{summary.get('toon_validity_pct', 0)} %",
            )
            st.caption(
                "Mesurée sur la réponse brute du LLM : le bloc `<<<…>>>` "
                "est-il extractible et parsable ? Cette métrique ne dépend "
                "plus du taux de repli, qui mesure autre chose."
            )
            st.metric(
                "Validité DAG (acyclique)",
                f"{summary.get('dag_validity_pct', 0)} %",
            )
            st.metric(
                "Taux de repli séquentiel",
                f"{summary.get('fallback_rate_pct', 0)} %",
            )
            st.caption(
                "Métrique **distincte** de la validité de format : un repli "
                "signale une planification dégradée, pas un TOON malformé."
            )
            st.metric("Étapes moyennes / plan", summary.get("avg_steps_per_plan", 0))

            latency = summary.get("latency")
            if latency:
                st.metric(
                    "Latence médiane (à chaud)",
                    f"{latency.get('warm_median_ms', 0):.0f} ms",
                    delta=f"moyenne {latency.get('warm_mean_ms', 0):.0f} ms",
                    delta_color="off",
                )
                st.caption(
                    f"Démarrage à froid isolé : "
                    f"{latency.get('cold_start_ms', 0):.0f} ms. "
                    f"Ancienne moyenne globale : "
                    f"{summary.get('avg_latency_ms', 0):.0f} ms."
                )
            else:
                st.metric(
                    "Latence moyenne", f"{summary.get('avg_latency_ms', 0):.0f} ms"
                )
        else:
            st.info("Résultats Planner indisponibles.")


# ═════════════════════════════════════════════════════════════════════════════
# Barre latérale et point d'entrée
# ═════════════════════════════════════════════════════════════════════════════


def render_sidebar() -> None:
    """Contexte projet, configuration et statut d'avancement."""
    st.sidebar.title("🧠 RAG-REASON")
    st.sidebar.caption("Module REASONING d'un agent RAG multi-sauts")

    st.sidebar.markdown("### Configuration")
    analyzer = get_analyzer()
    verifier = get_verifier()
    st.sidebar.markdown(
        f"- **Modèle de raisonnement :** `{verifier.model}`\n"
        f"- **Modèle de classification :** `{analyzer.model}`\n"
        f"- **Endpoint :** `{verifier.api_base}`\n"
        f"- **Orchestration :** LangGraph\n"
        f"- **Format LLM :** TOON"
    )

    st.sidebar.markdown("### Avancement")
    st.sidebar.markdown(
        "- ✅ S0-1 · Infra & contrats\n"
        "- ✅ S2 · Query Analyzer\n"
        "- ✅ S3 · Planner\n"
        "- ✅ S4 · Critic\n"
        "- ✅ S5 · Verifier\n"
        "- ✅ S6 · Orchestration LangGraph\n"
        "- ⏳ S7 · Évaluation RAGAS\n"
        "- ⏳ S8 · CI & documentation"
    )

    st.sidebar.error(
        "**Module ACTION non branché**\n\n"
        "La recherche documentaire réelle (dépôt `astraexec`, développé en "
        "binôme) n'est pas encore intégrée. **Tout retrieval affiché dans ce "
        "dashboard est simulé.**"
    )

    st.sidebar.caption(
        "Prérequis : `ollama serve` actif et modèle `qwen2.5:7b` disponible."
    )


def main() -> None:
    """Point d'entrée du dashboard."""
    st.set_page_config(
        page_title="RAG-REASON — Dashboard",
        page_icon="🧠",
        layout="wide",
    )
    render_sidebar()

    st.title("RAG-REASON — Moteur de raisonnement pour agent RAG")
    st.caption("Query Analyzer · Planner · Critic · Verifier · Orchestration LangGraph")

    tab_overview, tab_components, tab_pipeline, tab_quality, tab_metrics = st.tabs(
        [
            "🏛️ Vue d'ensemble",
            "🧩 Composants",
            "🔀 Pipeline orchestré",
            "🛡️ Qualité & Tests",
            "📊 Métriques",
        ]
    )

    with tab_overview:
        render_tab_overview()
    with tab_components:
        render_tab_components()
    with tab_pipeline:
        render_tab_pipeline()
    with tab_quality:
        render_tab_quality()
    with tab_metrics:
        render_tab_metrics()


main()
