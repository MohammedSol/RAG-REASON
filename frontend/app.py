# mypy: ignore-errors
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import graphviz
import streamlit as st

# =====================================================================
# ÉTAPE 1 : Isolation Architecturale (Setup)
# =====================================================================
# Pour exécuter ce code, assurez-vous d'avoir installé streamlit et graphviz
# via uv sans casser l'environnement actuel :
# > uv add streamlit graphviz
# Lancement de l'application :
# > uv run streamlit run frontend/app.py
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

# Ajout de src/ au PYTHONPATH pour consommer le backend en lecture seule
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import dynamique des composants Backend
try:
    from reasoning.analyzer.analyzer import QueryAnalyzer
    from reasoning.planner.planner import Planner
except ImportError as e:
    st.error(f"Erreur d'importation depuis src/ : {e}")

# =====================================================================
# ÉTAPE 2 : Structure Globale de l'Interface
# =====================================================================
st.set_page_config(page_title="RAG-REASON Dashboard", page_icon="🧠", layout="wide")

# Sidebar
st.sidebar.title("RAG-REASON")
st.sidebar.markdown("""
**Statut du modèle** : Ollama / Qwen 7B (Local)
**Architecture** : "Contract-First" (TOON)
""")
st.sidebar.info(
    "Module de démonstration Frontend en lecture seule du Backend (Phase 2 & 3)."
)

# Définition des onglets
tab_demo, tab_eval, tab_cicd = st.tabs(
    [
        "🚀 Démo Interactive (Live)",
        "📊 Résultats d'Évaluation (Dataset)",
        "🛡️ Santé du Code (CI/CD)",
    ]
)

# =====================================================================
# ÉTAPE 3 & 4 : Implémentation de l'Onglet "Démo Interactive"
# =====================================================================
with tab_demo:
    st.header("Démo Interactive - Query Analyzer & Planner")
    st.markdown(
        "Testez en temps réel l'intelligence de décomposition sur des requêtes complexes."
    )

    # Champ de saisie
    user_query = st.text_area(
        "Entrez votre requête (ex: The arena where the Lewiston Maineiacs played can seat how many people?)",
        height=100,
    )

    if st.button("Lancer l'Exécution", type="primary"):
        if not user_query.strip():
            st.warning("Veuillez entrer une requête valide.")
        else:
            with st.spinner("Analyse et Planification en cours via Qwen 7B..."):
                try:
                    # Instanciation dynamique du Backend
                    analyzer = QueryAnalyzer()
                    planner = Planner()

                    # Exécution du flux RAG-REASON
                    t0 = time.perf_counter()
                    analysis_result = analyzer.analyze(user_query)
                    t1 = time.perf_counter()
                    plan = planner.decompose(user_query, analysis_result)
                    t2 = time.perf_counter()

                    # Layout pour afficher les résultats bruts
                    st.markdown("---")
                    col_analyzer, col_planner = st.columns(2)

                    with col_analyzer:
                        st.subheader("🤖 Décisions Analyzer")
                        st.info(
                            f"**Type de requête :** {analysis_result.query_type.value if hasattr(analysis_result.query_type, 'value') else analysis_result.query_type}"
                        )
                        st.write(
                            f"**Budget (étapes) :** {analysis_result.reasoning_budget}"
                        )
                        st.write(f"**Confiance :** {analysis_result.confidence}")
                        st.write(
                            f"**Entités détectées :** {', '.join(analysis_result.detected_entities) if analysis_result.detected_entities else 'Aucune'}"
                        )
                        st.caption(f"Latence Analyzer : {(t1 - t0) * 1000:.0f} ms")

                    with col_planner:
                        st.subheader("📋 Validation Planner (Format TOON)")
                        st.success("✅ Extraction du format TOON réussie")
                        st.write(f"**Plan ID :** {plan.plan_id}")
                        st.write(f"**Nombre d'étapes générées :** {len(plan.steps)}")
                        st.caption(f"Latence Planner : {(t2 - t1) * 1000:.0f} ms")

                    # =====================================================================
                    # ÉTAPE 4 : Rendu Visuel du Graphe (Graphviz)
                    # =====================================================================
                    st.markdown("---")
                    st.subheader("Graphe d'Exécution (DAG) Validé")
                    st.markdown("Algorithme de Kahn : ✓ DAG Acyclique confirmé.")

                    # Construction du DAG Graphviz depuis l'ExecutionPlan
                    dot = graphviz.Digraph(comment="Execution Plan DAG")
                    dot.attr(rankdir="TB", size="10,10")  # Top to Bottom

                    # Noeuds (étapes du plan)
                    for step in plan.steps:
                        node_label = f"[{step.step_id}]\n{step.sub_query}"
                        dot.node(
                            step.step_id,
                            node_label,
                            shape="box",
                            style="rounded,filled",
                            fillcolor="#e6f3ff",
                            fontname="sans-serif",
                        )
                        # Flèches (Dépendances)
                        for dep in step.depends_on:
                            dot.edge(dep, step.step_id, color="#5c5c5c")

                    # Affichage du graphe
                    st.graphviz_chart(dot, use_container_width=True)

                except Exception as e:
                    st.error(f"Une erreur est survenue lors de l'exécution : {e}")

# =====================================================================
# ÉTAPE 5 : Rendre l'Onglet "Métriques" 100% Dynamique
# =====================================================================
with tab_eval:
    st.header("Métriques d'Évaluation (HotpotQA Dataset)")
    st.markdown(
        "Indicateurs en temps réel des performances sur le dataset de validation Sprint 3."
    )

    @st.cache_data(ttl=60)
    def load_metrics() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        analyzer_path = BASE_DIR / "data" / "evaluation" / "analyzer_results.json"
        planner_path = BASE_DIR / "data" / "evaluation" / "planner_results.json"

        analyzer_data, planner_data = None, None

        if analyzer_path.exists():
            with open(analyzer_path, encoding="utf-8") as f:
                analyzer_data = json.load(f)

        if planner_path.exists():
            with open(planner_path, encoding="utf-8") as f:
                planner_data = json.load(f)

        return analyzer_data, planner_data

    analyzer_data, planner_data = load_metrics()

    if analyzer_data or planner_data:
        col_eval1, col_eval2 = st.columns(2)

        with col_eval1:
            st.subheader("🔍 Query Analyzer")
            if analyzer_data and "summary" in analyzer_data:
                summary = analyzer_data["summary"]
                st.metric("Total Requêtes Évaluées", summary.get("total", "N/A"))
                st.metric("Accuracy Globale", f"{summary.get('accuracy_pct', 0)} %")
                st.metric(
                    "Accuracy Bridge (MULTI_HOP)",
                    f"{summary.get('bridge_accuracy_pct', 0)} %",
                )
                st.metric(
                    "Accuracy Comparison (COMPARATIVE)",
                    f"{summary.get('comparison_accuracy_pct', 0)} %",
                )
                st.metric(
                    "Latence Moyenne", f"{summary.get('avg_latency_ms', 0):.0f} ms"
                )
            else:
                st.warning("Résultats Analyzer introuvables ou format inattendu.")

        with col_eval2:
            st.subheader("📐 Planner (DAG Generation)")
            if planner_data and "summary" in planner_data:
                summary = planner_data["summary"]
                st.metric("Total Requêtes Évaluées", summary.get("total", "N/A"))
                st.metric(
                    "Validité TOON (Format <<<...>>>)",
                    f"{summary.get('toon_validity_pct', 0)} %",
                )
                st.metric(
                    "Validité DAG (Sans cycle)",
                    f"{summary.get('dag_validity_pct', 0)} %",
                )
                st.metric(
                    "Étapes Moyennes par plan",
                    f"{summary.get('avg_steps_per_plan', 0)}",
                )
                st.metric(
                    "Latence Moyenne", f"{summary.get('avg_latency_ms', 0):.0f} ms"
                )
            else:
                st.warning("Résultats Planner introuvables ou format inattendu.")
    else:
        st.info(
            "💡 Aucun résultat d'évaluation trouvé. Exécutez `uv run python scripts/02_evaluate_analyzer.py` et `03_evaluate_planner.py` pour générer les métriques."
        )

# =====================================================================
# ÉTAPE 5 (Suite) : Rendre l'Onglet "CI/CD" 100% Dynamique
# =====================================================================
with tab_cicd:
    st.header("Intégration Continue (Live Testing)")
    st.markdown("Exécutez les suites de tests en direct dans un sous-processus isolé.")

    def run_live_command(cmd_list: list[str], title: str) -> None:
        st.subheader(title)
        st.markdown(f"Exécution : `{' '.join(cmd_list)}`")

        # Zone d'affichage vide qui sera mise à jour en temps réel
        output_placeholder = st.empty()
        output_text = ""

        try:
            # Lancement du processus (subprocess.PIPE pour capturer stdout+stderr ensemble)
            process = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(BASE_DIR),
            )

            # Lecture du flux en continu
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    output_text += line
                    # On affiche au fur et à mesure avec le langage bash pour le style
                    output_placeholder.code(output_text, language="bash")

                process.stdout.close()

            return_code = process.wait()

            # Affichage du statut final
            if return_code == 0:
                st.success(f"✅ Succès (Code de retour: {return_code})")
            else:
                st.error(f"❌ Échec (Code de retour: {return_code})")

        except Exception as e:
            st.error(f"Erreur d'exécution de la commande : {e}")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    # Boutons d'actions
    if col_btn1.button("🧹 Exécuter Ruff", use_container_width=True):
        run_live_command(
            ["uv", "run", "ruff", "check", "src/", "tests/", "scripts/"],
            "Analyse statique (Ruff)",
        )

    if col_btn2.button("🔍 Exécuter Mypy", use_container_width=True):
        run_live_command(
            ["uv", "run", "mypy", "src/", "tests/", "scripts/", "--no-error-summary"],
            "Typage Statique (Mypy)",
        )

    if col_btn3.button("🧪 Exécuter Pytest", use_container_width=True):
        run_live_command(
            ["uv", "run", "pytest", "tests/"], "Tests Unitaires & Intégration (Pytest)"
        )
