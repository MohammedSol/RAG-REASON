"""
Tests d'intégration bout-en-bout — Sprint I4 : REASONING × ACTION réels.

Ces tests font traverser au système COMPLET de vraies questions du jeu
d'évaluation HotpotQA, avec un vrai retrieval sur le corpus indexé, jusqu'à
une réponse vérifiée. Aucun composant n'est simulé.

Prérequis :
    - `ollama serve` en arrière-plan, modèles qwen2.5:3b et qwen2.5:7b
    - l'API du module ACTION sur http://localhost:8000
      (dépôt `astraexec-integration`, venv isolé) :
          .venv/Scripts/python.exe -m uvicorn app.api.main:app --port 8000
      Le premier appel reconstruit l'index (~5 s).

Exécution :
    uv run pytest tests/integration/test_end_to_end.py -v -m integration

`USE_REAL_ACTION` est positionné à `true` par la fixture `real_action_env`,
pour ces tests UNIQUEMENT. Le défaut du dépôt reste faux.

INDISPONIBILITÉ DES SERVICES — échec explicite, jamais de skip
--------------------------------------------------------------
Le graphe applique un repli *fail-closed* : si le module ACTION est
injoignable, `retrieve` produit une réponse vide et l'exécution se poursuit.
Sans précaution, les paliers 1 à 3 passeraient au vert en ne mesurant qu'un
pipeline dégradé. Une sonde de connectivité échoue donc explicitement avant
toute mesure, sur les DEUX services.

ROBUSTESSE À LA VARIABILITÉ DU LLM
----------------------------------
Les assertions portent sur le comportement STRUCTUREL — le pipeline atteint
END, le budget est respecté, `is_grounded` est cohérent avec les sources
disponibles — jamais sur des valeurs exactes de score ou des formulations de
réponse, qui varient d'une exécution à l'autre malgré `temperature=0`.

Les défauts mis en évidence par ces paliers sont documentés dans
`docs/integration_e2e_findings.md`. Les tests les CONSTATENT (et échoueront
si le comportement change) ; ils ne les corrigent pas.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from reasoning.action_client import ActionClient
from reasoning.contracts.internal_models import QueryType
from reasoning.critic import Critic
from reasoning.graph.graph import build_graph
from reasoning.graph.state import GraphState, build_initial_state

# ─────────────────────────────────────────────────────────────────────────────
# Sondes de connectivité — les deux services, échec parlant
# ─────────────────────────────────────────────────────────────────────────────

_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_ACTION_BASE_URL: str = os.getenv("ACTION_BASE_URL", "http://localhost:8000")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_SET = _REPO_ROOT / "data" / "processed" / "hotpotqa_sprint3.json"
_MANIFEST = _REPO_ROOT / "data" / "corpus" / "corpus_manifest.json"

# Questions sélectionnées dans le jeu d'évaluation (Sprint I4, étape de
# présélection). Toutes ont leurs `supporting_facts` complets dans le corpus.
QID_SIMPLE = "5a75e05c55429976ec32bc5f"
QID_MULTI_HOP = "5a8c7595554299585d9e36b6"

# Question volontairement absente du corpus — entités inventées.
UNANSWERABLE_QUERY = (
    "What was the exact catalogue number of the Zorblatt Industries "
    "quantum flange manufactured on Titan in 2387?"
)


@pytest.fixture(scope="module")
def require_ollama() -> None:
    """Échoue explicitement — sans skip — si Ollama est injoignable."""
    try:
        httpx.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=5.0).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.fail(
            f"Ollama est indisponible sur {_OLLAMA_BASE_URL} "
            f"({type(exc).__name__}: {exc}). Lancer `ollama serve`. Ces tests "
            "ne sont volontairement PAS skippés : un skip masquerait "
            "l'absence de mesure.",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def require_action_api() -> None:
    """Échoue explicitement si le module ACTION est injoignable.

    Indispensable : le repli fail-closed du nœud `retrieve` rendrait sinon
    les paliers 1 à 3 verts sur un pipeline entièrement dégradé.
    """
    try:
        httpx.get(f"{_ACTION_BASE_URL}/health", timeout=5.0).raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.fail(
            f"Le module ACTION est indisponible sur {_ACTION_BASE_URL} "
            f"({type(exc).__name__}: {exc}). Lancer, depuis le dépôt "
            "astraexec-integration : "
            "`.venv/Scripts/python.exe -m uvicorn app.api.main:app --port 8000`. "
            "Sans lui, le graphe appliquerait son repli fail-closed et ces "
            "tests mesureraient un pipeline vide.",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def real_action_env() -> Any:
    """Active `USE_REAL_ACTION` pour ce module uniquement."""
    previous = os.environ.get("USE_REAL_ACTION")
    os.environ["USE_REAL_ACTION"] = "true"
    yield
    if previous is None:
        os.environ.pop("USE_REAL_ACTION", None)
    else:
        os.environ["USE_REAL_ACTION"] = previous


# ─────────────────────────────────────────────────────────────────────────────
# Données du jeu d'évaluation
# ─────────────────────────────────────────────────────────────────────────────


def _eval_question(qid: str) -> str:
    """Texte de la question du jeu d'évaluation."""
    records = json.loads(_EVAL_SET.read_text(encoding="utf-8"))
    for record in records:
        if record["id"] == qid:
            question: str = record["question"]
            return question
    raise AssertionError(f"question {qid} absente de {_EVAL_SET.name}")


def _gold_sources(qid: str) -> set[str]:
    """Noms de fichiers `.txt` des articles gold de la question."""
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    table = manifest["title_to_file"]
    for entry in manifest["questions"]:
        if entry["id"] == qid:
            return {table[t] for t in entry["gold_titles"] if t in table}
    raise AssertionError(f"question {qid} absente du manifeste du corpus")


# ─────────────────────────────────────────────────────────────────────────────
# Exécution tracée — un seul passage par palier, partagé entre assertions
# ─────────────────────────────────────────────────────────────────────────────


class Run:
    """Résultat d'une exécution complète du graphe, avec sa trace."""

    def __init__(self, query: str) -> None:
        self.query = query
        self.path: list[str] = []
        self.node_ms: list[tuple[str, int]] = []
        self.final: dict[str, Any] = {}
        self.total_ms: int = 0

    @property
    def agent_state(self) -> Any:
        return self.final["agent_state"]

    @property
    def chunks(self) -> list[Any]:
        chunks: list[Any] = self.final.get("retrieved_chunks") or []
        return chunks

    @property
    def sources(self) -> set[str]:
        return {c.source for c in self.chunks}

    def report(self) -> str:
        """Trace lisible, jointe aux échecs d'assertion."""
        lines = [
            f"\n  query      : {self.query}",
            f"  chemin     : {' -> '.join(self.path)}",
            f"  latences   : {self.node_ms} (total {self.total_ms} ms)",
            f"  chunks     : {len(self.chunks)} — sources {sorted(self.sources)}",
            f"  réponse    : {self.final.get('answer')!r}",
        ]
        agent = self.final.get("agent_state")
        if agent is not None:
            if agent.analysis is not None:
                lines.append(
                    f"  analyzer   : {agent.analysis.query_type.value} "
                    f"budget={agent.analysis.reasoning_budget}"
                )
            if agent.plan is not None:
                lines.append(
                    f"  plan       : {[s.sub_query for s in agent.plan.steps]}"
                )
            lines.append(
                f"  critic     : "
                f"{[(e.step_id, e.is_sufficient, e.relevance_score) for e in agent.evaluations]}"
            )
            if agent.verification is not None:
                lines.append(
                    f"  verifier   : is_grounded={agent.verification.is_grounded} "
                    f"score={agent.verification.faithfulness_score} "
                    f"unsupported={agent.verification.unsupported_claims}"
                )
            lines.append(f"  budget     : {agent.feedback_loop_count}")
        return "\n".join(lines)


def _loop_upper_bound(run: Run) -> int:
    """Borne supérieure du nombre de critiques d'une exécution (Lot A).

    Depuis le Lot A, ce n'est plus `reasoning_budget` qui borne le travail
    total : une étape ayant épuisé ses relances locales avance vers l'étape
    suivante même budget consommé (findings §4). La borne devient donc

        len(plan.steps) × (1 + max_retries)

    — chaque étape est tentée une fois puis relancée au plus `max_retries`
    fois. Elle reste finie, ce qui suffit à garantir la terminaison.

    `max_retries` est lu sur une instance réelle de `Critic` plutôt que
    recopié en dur : la borne ne peut pas dériver silencieusement si la valeur
    par défaut change.
    """
    plan = run.agent_state.plan
    assert plan is not None, run.report()
    max_retries: int = Critic().max_retries
    n_steps: int = len(plan.steps)
    return n_steps * (1 + max_retries)


async def _run_graph(query: str) -> Run:
    """Exécute le graphe réel et capture le chemin, les latences et l'état."""
    run = Run(query)
    graph = build_graph()
    started = time.perf_counter()
    previous = started

    initial: GraphState = build_initial_state(query)
    async for event in graph.astream(initial, stream_mode="updates"):
        now = time.perf_counter()
        for node, update in event.items():
            run.path.append(node)
            run.node_ms.append((node, round((now - previous) * 1000)))
            run.final.update(update)
        previous = now

    run.total_ms = round((time.perf_counter() - started) * 1000)
    return run


# ─────────────────────────────────────────────────────────────────────────────
# Palier 1 — requête classée SIMPLE
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
async def run_simple(
    require_ollama: None, require_action_api: None, real_action_env: None
) -> Run:
    """Exécution unique du palier 1, partagée par les assertions."""
    return await _run_graph(_eval_question(QID_SIMPLE))


@pytest.mark.integration
class TestPalier1Simple:
    """Chemin le plus court : analyze → plan → retrieve → critique → … → END."""

    async def test_pipeline_reaches_the_end(self, run_simple: Run) -> None:
        """Le pipeline traverse tous les nœuds jusqu'à `verify`."""
        assert run_simple.path[0] == "analyze_query", run_simple.report()
        assert run_simple.path[-1] == "verify", run_simple.report()
        for node in ("plan", "retrieve", "critique", "generate_answer"):
            assert node in run_simple.path, run_simple.report()

    async def test_answer_is_not_empty(self, run_simple: Run) -> None:
        """Une réponse est produite, et elle n'est pas vide."""
        answer = run_simple.final.get("answer")
        assert isinstance(answer, str), run_simple.report()
        assert answer.strip() != "", run_simple.report()

    async def test_retrieval_hits_a_gold_article(self, run_simple: Run) -> None:
        """Au moins un chunk provient d'un article gold de la question."""
        gold = _gold_sources(QID_SIMPLE)
        assert gold, "le manifeste ne déclare aucun article gold"
        assert gold & run_simple.sources, (
            f"aucun article gold récupéré. attendus={sorted(gold)}"
            + run_simple.report()
        )

    async def test_budget_is_respected(self, run_simple: Run) -> None:
        """Le compteur global ne dépasse pas le budget + la passe de vérification.

        `feedback_loop_count` est incrémenté par `critique` ET par `verify` ;
        la borne haute est donc budget + 1.
        """
        agent = run_simple.agent_state
        budget = agent.analysis.reasoning_budget
        assert agent.feedback_loop_count <= budget + 1, run_simple.report()

    async def test_verification_is_consistent_with_sources(
        self, run_simple: Run
    ) -> None:
        """Des chunks ont été récupérés : la vérification a pu s'appuyer dessus."""
        verification = run_simple.agent_state.verification
        assert verification is not None, run_simple.report()
        assert run_simple.chunks, run_simple.report()
        assert "no sources available for verification" not in (
            verification.unsupported_claims
        ), run_simple.report()

    async def test_simple_budget_makes_the_loop_unreachable(
        self, run_simple: Run
    ) -> None:
        """CONSTAT (findings §1) : avec un budget SIMPLE de 1, aucune relance.

        `route_after_critique` évalue `feedback_loop_count >= reasoning_budget`
        AVANT le verdict `is_sufficient`. À la première critique le compteur
        vaut déjà 1 : la garde globale l'emporte quel que soit le verdict du
        Critic. La boucle de rétroaction est donc structurellement inatteignable
        pour toute requête classée SIMPLE.
        """
        agent = run_simple.agent_state
        if agent.analysis.query_type is not QueryType.SIMPLE:
            pytest.fail(
                "la question n'a pas été classée SIMPLE — palier à resélectionner"
                + run_simple.report()
            )
        assert run_simple.path.count("retrieve") == 1, run_simple.report()
        assert len(agent.evaluations) == 1, run_simple.report()


# ─────────────────────────────────────────────────────────────────────────────
# Paliers 2 et 3 — requête MULTI_HOP et boucle de rétroaction
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
async def run_multi_hop(
    require_ollama: None, require_action_api: None, real_action_env: None
) -> Run:
    """Exécution unique de la question bridge, partagée paliers 2 et 3."""
    return await _run_graph(_eval_question(QID_MULTI_HOP))


@pytest.mark.integration
class TestPalier2MultiHop:
    """Décomposition en plusieurs étapes avec dépendance."""

    async def test_plan_has_at_least_two_steps(self, run_multi_hop: Run) -> None:
        """Le Planner décompose en au moins deux étapes."""
        plan = run_multi_hop.agent_state.plan
        assert plan is not None, run_multi_hop.report()
        assert len(plan.steps) >= 2, run_multi_hop.report()

    async def test_second_step_declares_its_dependency(
        self, run_multi_hop: Run
    ) -> None:
        """La seconde étape dépend explicitement de la première."""
        steps = run_multi_hop.agent_state.plan.steps
        assert steps[1].depends_on == [steps[0].step_id], run_multi_hop.report()

    async def test_pipeline_reaches_the_end(self, run_multi_hop: Run) -> None:
        """Malgré les relances, l'exécution se termine sur `verify`."""
        assert run_multi_hop.path[-1] == "verify", run_multi_hop.report()

    async def test_second_sub_query_does_not_carry_the_resolved_entity(
        self, run_multi_hop: Run
    ) -> None:
        """CONSTAT (findings §3) : la seconde sous-requête reste générique.

        Le Planner génère TOUTES les sous-requêtes en une passe, AVANT toute
        exécution. La seconde ne peut donc pas contenir l'entité résolue par la
        première — elle la désigne par une périphrase (« the identified
        woman »). Aucun mécanisme de substitution n'existe entre `critique` et
        `retrieve`.

        Ce test échouera — délibérément — le jour où un tel mécanisme sera
        ajouté. Ce sera alors le signal que le constat est levé.
        """
        steps = run_multi_hop.agent_state.plan.steps
        entities = run_multi_hop.agent_state.analysis.detected_entities
        second = steps[1].sub_query.lower()
        resolved_present = [e for e in entities if e and e.lower() in second]
        assert resolved_present == [], (
            "la seconde sous-requête intègre désormais une entité résolue : "
            "le constat des findings §3 est levé, mettre à jour la documentation"
            + run_multi_hop.report()
        )


@pytest.mark.integration
class TestPalier3FeedbackLoop:
    """La boucle Critic → retrieve, sur du vrai retrieval."""

    async def test_loop_actually_executed(self, run_multi_hop: Run) -> None:
        """Le Critic a rejeté et une relance a bien eu lieu."""
        assert run_multi_hop.path.count("retrieve") >= 2, (
            "aucune relance déclenchée — voir findings §2" + run_multi_hop.report()
        )
        evaluations = run_multi_hop.agent_state.evaluations
        assert any(not e.is_sufficient for e in evaluations), run_multi_hop.report()

    async def test_loop_is_bounded_by_plan_length_and_retries(
        self, run_multi_hop: Run
    ) -> None:
        """La boucle s'arrête. Borne : `n_steps × (1 + max_retries)` (Lot A).

        La borne n'est PLUS `reasoning_budget`. Depuis le Lot A, une étape qui
        a épuisé ses relances locales avance vers l'étape suivante même si le
        budget global est consommé — sans quoi une seule étape affamait tout le
        plan (findings §4). Le budget reste prioritaire dans tous les autres
        cas, mais il n'est plus la borne supérieure du travail total.

        La terminaison reste garantie : la branche d'avancement retire une
        étape de la file à chaque déclenchement, et le plan est fini.
        """
        agent = run_multi_hop.agent_state
        bound = _loop_upper_bound(run_multi_hop)

        assert len(agent.evaluations) <= bound, run_multi_hop.report()
        # `feedback_loop_count` compte aussi le passage `verify`, d'où le +1.
        assert agent.feedback_loop_count <= bound + 1, run_multi_hop.report()

    async def test_relaunch_returns_different_chunks(self, run_multi_hop: Run) -> None:
        """CORRECTION §2 : la relance rapporte des chunks différents.

        Avant le Lot A, la sous-requête n'était jamais réécrite entre deux
        tentatives et `CriticEvaluation.feedback` n'était lu par aucun code de
        production : le moteur, déterministe, rapportait exactement les mêmes
        chunks (mesuré : 3 passages × 5 chunks strictement identiques).

        Le nœud `retrieve` enrichit désormais la sous-requête d'une relance
        avec les `missing_aspects` du dernier verdict du Critic.

        L'assertion porte sur « AU MOINS UN passage de relance diffère », et
        non « tous » : quand le Critic ne renvoie aucun terme absent de la
        requête d'origine, l'enrichissement est légitimement sans effet et le
        passage redevient identique — comportement observé et journalisé
        comme « sous-requête INCHANGÉE ».
        """
        n_retrieves = run_multi_hop.path.count("retrieve")
        if n_retrieves < 2:
            pytest.fail(
                "aucune relance : la correction ne peut pas être mesurée"
                + run_multi_hop.report()
            )

        chunks = run_multi_hop.chunks
        assert len(chunks) % n_retrieves == 0, run_multi_hop.report()
        per_pass = len(chunks) // n_retrieves
        passes = [
            [(c.chunk_id, c.source) for c in chunks[i * per_pass : (i + 1) * per_pass]]
            for i in range(n_retrieves)
        ]

        assert any(p != passes[0] for p in passes[1:]), (
            "tous les passages sont identiques : la propagation du feedback "
            "est inopérante, voir findings §2" + run_multi_hop.report()
        )

    async def test_all_planned_steps_are_attempted(self, run_multi_hop: Run) -> None:
        """CORRECTION §4 : chaque étape du plan est effectivement tentée.

        Avant le Lot A, les relances de `step_1` consommaient l'intégralité du
        budget global et `step_2` n'était jamais exécuté — le plan multi-hop
        n'aboutissait donc jamais. La garde locale `max_retries`, pourtant
        présente dans `policy.py`, ne pouvait jamais s'appliquer : la garde
        globale la devançait systématiquement.
        """
        agent = run_multi_hop.agent_state
        evaluated = {e.step_id for e in agent.evaluations}
        planned = {s.step_id for s in agent.plan.steps}

        assert evaluated == planned, (
            f"étapes jamais tentées : {sorted(planned - evaluated)} — la "
            "correction du §4 est inopérante" + run_multi_hop.report()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Palier 4 — cas dégradés
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
async def run_unanswerable(
    require_ollama: None, require_action_api: None, real_action_env: None
) -> Run:
    """Question dont la réponse est absente du corpus."""
    return await _run_graph(UNANSWERABLE_QUERY)


@pytest.mark.integration
class TestPalier4Degraded:
    """Cas dégradés : réponse absente, module ACTION arrêté, budget épuisé."""

    # ── 4.2 — réponse absente du corpus ───────────────────────────────────

    async def test_unanswerable_query_terminates(self, run_unanswerable: Run) -> None:
        """Le pipeline se termine proprement, sans boucle."""
        assert run_unanswerable.path[-1] == "verify", run_unanswerable.report()
        agent = run_unanswerable.agent_state
        assert agent.feedback_loop_count <= agent.analysis.reasoning_budget + 1, (
            run_unanswerable.report()
        )

    async def test_unanswerable_query_does_not_hallucinate(
        self, run_unanswerable: Run
    ) -> None:
        """Le système signale l'incertitude plutôt que d'inventer une réponse.

        L'assertion porte sur ce qui est vérifiable sans dépendre d'une
        formulation : le Verifier ne relève AUCUNE affirmation non étayée. Une
        réponse inventée produirait des `unsupported_claims`.
        """
        verification = run_unanswerable.agent_state.verification
        assert verification is not None, run_unanswerable.report()
        assert verification.unsupported_claims == [], (
            "des affirmations non étayées ont été produites sur une question "
            "sans réponse dans le corpus" + run_unanswerable.report()
        )

    async def test_critic_rejects_off_topic_context(
        self, run_unanswerable: Run
    ) -> None:
        """Le Critic rejette un contexte hors-sujet malgré des scores moteur élevés.

        Observation du palier 5 : le prompt du Critic affiche le
        `relevance_score` du moteur, qui reste haut (~0,86) même hors-sujet —
        c'est un rang normalisé intra-requête. Ce test vérifie que le Critic ne
        s'y laisse pas prendre.
        """
        evaluations = run_unanswerable.agent_state.evaluations
        assert evaluations, run_unanswerable.report()
        assert all(not e.is_sufficient for e in evaluations), (
            "le Critic a jugé suffisant un contexte manifestement hors-sujet"
            + run_unanswerable.report()
        )
        top_engine_score = max(
            (c.relevance_score for c in run_unanswerable.chunks), default=0.0
        )
        assert top_engine_score > 0.5, (
            "le moteur n'a pas produit le score élevé attendu — l'observation "
            "du palier 5 ne peut pas être mesurée ici" + run_unanswerable.report()
        )

    # ── 4.1 — module ACTION arrêté ────────────────────────────────────────

    async def test_action_module_down_degrades_cleanly(
        self, require_ollama: None, real_action_env: None, monkeypatch: Any
    ) -> None:
        """Module ACTION injoignable : repli fail-closed, aucune exception.

        L'indisponibilité est simulée en pointant le client vers un port fermé
        plutôt qu'en arrêtant réellement l'API — le test reste ainsi exécutable
        sans manipulation externe, et n'interfère pas avec les autres paliers
        du module. Le client, le transport HTTP et l'échec de connexion sont
        RÉELS ; seule l'adresse change.

        Le remplacement porte sur la fabrique utilisée par `build_graph`, et
        non sur `reasoning.action_client._ACTION_BASE_URL` : cette globale sert
        de valeur par défaut à `ActionClient.__init__`, évaluée une fois à la
        définition de la fonction. La réassigner n'aurait aucun effet.
        """
        monkeypatch.setattr(
            "reasoning.graph.graph.ActionClient",
            lambda: ActionClient(base_url="http://localhost:9", timeout=2.0),
        )

        run = await _run_graph(_eval_question(QID_MULTI_HOP))

        assert run.path[-1] == "verify", run.report()
        assert run.chunks == [], run.report()

        verification = run.agent_state.verification
        assert verification is not None, run.report()
        assert verification.is_grounded is False, run.report()
        assert verification.faithfulness_score == 0.0, run.report()
        assert "no sources available for verification" in (
            verification.unsupported_claims
        ), run.report()

        # Borne du Lot A : `n_steps × (1 + max_retries)`, et non plus le
        # budget global. Le repli fail-closed n'en est pas affecté — il agit à
        # chaque passage —, mais le nombre de passages a changé : depuis le
        # Lot A, les étapes suivantes du plan sont elles aussi tentées, à vide.
        agent = run.agent_state
        assert agent.feedback_loop_count <= _loop_upper_bound(run) + 1, run.report()

    # ── 4.3 — épuisement du budget global ─────────────────────────────────

    async def test_budget_exhaustion_exits_cleanly(self, run_multi_hop: Run) -> None:
        """Sortie propre par `generate_answer`, jamais de boucle infinie.

        La garde globale de `route_after_critique` reste prioritaire sur le
        verdict du Critic et sur `max_retries` — sauf dans la seule branche
        d'avancement du Lot A, qui consomme une étape du plan et ne peut donc
        pas boucler.

        L'assertion sur le nombre de critiques ne peut plus être une égalité
        avec `reasoning_budget` : ce budget n'est plus la borne du travail
        total. Ce qui est vérifié ici est ce qui compte réellement — le nombre
        de critiques reste sous une borne FINIE, et la sortie se fait bien par
        `generate_answer` puis `verify`.
        """
        agent = run_multi_hop.agent_state

        # Borne FINIE respectée — la seule propriété qui garantit la
        # terminaison. Volontairement pas d'égalité avec `reasoning_budget` :
        # selon les verdicts du Critic, l'exécution peut s'arrêter avant de
        # l'atteindre comme le dépasser, et figer l'un ou l'autre rendrait ce
        # test tributaire de la variabilité du LLM.
        assert len(agent.evaluations) <= _loop_upper_bound(run_multi_hop), (
            run_multi_hop.report()
        )
        assert agent.feedback_loop_count <= _loop_upper_bound(run_multi_hop) + 1, (
            run_multi_hop.report()
        )
        assert run_multi_hop.path[-2] == "generate_answer", run_multi_hop.report()
        assert run_multi_hop.path[-1] == "verify", run_multi_hop.report()
