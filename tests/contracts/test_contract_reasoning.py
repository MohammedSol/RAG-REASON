"""
Tests de contrat — Sprint I3, côté CONSOMMATEUR (module REASONING).

Vérifient que le module REASONING sait produire et consommer exactement les
documents JSON de `tests/contracts/fixtures/`, qui sont les mêmes fichiers que
ceux chargés par le module ACTION (voir `tests/contracts/README.md`).

Aucun appel réseau : `httpx.post` est mocké. L'aller-retour réel est traité
séparément (Sprint I3, étape 5).

Exécution :
    uv run pytest tests/contracts/ -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reasoning.action_client import ActionClient
from reasoning.contracts.action_interface import RetrievalRequest, RetrievalResponse
from reasoning.contracts.internal_models import AgentState, ExecutionPlan, PlanStep
from reasoning.graph.nodes import make_retrieve_node
from reasoning.graph.state import GraphState

FIXTURES_DIR = Path(__file__).parent / "fixtures"

REQUEST_FIXTURES = (
    "retrieval_request_nominal.json",
    "retrieval_request_with_filters.json",
    "retrieval_request_top_k_high.json",
)
RESPONSE_FIXTURES = (
    "retrieval_response_nominal.json",
    "retrieval_response_empty.json",
)

# Plafond du moteur `fusion_search` du module ACTION, relevé au Sprint I3 :
# `FusionSearch.search(query, top_k=5)` et `FusionTool.execute` ne passe pas
# de `top_k`. Constaté, non imposé — ce test documente l'écart, il ne le
# corrige pas (docs/integration_contract_findings.md §1).
ENGINE_MAX_RESULTS_OBSERVED = 5


def load_fixture(name: str) -> dict[str, Any]:
    """Charge une fixture JSON partagée.

    Args:
        name: Nom du fichier dans `tests/contracts/fixtures/`.

    Returns:
        Le document JSON désérialisé.
    """
    path = FIXTURES_DIR / name
    assert path.is_file(), f"fixture partagée absente : {path}"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _http_response(payload: dict[str, Any]) -> MagicMock:
    """Fausse réponse httpx 200 portant `payload`."""
    response = MagicMock()
    response.status_code = 200
    response.is_success = True
    response.text = json.dumps(payload)
    response.json.return_value = payload
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Validité des fixtures au regard du contrat figé
# ─────────────────────────────────────────────────────────────────────────────


class TestFixturesAreContractValid:
    """Les 5 fixtures sont valides pour la copie REASONING du contrat."""

    @pytest.mark.parametrize("name", REQUEST_FIXTURES)
    def test_request_fixture_validates(self, name: str) -> None:
        """Chaque fixture de requête est un RetrievalRequest valide."""
        request = RetrievalRequest.model_validate(load_fixture(name))

        assert request.query_id == "plan-derrickson-nationality"
        assert request.sub_query.strip() != ""
        assert request.hop_index >= 0
        assert request.top_k > 0

    @pytest.mark.parametrize("name", RESPONSE_FIXTURES)
    def test_response_fixture_validates(self, name: str) -> None:
        """Chaque fixture de réponse est un RetrievalResponse valide."""
        response = RetrievalResponse.model_validate(load_fixture(name))

        assert response.query_id == "plan-derrickson-nationality"
        assert isinstance(response.chunks, list)

    def test_request_fixtures_round_trip_unchanged(self) -> None:
        """model_dump() reproduit la fixture à l'identique — aucun champ perdu.

        Contrôle du sens inverse : si le contrat gagnait un champ, ou en
        renommait un, la sérialisation cesserait de correspondre au document
        partagé et ce test le signalerait.
        """
        for name in REQUEST_FIXTURES:
            raw = load_fixture(name)
            assert RetrievalRequest.model_validate(raw).model_dump() == raw, name

    def test_response_fixtures_round_trip_unchanged(self) -> None:
        """Même contrôle pour les réponses."""
        for name in RESPONSE_FIXTURES:
            raw = load_fixture(name)
            assert RetrievalResponse.model_validate(raw).model_dump() == raw, name


# ─────────────────────────────────────────────────────────────────────────────
# Le plan du Planner alimente une requête conforme
# ─────────────────────────────────────────────────────────────────────────────


class TestPlanFeedsConformingRequest:
    """Un ExecutionPlan produit une RetrievalRequest de la forme des fixtures.

    Le test exerce la vraie fabrique `make_retrieve_node` — c'est le code de
    production qui construit la requête, jamais une réimplémentation dans le
    test. Le client est un espion : il capture la requête émise et retourne la
    fixture de réponse nominale.
    """

    @staticmethod
    def _plan() -> ExecutionPlan:
        """Plan à deux étapes, de la forme produite par le Planner."""
        return ExecutionPlan(
            plan_id="plan-derrickson-nationality",
            original_query=(
                "Were Scott Derrickson and Ed Wood of the same nationality?"
            ),
            steps=[
                PlanStep(
                    step_id="step_1", sub_query="What nationality is Scott Derrickson?"
                ),
                PlanStep(
                    step_id="step_2",
                    sub_query="What nationality is Ed Wood?",
                    depends_on=["step_1"],
                ),
            ],
            dependencies_graph={"step_1": [], "step_2": ["step_1"]},
        )

    @staticmethod
    def _state(plan: ExecutionPlan) -> GraphState:
        """GraphState complet, prêt pour l'exécution du nœud `retrieve`.

        Toutes les clés du TypedDict sont fournies — un état partiel passerait
        à l'exécution mais masquerait une évolution du contrat d'état.
        """
        return GraphState(
            agent_state=AgentState(original_query=plan.original_query, plan=plan),
            retrieved_chunks=[],
            retry_counts={},
            pending_step_ids=["step_1"],
            current_step_id=None,
            last_retrieval_response=None,
            answer=None,
            next_route="",
        )

    async def test_emitted_request_matches_fixture_shape(self) -> None:
        """La requête émise a exactement les champs de la fixture nominale."""
        captured: list[RetrievalRequest] = []
        fixture_response = load_fixture("retrieval_response_nominal.json")

        class SpyClient:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
                captured.append(request)
                return RetrievalResponse.model_validate(fixture_response)

        plan = self._plan()
        node = make_retrieve_node(SpyClient())
        await node(self._state(plan))

        assert len(captured) == 1
        emitted = captured[0].model_dump()
        expected_shape = load_fixture("retrieval_request_nominal.json")

        # Mêmes clés, mêmes types — c'est la conformité structurelle exigée.
        assert emitted.keys() == expected_shape.keys()
        for key, reference in expected_shape.items():
            assert isinstance(emitted[key], type(reference)) or reference is None, key

        # Et les valeurs qui doivent provenir du plan en proviennent réellement.
        assert emitted["query_id"] == plan.plan_id
        assert emitted["sub_query"] == plan.steps[0].sub_query
        assert emitted["hop_index"] == 0

    async def test_emitted_request_is_contract_valid(self) -> None:
        """La requête émise par le nœud est elle-même un RetrievalRequest valide."""
        captured: list[RetrievalRequest] = []
        fixture_response = load_fixture("retrieval_response_nominal.json")

        class SpyClient:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
                captured.append(request)
                return RetrievalResponse.model_validate(fixture_response)

        node = make_retrieve_node(SpyClient())
        await node(self._state(self._plan()))

        revalidated = RetrievalRequest.model_validate(captured[0].model_dump())
        assert revalidated == captured[0]

    async def test_emitted_top_k_versus_engine_ceiling(self) -> None:
        """Constat, non correction : `top_k` émis face au plafond du moteur.

        Le `top_k` ne vient PAS du PlanStep — `PlanStep` n'a pas ce champ. Il
        vient du paramètre par défaut de `make_retrieve_node`
        (`_DEFAULT_TOP_K`, src/reasoning/graph/nodes.py). Ce test fige la
        valeur effectivement émise et la compare au plafond relevé côté
        fournisseur.
        """
        captured: list[RetrievalRequest] = []
        fixture_response = load_fixture("retrieval_response_nominal.json")

        class SpyClient:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
                captured.append(request)
                return RetrievalResponse.model_validate(fixture_response)

        node = make_retrieve_node(SpyClient())
        await node(self._state(self._plan()))

        assert captured[0].top_k == 5
        assert captured[0].top_k <= ENGINE_MAX_RESULTS_OBSERVED, (
            "Le top_k demandé dépasse le plafond du moteur ACTION : "
            "voir docs/integration_contract_findings.md §1."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ActionClient désérialise les fixtures de réponse
# ─────────────────────────────────────────────────────────────────────────────


class TestClientDeserializesFixtures:
    """`ActionClient` accepte chaque fixture de réponse partagée."""

    @pytest.mark.parametrize("name", RESPONSE_FIXTURES)
    @patch("reasoning.action_client.httpx.post")
    def test_fixture_is_deserialized(self, mock_post: MagicMock, name: str) -> None:
        """La fixture traverse le client et ressort en RetrievalResponse."""
        payload = load_fixture(name)
        mock_post.return_value = _http_response(payload)

        request = RetrievalRequest.model_validate(
            load_fixture("retrieval_request_nominal.json")
        )
        response = ActionClient().retrieve(request)

        assert isinstance(response, RetrievalResponse)
        assert response.query_id == request.query_id
        assert len(response.chunks) == len(payload["chunks"])

    @patch("reasoning.action_client.httpx.post")
    def test_nominal_fixture_values_survive_deserialization(
        self, mock_post: MagicMock
    ) -> None:
        """Les valeurs des chunks sont préservées, champ par champ."""
        payload = load_fixture("retrieval_response_nominal.json")
        mock_post.return_value = _http_response(payload)

        request = RetrievalRequest.model_validate(
            load_fixture("retrieval_request_nominal.json")
        )
        response = ActionClient().retrieve(request)

        assert [c.chunk_id for c in response.chunks] == ["2557", "987", "986"]
        assert [c.source for c in response.chunks] == [
            "Scott_Derrickson.txt",
            "Ed_Wood_(film).txt",
            "Ed_Wood.txt",
        ]
        assert response.chunks[0].relevance_score == pytest.approx(0.95955)
        assert response.retrieval_score == pytest.approx(0.9266)

    @patch("reasoning.action_client.httpx.post")
    def test_empty_response_is_a_valid_business_case(
        self, mock_post: MagicMock
    ) -> None:
        """Aucun chunk trouvé : accepté sans erreur, ce n'est pas une panne.

        Confondre « rien trouvé » et « le service est tombé » rendrait toute
        mesure de qualité du retrieval ininterprétable — c'est précisément la
        distinction que les exceptions typées du Sprint I2 préservent.
        """
        payload = load_fixture("retrieval_response_empty.json")
        mock_post.return_value = _http_response(payload)

        request = RetrievalRequest.model_validate(
            load_fixture("retrieval_request_nominal.json")
        )
        response = ActionClient().retrieve(request)

        assert response.chunks == []
        assert response.retrieval_score is None
        assert response.query_id == request.query_id

    async def test_empty_response_traverses_the_retrieve_node(self) -> None:
        """Le nœud `retrieve` propage une réponse vide sans lever ni journaliser.

        Contrôle complémentaire : le repli fail-closed du nœud produit aussi une
        réponse vide. Il faut donc s'assurer qu'une réponse vide *légitime* n'est
        pas confondue avec ce repli — ici le client n'a levé aucune exception.
        """
        payload = load_fixture("retrieval_response_empty.json")

        class EmptyClient:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
                return RetrievalResponse.model_validate(payload)

        plan = TestPlanFeedsConformingRequest._plan()
        node = make_retrieve_node(EmptyClient())
        result = await node(TestPlanFeedsConformingRequest._state(plan))

        assert result["retrieved_chunks"] == []
        assert result["last_retrieval_response"].query_id == plan.plan_id
        assert result["current_step_id"] == "step_1"


# ─────────────────────────────────────────────────────────────────────────────
# Garde-fou : les deux copies du contrat ne divergent pas silencieusement
# ─────────────────────────────────────────────────────────────────────────────


class TestGraphStateImportIsStable:
    """Vérifie que le type GraphState reste importable et nommé tel quel.

    Sans cette assertion, l'import de `GraphState` en tête de module ne serait
    exercé par aucun test et une suppression passerait inaperçue jusqu'au
    prochain lancement du graphe.
    """

    def test_graph_state_declares_expected_keys(self) -> None:
        """GraphState expose les clés lues par le nœud retrieve."""
        keys = set(GraphState.__annotations__)
        assert {"agent_state", "pending_step_ids", "retrieved_chunks"} <= keys
