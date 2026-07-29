"""
Tests unitaires — toon_utils.py (couverture 100%).

Objectif : exercer toutes les branches de parse_toon_to_dict,
dump_dict_to_toon, _extract_toon_block et _infer_value afin
d'atteindre une couverture de 100 % sur le module.

Exécution :
    uv run pytest tests/unit/test_toon_utils.py -v --cov=src/reasoning/shared/toon_utils
"""

from __future__ import annotations

from typing import Any

import pytest
from reasoning.shared.toon_utils import (
    ToonParseError,
    _extract_toon_block,
    _infer_value,
    dump_dict_to_toon,
    parse_toon_to_dict,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tests — ToonParseError
# ─────────────────────────────────────────────────────────────────────────────


class TestToonParseError:
    """Vérifie la structure et le comportement de l'exception personnalisée."""

    def test_attributes_are_stored(self) -> None:
        """Les attributs `reason` et `raw` sont accessibles après construction."""
        exc = ToonParseError(reason="raison de test", raw="entrée brute")
        assert exc.reason == "raison de test"
        assert exc.raw == "entrée brute"

    def test_default_raw_is_empty_string(self) -> None:
        """`raw` vaut '' par défaut si non fourni."""
        exc = ToonParseError(reason="sans raw")
        assert exc.raw == ""

    def test_str_representation_contains_reason(self) -> None:
        """La représentation str contient la raison de l'erreur."""
        exc = ToonParseError(reason="motif explicite", raw="données")
        assert "motif explicite" in str(exc)

    def test_is_subclass_of_value_error(self) -> None:
        """`ToonParseError` est un `ValueError` pour compatibilité with existing except blocks."""
        with pytest.raises(ValueError):
            raise ToonParseError(reason="test héritage")

    def test_long_raw_is_truncated_in_str(self) -> None:
        """Une chaîne `raw` très longue est tronquée dans le message (limite 120 chars)."""
        long_raw = "x" * 300
        exc = ToonParseError(reason="trop long", raw=long_raw)
        # Le message ne doit pas inclure les 300 caractères complets
        assert len(str(exc)) < 300


# ─────────────────────────────────────────────────────────────────────────────
# Tests — _extract_toon_block (fonction interne)
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractToonBlock:
    """Tests de la fonction interne d'extraction de bloc TOON."""

    def test_cas1_delimiteurs_officiels(self) -> None:
        """Cas 1 : délimiteurs <<< >>> — le contenu interne est retourné."""
        raw = "<<<\nquery_type :: SIMPLE\n>>>"
        result = _extract_toon_block(raw)
        assert "query_type :: SIMPLE" in result
        # Les délimiteurs eux-mêmes ne doivent pas être dans le contenu interne
        assert "<<<" not in result
        assert ">>>" not in result

    def test_cas1_delimiteurs_avec_texte_parasite(self) -> None:
        """Cas 1 : le bloc <<< >>> est correctement extrait même entouré de texte."""
        raw = "Voici la réponse :\n<<<\nconfidence :: 0.9\n>>>\nFin."
        result = _extract_toon_block(raw)
        assert "confidence :: 0.9" in result

    def test_cas2_bloc_markdown_toon(self) -> None:
        """Cas 2 : bloc ```toon ... ``` contenant des paires clé :: valeur."""
        raw = "```toon\nquery_type :: MULTI_HOP\nreasoning_budget :: 3\n```"
        result = _extract_toon_block(raw)
        assert "query_type :: MULTI_HOP" in result

    def test_cas2_bloc_markdown_generique(self) -> None:
        """Cas 2 : bloc ``` ... ``` générique (sans label 'toon') avec paires ::."""
        raw = "```\nconfidence :: 0.85\nreasoning_budget :: 2\n```"
        result = _extract_toon_block(raw)
        assert "confidence :: 0.85" in result

    def test_cas2_bloc_markdown_sans_paires_ignore(self) -> None:
        """Cas 2 : un bloc Markdown sans '::' est ignoré, fallback sur Cas 3."""
        raw = "```\nPas de paires ici\n```\nquery_type :: SIMPLE"
        # Le bloc markdown n'a pas de '::' → on tombe sur le mode dégradé (Cas 3)
        result = _extract_toon_block(raw)
        assert "query_type :: SIMPLE" in result

    def test_cas3_mode_degrade_sans_delimiteurs(self) -> None:
        """Cas 3 : lignes clé :: valeur sans délimiteurs — mode dégradé."""
        raw = "query_type :: AMBIGUOUS\nreasoning_budget :: 0"
        result = _extract_toon_block(raw)
        assert "query_type :: AMBIGUOUS" in result
        assert "reasoning_budget :: 0" in result

    def test_cas3_mode_degrade_ignore_lignes_sans_double_colon(self) -> None:
        """Cas 3 : seules les lignes contenant '::' sont collectées."""
        raw = "Ligne normale\nkey :: value\nAutre ligne"
        result = _extract_toon_block(raw)
        assert "key :: value" in result
        assert "Ligne normale" not in result

    def test_leve_toon_parse_error_si_aucun_contenu(self) -> None:
        """ToonParseError si aucun délimiteur, bloc Markdown ni paire clé :: valeur."""
        with pytest.raises(ToonParseError) as exc_info:
            _extract_toon_block("Aucune structure TOON ici.")
        assert "aucun bloc TOON" in str(exc_info.value)

    def test_leve_toon_parse_error_bloc_markdown_sans_paires_et_rien_d_autre(
        self,
    ) -> None:
        """ToonParseError si le seul contenu est un bloc Markdown sans '::' et pas
        de lignes libres avec '::'."""
        with pytest.raises(ToonParseError):
            _extract_toon_block("```\ncontenu sans separateur\n```")


# ─────────────────────────────────────────────────────────────────────────────
# Tests — _infer_value (fonction interne)
# ─────────────────────────────────────────────────────────────────────────────


class TestInferValue:
    """Tests exhaustifs de l'inférence de type à partir d'une valeur brute TOON."""

    def test_valeur_vide_retourne_none(self) -> None:
        """Chaîne vide (après '::') → None Python."""
        assert _infer_value("") is None
        assert _infer_value("   ") is None

    def test_liste_multi_elements(self) -> None:
        """Valeur contenant '|' → liste de str strippées."""
        result = _infer_value("LangChain | LangSmith")
        assert result == ["LangChain", "LangSmith"]

    def test_liste_mono_element_avec_pipe_final(self) -> None:
        """Convention mono-élément : 'valeur |' → liste à un seul élément."""
        result = _infer_value("rétropropagation |")
        assert result == ["rétropropagation"]

    def test_liste_filtre_elements_vides(self) -> None:
        """Les fragments vides autour des '|' sont filtrés."""
        result = _infer_value(" A |  | B ")
        assert "" not in result
        assert "A" in result
        assert "B" in result

    def test_entier_positif(self) -> None:
        """Chaîne représentant un entier positif → int."""
        assert _infer_value("3") == 3
        assert isinstance(_infer_value("3"), int)

    def test_entier_negatif(self) -> None:
        """Chaîne représentant un entier négatif → int négatif."""
        assert _infer_value("-1") == -1
        assert isinstance(_infer_value("-1"), int)

    def test_entier_zero(self) -> None:
        """'0' → int 0 (budget AMBIGUOUS)."""
        result = _infer_value("0")
        assert result == 0
        assert isinstance(result, int)

    def test_flottant_standard(self) -> None:
        """Chaîne représentant un float → float Python."""
        result = _infer_value("0.92")
        assert result == pytest.approx(0.92)
        assert isinstance(result, float)

    def test_flottant_negatif(self) -> None:
        """Float négatif → float négatif."""
        result = _infer_value("-1.5")
        assert result == pytest.approx(-1.5)

    def test_flottant_sans_zero_initial(self) -> None:
        """'.5' (sans zéro initial) → float 0.5."""
        result = _infer_value(".5")
        assert result == pytest.approx(0.5)

    def test_chaine_brute(self) -> None:
        """Valeur ni vide, ni liste, ni numérique → str strippée."""
        assert _infer_value("MULTI_HOP") == "MULTI_HOP"
        assert _infer_value("  SIMPLE  ") == "SIMPLE"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — parse_toon_to_dict
# ─────────────────────────────────────────────────────────────────────────────


class TestParseToonToDict:
    """Tests de l'API publique de désérialisation TOON."""

    def test_bloc_complet_avec_delimiteurs(self) -> None:
        """Parse nominal d'un bloc TOON bien formé avec délimiteurs."""
        raw = (
            "<<<\n"
            "query_type :: MULTI_HOP\n"
            "confidence :: 0.92\n"
            "detected_entities :: LangChain | LangSmith\n"
            "reasoning_budget :: 3\n"
            ">>>"
        )
        result = parse_toon_to_dict(raw)
        assert result["query_type"] == "MULTI_HOP"
        assert result["confidence"] == pytest.approx(0.92)
        assert result["detected_entities"] == ["LangChain", "LangSmith"]
        assert result["reasoning_budget"] == 3

    def test_bloc_markdown_toon(self) -> None:
        """Parse d'un bloc encadré par ```toon ... ```."""
        raw = "```toon\nquery_type :: SIMPLE\nreasoning_budget :: 1\n```"
        result = parse_toon_to_dict(raw)
        assert result["query_type"] == "SIMPLE"
        assert result["reasoning_budget"] == 1

    def test_mode_degrade(self) -> None:
        """Parse en mode dégradé : lignes clé :: valeur sans délimiteurs."""
        raw = "query_type :: AMBIGUOUS\nreasoning_budget :: 0"
        result = parse_toon_to_dict(raw)
        assert result["query_type"] == "AMBIGUOUS"
        assert result["reasoning_budget"] == 0

    def test_champ_none(self) -> None:
        """Valeur vide après '::' → None dans le dictionnaire."""
        raw = "<<<\nfilters ::\n>>>"
        result = parse_toon_to_dict(raw)
        assert result["filters"] is None

    def test_liste_mono_element(self) -> None:
        """Une entité seule avec '|' final → liste à un élément."""
        raw = "<<<\ndetected_entities :: OpenAI |\n>>>"
        result = parse_toon_to_dict(raw)
        assert result["detected_entities"] == ["OpenAI"]

    def test_lignes_vides_ignorees(self) -> None:
        """Les lignes vides à l'intérieur du bloc sont silencieusement ignorées."""
        raw = "<<<\n\nquery_type :: SIMPLE\n\nreasoning_budget :: 1\n\n>>>"
        result = parse_toon_to_dict(raw)
        assert len(result) == 2

    def test_ligne_malformee_sans_cle_ignoree(self) -> None:
        """Ligne ':: valeur' (clé vide) est ignorée sans lever d'erreur."""
        raw = "<<<\n:: valeur_orpheline\nquery_type :: SIMPLE\n>>>"
        result = parse_toon_to_dict(raw)
        assert "query_type" in result
        assert "" not in result  # la clé vide ne doit pas apparaître

    def test_partition_sur_premier_double_colon_uniquement(self) -> None:
        """Si la valeur contient '::' elle-même, seul le premier sépare clé/valeur."""
        raw = "<<<\ndescription :: valeur :: avec :: colons\n>>>"
        result = parse_toon_to_dict(raw)
        assert result["description"] == "valeur :: avec :: colons"

    def test_leve_toon_parse_error_sur_chaine_vide(self) -> None:
        """ToonParseError si la chaîne d'entrée est vide."""
        with pytest.raises(ToonParseError) as exc_info:
            parse_toon_to_dict("")
        assert "vide" in str(exc_info.value)

    def test_leve_toon_parse_error_sur_espaces_uniquement(self) -> None:
        """ToonParseError si la chaîne ne contient que des espaces."""
        with pytest.raises(ToonParseError):
            parse_toon_to_dict("   ")

    def test_leve_toon_parse_error_si_bloc_extrait_sans_paires(self) -> None:
        """ToonParseError si _extract_toon_block réussit mais que aucune paire n'est parsée.

        Cas : le bloc <<< >>> existe mais son contenu est constitué uniquement
        de lignes sans '::' (ex. bloc vide ou entièrement commenté).
        """
        raw = "<<<\nligne sans separateur\nune autre sans\n>>>"
        with pytest.raises(ToonParseError) as exc_info:
            parse_toon_to_dict(raw)
        assert "aucune paire" in str(exc_info.value)

    def test_leve_toon_parse_error_si_aucun_contenu_toon(self) -> None:
        """ToonParseError propagée depuis _extract_toon_block si aucun contenu TOON."""
        with pytest.raises(ToonParseError):
            parse_toon_to_dict("Texte libre sans structure TOON.")

    def test_retourne_dict_any(self) -> None:
        """Le type de retour est toujours un dict."""
        raw = "<<<\nkey :: value\n>>>"
        result = parse_toon_to_dict(raw)
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — dump_dict_to_toon
# ─────────────────────────────────────────────────────────────────────────────


class TestDumpDictToToon:
    """Tests de l'API publique de sérialisation vers le format TOON."""

    def test_structure_de_base(self) -> None:
        """La sortie est bien encadrée par <<< et >>> avec une ligne par clé."""
        result = dump_dict_to_toon({"key": "value"})
        assert result.startswith("<<<")
        assert result.endswith(">>>")
        assert "key :: value" in result

    def test_serialise_entier(self) -> None:
        """Un entier est sérialisé sans décimale."""
        result = dump_dict_to_toon({"budget": 3})
        assert "budget :: 3" in result

    def test_serialise_float(self) -> None:
        """Un float est sérialisé en notation compacte (format :g)."""
        result = dump_dict_to_toon({"score": 0.92})
        assert "score :: 0.92" in result

    def test_serialise_none(self) -> None:
        """None produit une valeur vide après '::'."""
        result = dump_dict_to_toon({"filters": None})
        assert "filters :: " in result

    def test_serialise_liste_multi_elements(self) -> None:
        """Une liste de plusieurs éléments est jointe avec ' | '."""
        result = dump_dict_to_toon({"entities": ["A", "B", "C"]})
        assert "entities :: A | B | C" in result

    def test_serialise_liste_mono_element(self) -> None:
        """Une liste d'un seul élément reçoit un '|' final pour rester une liste au parse."""
        result = dump_dict_to_toon({"entities": ["seul"]})
        assert "entities :: seul |" in result

    def test_serialise_liste_vide(self) -> None:
        """Une liste vide produit une valeur vide après '::'."""
        result = dump_dict_to_toon({"entities": []})
        assert "entities :: " in result

    def test_serialise_bool(self) -> None:
        """Un booléen est sérialisé via str() → 'True' ou 'False'."""
        result = dump_dict_to_toon({"flag": True})
        assert "flag :: True" in result

    def test_serialise_dict_vide(self) -> None:
        """Un dict vide produit un bloc TOON avec uniquement <<< et >>>."""
        result = dump_dict_to_toon({})
        lines = result.strip().splitlines()
        assert lines[0] == "<<<"
        assert lines[-1] == ">>>"
        assert len(lines) == 2

    def test_leve_type_error_sur_non_dict(self) -> None:
        """TypeError si l'argument n'est pas un dict."""
        bad_input: Any = "pas un dict"
        with pytest.raises(TypeError) as exc_info:
            dump_dict_to_toon(bad_input)
        assert "dict" in str(exc_info.value)

    def test_leve_type_error_sur_liste(self) -> None:
        """TypeError si on passe une liste au lieu d'un dict."""
        bad_input: Any = ["a", "b"]
        with pytest.raises(TypeError):
            dump_dict_to_toon(bad_input)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Roundtrip (intégrité parse ↔ dump)
# ─────────────────────────────────────────────────────────────────────────────


class TestRoundtrip:
    """Vérifie l'intégrité complète : dump → parse doit restituer le dict original."""

    def test_roundtrip_dict_complet(self) -> None:
        """Roundtrip complet avec tous les types de valeurs supportés."""
        original: dict[str, Any] = {
            "query_type": "MULTI_HOP",
            "confidence": 0.92,
            "detected_entities": ["LangChain", "LangSmith"],
            "reasoning_budget": 3,
            "filters": None,
        }
        toon_str = dump_dict_to_toon(original)
        recovered = parse_toon_to_dict(toon_str)
        assert recovered == original

    def test_roundtrip_liste_mono_element(self) -> None:
        """Roundtrip avec une liste d'un seul élément : l'élément doit rester une liste."""
        original: dict[str, Any] = {
            "detected_entities": ["OpenAI"],
            "reasoning_budget": 1,
        }
        recovered = parse_toon_to_dict(dump_dict_to_toon(original))
        assert recovered["detected_entities"] == ["OpenAI"]
        assert isinstance(recovered["detected_entities"], list)

    def test_roundtrip_valeurs_none(self) -> None:
        """Roundtrip avec des valeurs None : elles doivent rester None."""
        original: dict[str, Any] = {"filters": None, "metadata": None}
        recovered = parse_toon_to_dict(dump_dict_to_toon(original))
        assert recovered["filters"] is None
        assert recovered["metadata"] is None

    def test_roundtrip_entier_zero(self) -> None:
        """Roundtrip du budget 0 (AMBIGUOUS) : doit rester int 0."""
        original: dict[str, Any] = {"reasoning_budget": 0}
        recovered = parse_toon_to_dict(dump_dict_to_toon(original))
        assert recovered["reasoning_budget"] == 0
        assert isinstance(recovered["reasoning_budget"], int)

    def test_roundtrip_float_precision(self) -> None:
        """Roundtrip de floats courants : la précision doit être conservée."""
        for val in [0.55, 0.92, 0.97, 1.0]:
            original: dict[str, Any] = {"confidence": val}
            recovered = parse_toon_to_dict(dump_dict_to_toon(original))
            assert recovered["confidence"] == pytest.approx(val)

    def test_roundtrip_dict_vide_leve_toon_parse_error(self) -> None:
        """Dump d'un dict vide puis parse → ToonParseError car aucune paire clé/valeur."""
        toon_str = dump_dict_to_toon({})
        # Le bloc <<< >>> est présent mais vide → aucune paire parseable
        with pytest.raises(ToonParseError):
            parse_toon_to_dict(toon_str)
