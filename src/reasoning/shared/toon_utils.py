"""
Utilitaire partagé de sérialisation TOON (Token-Oriented Object Notation).

Ce module est le seul point de vérité pour la conversion entre le format TOON
(utilisé dans les communications LLM et inter-modules) et les structures de
données Python natives.

Responsabilités :
    - Parser une chaîne TOON brute (sortie LLM) en ``dict[str, Any]``
    - Sérialiser un ``dict[str, Any]`` (ou ``.model_dump()`` Pydantic) en TOON
    - Parser un bloc TOON multi-enregistrements en ``list[dict[str, Any]]``

Contraintes de conception :
    - Aucune dépendance extérieure : bibliothèque standard Python uniquement
    - Agnostique au schéma : aucune connaissance des modèles Pydantic spécifiques
    - Partageable tel quel avec le module ACTION sans modification

Syntaxe TOON v1.0 :
    - Délimiteurs de bloc  : ``<<<`` (ouverture) et ``>>>`` (fermeture)
    - Séparateur clé/valeur: ``::``
    - Séparateur de liste  : ``|``
    - Valeur None          : chaîne vide après ``::``
    - Une paire clé/valeur par ligne

Exemple de bloc valide :
    <<<
    query_type :: MULTI_HOP
    confidence :: 0.92
    detected_entities :: LangChain | LangSmith
    reasoning_budget :: 3
    >>>
"""

from __future__ import annotations

import re
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Exception personnalisée
# ─────────────────────────────────────────────────────────────────────────────


class ToonParseError(ValueError):
    """Levée quand une chaîne TOON ne peut pas être parsée en dictionnaire.

    Hérite de ``ValueError`` pour être compatible avec les blocs ``except``
    existants dans ``analyzer.py`` sans modification supplémentaire.

    Attributes:
        raw: La chaîne d'entrée à l'origine de l'erreur.
        reason: Description textuelle de la cause de l'échec.
    """

    def __init__(self, reason: str, raw: str = "") -> None:
        self.raw = raw
        self.reason = reason
        super().__init__(f"ToonParseError — {reason} | input={raw!r:.120}")


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions internes
# ─────────────────────────────────────────────────────────────────────────────


def _extract_toon_block(raw: str) -> str:
    """Extrait le contenu brut d'un bloc TOON délimité par ``<<<`` et ``>>>``.

    Si les délimiteurs sont absents, tente un mode dégradé en collectant
    toutes les lignes contenant ``::`` (réponse LLM sans encadrement).

    Args:
        raw: La chaîne brute retournée par le LLM, éventuellement
             entourée de texte parasite ou de blocs Markdown.

    Returns:
        Le contenu interne du bloc TOON, sans les délimiteurs ``<<<``/``>>>``.
        En mode dégradé, retourne les lignes ``clé :: valeur`` jointes par
        ``\\n``.

    Raises:
        ToonParseError: Si aucun bloc ni aucune ligne ``clé :: valeur``
            n'est détectable dans la chaîne.
    """
    raw = raw.strip()

    # ── Cas 1 : délimiteurs officiels <<< ... >>> présents ────────────────────
    match = re.search(r"<<<(.*?)>>>", raw, re.DOTALL)
    if match:
        return match.group(1)

    # ── Cas 2 : bloc Markdown ```toon ... ``` ou ``` ... ``` ─────────────────
    match = re.search(r"```(?:toon)?\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        candidate = match.group(1)
        if "::" in candidate:
            return candidate

    # ── Cas 3 : mode dégradé — lignes «clé :: valeur» dans le texte libre ────
    kv_lines = [line for line in raw.splitlines() if "::" in line]
    if kv_lines:
        return "\n".join(kv_lines)

    raise ToonParseError(
        reason=(
            "aucun bloc TOON (<<<>>>), bloc Markdown "
            "ni ligne 'clé :: valeur' détectable"
        ),
        raw=raw,
    )


def _infer_value(raw_value: str) -> Any:
    """Infère le type Python d'une valeur TOON brute.

    Ordre d'inférence (prioritaire → dernier recours) :
    1. Valeur vide → ``None``
    2. Présence de ``|`` → liste de chaînes strippées
    3. Entier (possiblement négatif) → ``int``
    4. Flottant → ``float``
    5. Par défaut → ``str`` brute strippée

    Args:
        raw_value: La valeur brute extraite après le ``::`` de la ligne TOON.

    Returns:
        La valeur Python typée.
    """
    stripped = raw_value.strip()

    # None
    if stripped == "":
        return None

    # Liste (séparateur |)
    if "|" in stripped:
        return [item.strip() for item in stripped.split("|") if item.strip()]

    # Entier (ex : "3", "-1")
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)

    # Flottant (ex : "0.92", "-1.5", ".5")
    if re.fullmatch(r"-?\d*\.\d+", stripped):
        return float(stripped)

    # Chaîne brute
    return stripped


def _parse_block_lines(content: str) -> dict[str, Any]:
    """Parse le contenu interne d'un bloc TOON en dictionnaire Python.

    Traite les lignes de la forme ``clé :: valeur`` en ignorant les lignes
    vides et les lignes sans séparateur ``::``.

    Cette fonction est une extraction du cœur de parsing de
    ``parse_toon_to_dict()`` ; elle est également utilisée par
    ``parse_toon_records()`` pour parser chaque enregistrement individuellement.
    Son comportement est identique à la boucle inline qu'elle remplace.

    Args:
        content: Le contenu brut d'un bloc TOON, sans les délimiteurs
            ``<<<``/``>>>``. Peut contenir des lignes vides.

    Returns:
        Un dictionnaire ``dict[str, Any]`` des paires clé/valeur parsées.
        Retourne un dict vide si aucune paire n'est trouvée.
    """
    result: dict[str, Any] = {}

    for line in content.splitlines():
        line = line.strip()

        # Ignorer les lignes vides et les lignes sans séparateur
        if not line or "::" not in line:
            continue

        # Partitionner sur le PREMIER "::" uniquement
        key, _, raw_value = line.partition("::")
        key = key.strip()

        # Ignorer les clés vides (ligne malformée type ":: valeur")
        if not key:
            continue

        result[key] = _infer_value(raw_value)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# API publique
# ─────────────────────────────────────────────────────────────────────────────


def parse_toon_to_dict(raw: str) -> dict[str, Any]:
    """Convertit une chaîne TOON brute en dictionnaire Python.

    Processus :
        1. Extraction du bloc TOON via ``_extract_toon_block()``
        2. Parsing ligne par ligne des paires ``clé :: valeur``
        3. Inférence de type pour chaque valeur via ``_infer_value()``

    La fonction est tolérante aux imperfections de formatage du LLM :
    lignes vides ignorées, espaces superflus strippés, casse insensible
    non forcée (la normalisation de casse reste à la charge de Pydantic).

    Args:
        raw: La chaîne brute retournée par le LLM. Peut contenir du texte
             parasite, des blocs Markdown, ou être un bloc TOON pur.

    Returns:
        Un dictionnaire ``dict[str, Any]`` prêt à être passé à
        ``ModelName.model_validate()``.

    Raises:
        ToonParseError: Si aucune paire clé/valeur n'est détectable,
            ou si la chaîne d'entrée est vide.

    Examples:
        >>> raw = \"\"\"<<<
        ... query_type :: SIMPLE
        ... confidence :: 0.97
        ... detected_entities :: rétropropagation
        ... reasoning_budget :: 1
        ... >>>\"\"\"
        >>> parse_toon_to_dict(raw)
        {'query_type': 'SIMPLE', 'confidence': 0.97,
         'detected_entities': ['rétropropagation'], 'reasoning_budget': 1}
    """
    if not raw or not raw.strip():
        raise ToonParseError(reason="chaîne d'entrée vide", raw=raw)

    block_content = _extract_toon_block(raw)
    result = _parse_block_lines(block_content)

    if not result:
        raise ToonParseError(
            reason="aucune paire clé/valeur parsée depuis le bloc extrait",
            raw=raw,
        )

    return result


def dump_dict_to_toon(data: dict[str, Any]) -> str:
    """Sérialise un dictionnaire Python en chaîne au format TOON.

    Règles de sérialisation :
        - ``list``  → éléments joints par `` | ``  (liste vide → vide)
        - ``None``  → chaîne vide après ``::``
        - ``bool``  → ``True`` ou ``False`` (minuscule non forcée)
        - ``int``/``float`` → représentation décimale standard
        - ``str``   → valeur brute (sans guillemets ni échappement)

    Args:
        data: Le dictionnaire à sérialiser. Typiquement le retour de
              ``.model_dump()`` d'un modèle Pydantic, mais la fonction
              accepte n'importe quel ``dict[str, Any]``.

    Returns:
        Une chaîne multi-lignes encadrée par ``<<<`` et ``>>>``, conforme
        à la syntaxe TOON v1.0.

    Raises:
        TypeError: Si ``data`` n'est pas un dictionnaire.

    Examples:
        >>> dump_dict_to_toon({
        ...     "query_type": "MULTI_HOP",
        ...     "confidence": 0.92,
        ...     "detected_entities": ["LangChain", "LangSmith"],
        ...     "reasoning_budget": 3,
        ... })
        '<<<\\nquery_type :: MULTI_HOP\\nconfidence :: 0.92\\n...'
    """
    if not isinstance(data, dict):
        raise TypeError(
            f"dump_dict_to_toon attend un dict, reçu {type(data).__name__!r}"
        )

    lines: list[str] = ["<<<"]

    for key, value in data.items():
        if isinstance(value, list):
            if not value:
                serialized = ""
            elif len(value) == 1:
                # Séparateur final explicite pour disambiguïser list[str] vs str
                serialized = f"{value[0]} |"
            else:
                serialized = " | ".join(str(item) for item in value)
        elif value is None:
            serialized = ""
        elif isinstance(value, float):
            # Représentation compacte : 0.92 plutôt que 0.9200000000000001
            serialized = f"{value:g}"
        else:
            serialized = str(value)

        lines.append(f"{key} :: {serialized}")

    lines.append(">>>")
    return "\n".join(lines)


_RECORD_SEP = "---"


def parse_toon_records(raw: str) -> list[dict[str, Any]]:
    """Parse un bloc TOON multi-enregistrements en liste de dictionnaires.

    Utilisé exclusivement par le Verifier (Sprint 5) dont le format de sortie
    LLM est un unique bloc ``<<<...>>>`` contenant plusieurs enregistrements
    de claim séparés par une ligne ``---``.

    Processus :
        1. Extraction du bloc TOON via ``_extract_toon_block()``
        2. Découpage du contenu sur les lignes égales à ``---`` (après strip)
        3. Parsing de chaque section via ``_parse_block_lines()``
        4. Neutralisation des sections vides de BORDURE (séparateur parasite
           en tête ou en fin de bloc) — artefact de formatage LLM courant
        5. Rejet des sections vides INTERMÉDIAIRES (``---`` consécutifs au
           milieu du bloc), qui signalent un enregistrement manquant

    Compatibilité :
        ``parse_toon_to_dict()``, ``_extract_toon_block()``,
        ``dump_dict_to_toon()`` et ``_infer_value()`` ne sont pas modifiées
        et restent disponibles pour Analyzer, Planner et Critic.

    Piège de typage :
        ``is_supported`` (champ Verifier) est parsé comme chaîne
        ``"true"``/``"false"`` — jamais converti en ``bool`` Python.
        Comparer avec ``record["is_supported"].strip().lower() == "true"``.

    Args:
        raw: La chaîne brute retournée par le LLM. Peut contenir du texte
             parasite autour du bloc ``<<<...>>>``.

    Returns:
        Liste de ``dict[str, Any]``, un dict par enregistrement, dans
        l'ordre d'apparition. Retourne une liste vide si le bloc
        ``<<<...>>>`` est présent mais ne contient aucun enregistrement
        (bloc vide, ex. zéro claim, ou bloc composé uniquement de
        séparateurs).

    Raises:
        ToonParseError: Si le bloc ``<<<...>>>`` est absent ou malformé
            (propagée depuis ``_extract_toon_block()``), ou si un
            enregistrement INTERMÉDIAIRE entre deux séparateurs ``---``
            est vide (absence totale de paires clé/valeur dans la section).
            Un séparateur en tête ou en fin de bloc n'est PAS une erreur :
            il est ignoré silencieusement.

    Examples:
        >>> raw = '''
        ... <<<
        ... claim_text      :: Rabat est la capitale du Maroc
        ... is_supported    :: true
        ... source_chunk_id :: c1
        ... ---
        ... claim_text      :: La ville compte 500000 habitants
        ... is_supported    :: false
        ... source_chunk_id ::
        ... >>>
        ... '''
        >>> records = parse_toon_records(raw)
        >>> len(records)
        2
        >>> records[0]["claim_text"]
        'Rabat est la capitale du Maroc'
        >>> records[0]["is_supported"]  # str, PAS bool
        'true'
        >>> records[1]["source_chunk_id"] is None  # valeur vide → None
        True
    """
    if not raw or not raw.strip():
        raise ToonParseError(reason="chaîne d'entrée vide", raw=raw)

    block_content = _extract_toon_block(raw)

    # Algorithme d'accumulation : regroupe les lignes par séparateur "---"
    current_lines: list[str] = []
    sections: list[str] = []
    for line in block_content.splitlines():
        if line.strip() == _RECORD_SEP:
            sections.append("\n".join(current_lines))
            current_lines = []
        else:
            current_lines.append(line)
    # Dernière section (pas de "---" terminal)
    sections.append("\n".join(current_lines))

    # Parsing de chaque section, en conservant son index d'ORIGINE : après
    # neutralisation des sections de bordure ci-dessous, l'index d'une
    # malformation interne doit rester celui du bloc tel que produit par le LLM.
    parsed_sections: list[tuple[int, dict[str, Any]]] = [
        (i, _parse_block_lines(section)) for i, section in enumerate(sections)
    ]

    # Un séparateur '---' parasite en TÊTE ou en FIN de bloc génère une section
    # vide dénuée d'ambiguïté sémantique : aucun enregistrement n'est perdu, il
    # s'agit d'un simple artefact de formatage (le LLM clôt volontiers sa liste
    # par un séparateur). On l'ignore silencieusement.
    # Une section vide INTERMÉDIAIRE reste au contraire une vraie malformation :
    # elle signale un enregistrement manquant entre deux séparateurs.
    while parsed_sections and not parsed_sections[0][1]:
        parsed_sections.pop(0)
    while parsed_sections and not parsed_sections[-1][1]:
        parsed_sections.pop()

    # Bloc entièrement vide (<<<\n>>>) ou composé uniquement de séparateurs
    # → zéro claim, comportement inchangé : liste vide, pas d'erreur.
    if not parsed_sections:
        return []

    records: list[dict[str, Any]] = []

    for index, parsed in parsed_sections:
        if not parsed:
            raise ToonParseError(
                reason=(
                    f"enregistrement vide détecté à l'index {index} "
                    "(aucune paire clé/valeur entre deux séparateurs '---')"
                ),
                raw=raw,
            )
        records.append(parsed)

    return records
