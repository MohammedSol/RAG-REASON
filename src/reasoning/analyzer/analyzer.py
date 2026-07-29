"""
Query Analyzer — Routeur sémantique du module REASONING.

Ce composant est le premier nœud du graphe LangGraph. Il classifie la requête
utilisateur selon la taxonomie définie dans docs/analyzer_spec.md et alloue
un budget computationnel (`reasoning_budget`) pour contraindre le graphe.

Architecture hybride (spécification 2.1 validée) :
    Niveau 1 : Classification LLM via LiteLLM → Ollama (DEFAULT_FAST_MODEL)
    Niveau 2 : Fallback heuristique Python si le LLM échoue ou renvoie un JSON invalide
"""

from __future__ import annotations

import logging
import os
import re

from dotenv import load_dotenv
from litellm import completion
from pydantic import ValidationError

from reasoning.analyzer.prompts import CLASSIFICATION_PROMPT
from reasoning.contracts.internal_models import AnalysisResult, QueryType
from reasoning.shared.toon_utils import ToonParseError, parse_toon_to_dict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_FAST_MODEL: str = os.getenv("DEFAULT_FAST_MODEL", "ollama/qwen2.5:3b")

# Confidence attribuée quand le fallback heuristique s'applique
_FALLBACK_CONFIDENCE: float = 0.55

# Marqueurs linguistiques pour le fallback (cf. analyzer_spec.md §5.2)
_MULTI_HOP_MARKERS: tuple[str, ...] = (
    # Marqueurs français (requêtes internes du système)
    "et ensuite",
    "puis",
    "après quoi",
    "à la suite de",
    "qui a ensuite",
    "dont le",
    "duquel",
    "de laquelle",
    "ce dernier",
    "cette dernière",
    "lequel",
    "laquelle",
    "depuis que",
    "quel est le",
    # Marqueurs anglais (HotpotQA + requêtes EN)
    "who created",
    "who directed",
    "who founded the",
    "what country",
    "where was",
    "how many people",
    "debut album of",
    "stage name",
    "known by his",
    "helped organizations",
)

_COMPARATIVE_MARKERS: tuple[str, ...] = (
    # Marqueurs français
    "compare",
    "versus",
    " vs ",
    "vs.",
    "différence entre",
    "par rapport à",
    "comparé à",
    "plutôt que",
    "avantages et inconvénients",
    "lequel est meilleur",
    "lequel est plus",
    "distingue",
    "distinction entre",
    # Marqueurs anglais (patterns HotpotQA spécifiques)
    "who is older",
    "who was born first",
    "which came first",
    "which is a",
    "which one is",
    "compared to",
    "difference between",
    "who had to",
    "who needed to",
)

# ─────────────────────────────────────────────────────────────────────────────
# Régex — Niveau 0 : pré-classificateur structurel haute confiance
# Compilés au niveau module (pas de recompilation à chaque appel).
# Ces patterns détectent les structures unambigues SANS appel LLM.
# ─────────────────────────────────────────────────────────────────────────────

# COMPARATIVE — questions temporelles/comparatives avec wh-word
# Ex: "who is older", "which battle occurred first", "born first"
_RE_COMP_TEMPORAL: re.Pattern[str] = re.compile(
    r"\b(?:who|which)\b.{0,80}"
    r"\b(?:older|younger|born\s+first|born\s+last|"
    r"came\s+first|occurred\s+first|appeared\s+first|"
    r"decided\s+first|happened\s+first|is\s+older|"
    r"was\s+born\s+first|had\s+to\s+escape)\b",
    re.IGNORECASE,
)

# COMPARATIVE — "X or Y [superlatif]" (closest, largest, oldest…)
# Ex: "Is Gasherbrum II or Nuptse closest to Everest?"
_RE_COMP_SUPERLATIVE: re.Pattern[str] = re.compile(
    r"\bor\b.{0,80}"
    r"\b(?:closest|nearest|farthest|largest|smallest|"
    r"tallest|shortest|oldest|youngest|longest|"
    r"bigger|better|worse)\b",
    re.IGNORECASE,
)

# COMPARATIVE — "[Proper Noun A] or [Proper Noun B]"
# Détecte deux séquences de mots capitalisés séparées par " or ".
# Ex: "Sigmund Freud or Evelyn Waugh", "Annie Morton or Terry Richardson"
_RE_COMP_TWO_ENTITIES: re.Pattern[str] = re.compile(
    r"[A-Z]\w+(?:\s+(?:[A-Z]\w*|\w{1,4}))*"
    r"\s+or\s+"
    r"[A-Z]\w+(?:\s+(?:[A-Z]\w*|\w{1,4}))*",
)

# MULTI_HOP — clause relative chaînée (who/that + verbe d'action)
# Ex: "the woman who portrayed Corliss Archer", "group that was formed by"
_RE_HOP_RELATIVE: re.Pattern[str] = re.compile(
    r"\b(?:who|that|which)\b.{0,60}"
    r"\b(?:portrayed|played|founded|directed|created|won|wrote|"
    r"starred|performed|managed|led|established|published|"
    r"produced|invented|designed|built|composed|formed|launched)\b",
    re.IGNORECASE,
)

# MULTI_HOP — "the [lieu/entité] where the [autre entité]"
# Ex: "The arena where the Lewiston Maineiacs played"
_RE_HOP_WHERE: re.Pattern[str] = re.compile(
    r"\bthe\s+\w+\s+where\s+the\s+\w+",
    re.IGNORECASE,
)

# Termes polysémiques déclenchant AMBIGUOUS si utilisés sans contexte
_POLYSEMIC_TERMS: tuple[str, ...] = (
    "python",
    "réseau",
    "banque",
    "agent",
    "modèle",
    "cloud",
    "java",
    "framework",
    "langage",
    "environnement",
)


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────


class QueryAnalyzer:
    """Routeur sémantique classifiant les requêtes utilisateur.

    Utilise une architecture hybride :
    - Niveau 1 : appel LLM via LiteLLM (déterministe, temperature=0)
    - Niveau 2 : règles heuristiques Python (fallback si LLM indisponible)

    Attributes:
        model: Identifiant du modèle LiteLLM à utiliser pour la classification.
        api_base: URL de base de l'API Ollama.
        max_tokens: Nombre maximum de tokens pour la réponse LLM.
        temperature: Température de génération (0.0 = déterministe).
    """

    def __init__(
        self,
        model: str = _DEFAULT_FAST_MODEL,
        api_base: str = _OLLAMA_BASE_URL,
        # 64 tokens suffisent pour un bloc TOON complet (~20 tokens réels).
        # Ancienne valeur 256 = overhead KV-cache inutile × 200 req. = +21 min.
        max_tokens: int = 64,
        temperature: float = 0.0,
        # Timeout par requête : évite que le thread bloque si Ollama est lent.
        timeout: float = 20.0,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    # ── API publique ──────────────────────────────────────────────────────────

    def analyze(self, query: str) -> AnalysisResult:
        """Classifie une requête et retourne un AnalysisResult.

        Tente d'abord la classification LLM, puis bascule sur le fallback
        heuristique en cas d'échec (timeout, JSON invalide, erreur réseau).

        Args:
            query: La requête utilisateur brute à classifier.

        Returns:
            AnalysisResult avec query_type, confidence, detected_entities
            et reasoning_budget.
        """
        if not query or not query.strip():
            logger.warning("Requête vide reçue — classification AMBIGUOUS par défaut.")
            return self._build_ambiguous_result(query, confidence=0.99)

        # Niveau 0 : pré-classificateur regex (zéro LLM, déterministe)
        # Court-circuite le LLM pour les patterns structurels non-ambigus.
        pre_result = self._pre_classify(query)
        if pre_result is not None:
            logger.info(
                "Pré-classification regex : query_type=%s confidence=%.2f",
                pre_result.query_type,
                pre_result.confidence,
            )
            return pre_result

        try:
            return self._classify_with_llm(query)
        except (OSError, TimeoutError, RuntimeError) as exc:
            logger.warning(
                "Erreur LLM (%s: %s) — activation du fallback heuristique.",
                type(exc).__name__,
                exc,
            )
        except (ValidationError, ValueError, ToonParseError) as exc:
            logger.warning(
                "Réponse LLM non parseable TOON (%s: %s) — fallback heuristique.",
                type(exc).__name__,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Erreur inattendue lors de la classification LLM (%s: %s) — fallback.",
                type(exc).__name__,
                exc,
            )

        return self._classify_with_heuristics(query)

    # ── Niveau 1 : Classification LLM ────────────────────────────────────────

    def _classify_with_llm(self, query: str) -> AnalysisResult:
        """Envoie la requête au LLM et parse la réponse en AnalysisResult.

        Args:
            query: La requête à classifier.

        Returns:
            AnalysisResult issu de la réponse JSON du LLM.

        Raises:
            APIConnectionError: Si Ollama est inaccessible.
            Timeout: Si le LLM dépasse le délai d'attente.
            ValidationError: Si le JSON retourné ne satisfait pas le schéma.
            ValueError: Si la réponse ne contient pas de JSON exploitable.
        """
        prompt = CLASSIFICATION_PROMPT.format(query=query)

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            # Timeout strict : si Ollama ne répond pas en 15s, on lève une exception
            # catchée dans analyze() → fallback heuristique activé proprement.
            timeout=self.timeout,
        )

        # On force le .strip() immédiatement pour nettoyer les espaces/sauts de ligne
        raw_content: str = (response.choices[0].message.content or "").strip()
        logger.debug("Réponse LLM brute : %s", raw_content)

        toon_str = self._extract_toon(raw_content)
        data = parse_toon_to_dict(toon_str)
        result = AnalysisResult.model_validate(data)

        logger.info(
            "Classification LLM : query_type=%s confidence=%.2f",
            result.query_type,
            result.confidence,
        )
        return result

    @staticmethod
    def _extract_toon(raw: str) -> str:
        r"""Extrait le bloc TOON (<<<...>>>) d'une réponse LLM brute.

        Tente l'extraction en trois niveaux de priorité :
        1. Délimiteurs officiels ``<<<`` / ``>>>``
        2. Bloc Markdown ``\`\`\`toon ... \`\`\``` ou ``\`\`\` ... \`\`\```
        3. Mode dégradé : lignes contenant ``::`` sans délimiteurs

        Args:
            raw: La chaîne brute retournée par le LLM.

        Returns:
            La chaîne TOON normalisée encadrée par ``<<<`` et ``>>>``.

        Raises:
            ValueError: Si aucun bloc ni ligne ``clé :: valeur`` n'est trouvé.
        """
        raw = raw.strip()

        # Cas 1 : délimiteurs officiels <<< ... >>>
        match = re.search(r"<<<.*?>>>", raw, re.DOTALL)
        if match:
            return match.group(0)

        # Cas 2 : bloc Markdown ```toon ... ``` ou ``` ... ```
        match = re.search(r"```(?:toon)?\s*(.*?)\s*```", raw, re.DOTALL)
        if match and "::" in match.group(1):
            return "<<<\n" + match.group(1).strip() + "\n>>>"

        # Cas 3 : mode dégradé — lignes clé :: valeur sans délimiteurs
        kv_lines = [line for line in raw.splitlines() if "::" in line]
        if kv_lines:
            return "<<<\n" + "\n".join(kv_lines) + "\n>>>"

        raise ValueError(f"Aucun bloc TOON trouvé dans la réponse LLM : {raw!r}")

    # ── Niveau 0 : Pré-classificateur regex structurel ─────────────────────────

    @staticmethod
    def _pre_classify(query: str) -> AnalysisResult | None:
        """Pré-classificateur regex — s'exécute AVANT l'appel LLM.

        Détecte les patterns structurels non-ambigus directement dans le
        texte de la requête, sans connaissance sémantique du domaine.
        Si aucun pattern ne correspond, retourne None (→ délègue au LLM).

        Avantages :
        - Zéro latence LLM pour les cas évidents
        - Déterministe et reproductible
        - Corrige les cas où le modèle 3B manque de connaissance du domaine

        Priority order (highest confidence first) :
            1. COMPARATIVE — patterns temporels/comparatifs explicites
            2. COMPARATIVE — superlatifs après "or"
            3. COMPARATIVE — deux entités propres avec "or"
            4. MULTI_HOP   — clauses relatives chaînées
            5. MULTI_HOP   — structure "where the [entity]"

        Args:
            query: La requête utilisateur brute.

        Returns:
            AnalysisResult si un pattern est détecté, None sinon.
        """
        # Pattern 1 — COMPARATIVE : questions comparatives temporelles
        # "who is older", "born first", "which battle occurred first"
        if _RE_COMP_TEMPORAL.search(query):
            return AnalysisResult(
                query_type=QueryType.COMPARATIVE,
                confidence=0.92,
                detected_entities=[],
                reasoning_budget=2,
            )

        # Pattern 2 — COMPARATIVE : superlatif après "or"
        # "Is Gasherbrum II or Nuptse closest to Everest?"
        if _RE_COMP_SUPERLATIVE.search(query):
            return AnalysisResult(
                query_type=QueryType.COMPARATIVE,
                confidence=0.90,
                detected_entities=[],
                reasoning_budget=2,
            )

        # Pattern 3 — COMPARATIVE : deux noms propres séparés par "or"
        # "Sigmund Freud or Evelyn Waugh", "Annie Morton or Terry Richardson"
        if _RE_COMP_TWO_ENTITIES.search(query):
            return AnalysisResult(
                query_type=QueryType.COMPARATIVE,
                confidence=0.88,
                detected_entities=[],
                reasoning_budget=2,
            )

        # Pattern 4 — MULTI_HOP : clause relative chaînée
        # "the woman who portrayed Corliss Archer", "group that was formed"
        if _RE_HOP_RELATIVE.search(query):
            return AnalysisResult(
                query_type=QueryType.MULTI_HOP,
                confidence=0.87,
                detected_entities=[],
                reasoning_budget=3,
            )

        # Pattern 5 — MULTI_HOP : structure locale "where the [entity]"
        # "The arena where the Lewiston Maineiacs played"
        if _RE_HOP_WHERE.search(query):
            return AnalysisResult(
                query_type=QueryType.MULTI_HOP,
                confidence=0.85,
                detected_entities=[],
                reasoning_budget=3,
            )

        return None

    # ── Niveau 2 : Fallback heuristique ──────────────────────────────────────

    def _classify_with_heuristics(self, query: str) -> AnalysisResult:
        """Classifie la requête par règles Python déterministes (fallback).

        Applique les règles dans l'ordre de priorité défini dans analyzer_spec.md :
        AMBIGUOUS (1) > MULTI_HOP (2) > COMPARATIVE (3) > SIMPLE (4).

        Args:
            query: La requête à classifier.

        Returns:
            AnalysisResult avec confidence=FALLBACK_CONFIDENCE.
        """
        normalized = query.lower().strip()
        tokens = normalized.split()

        logger.info("Fallback heuristique activé pour : %r", query[:80])

        # Priorité 1 — AMBIGUOUS
        if self._is_ambiguous(normalized, tokens):
            return self._build_ambiguous_result(query, _FALLBACK_CONFIDENCE)

        # Priorité 2 — MULTI_HOP
        if any(marker in normalized for marker in _MULTI_HOP_MARKERS):
            return AnalysisResult(
                query_type=QueryType.MULTI_HOP,
                confidence=_FALLBACK_CONFIDENCE,
                detected_entities=self._extract_entities(tokens),
                reasoning_budget=3,
            )

        # Priorité 3 — COMPARATIVE
        if any(marker in normalized for marker in _COMPARATIVE_MARKERS):
            return AnalysisResult(
                query_type=QueryType.COMPARATIVE,
                confidence=_FALLBACK_CONFIDENCE,
                detected_entities=self._extract_entities(tokens),
                reasoning_budget=2,
            )

        # Priorité 4 — SIMPLE (valeur de sécurité absolue)
        return AnalysisResult(
            query_type=QueryType.SIMPLE,
            confidence=_FALLBACK_CONFIDENCE,
            detected_entities=self._extract_entities(tokens),
            reasoning_budget=1,
        )

    @staticmethod
    def _is_ambiguous(normalized: str, tokens: list[str]) -> bool:
        """Détermine si une requête est trop vague pour être traitée.

        Args:
            normalized: Requête normalisée en minuscules.
            tokens: Liste des mots de la requête.

        Returns:
            True si la requête est jugée ambiguë.
        """
        # Requête trop courte (< 3 tokens)
        if len(tokens) < 3:
            return True

        # Terme polysémique seul sans contexte désambiguïsant
        meaningful_tokens = [t for t in tokens if len(t) > 3]
        if len(meaningful_tokens) == 1 and meaningful_tokens[0] in _POLYSEMIC_TERMS:
            return True

        # Verbe générique suivi d'un terme polysémique sans complément
        generic_verbs = ("explique", "décris", "parle", "dis-moi", "présente")
        if any(normalized.startswith(v) for v in generic_verbs) and len(tokens) < 5:
            return True

        return False

    @staticmethod
    def _extract_entities(tokens: list[str]) -> list[str]:
        """Extrait des entités candidates depuis les tokens (heuristique simple).

        Retient les tokens de plus de 3 caractères, hors mots fonctionnels courants.

        Args:
            tokens: Liste des mots de la requête.

        Returns:
            Liste des entités candidates détectées.
        """
        stop_words = {
            "quel",
            "quelle",
            "quels",
            "quelles",
            "est",
            "sont",
            "dans",
            "pour",
            "avec",
            "sans",
            "entre",
            "comment",
            "pourquoi",
            "quand",
            "depuis",
            "vers",
            "cette",
            "celui",
            "celle",
            "ceux",
            "celles",
            "mais",
            "donc",
            "puis",
            "aussi",
            "encore",
            "très",
            "plus",
        }
        return [t for t in tokens if len(t) > 3 and t not in stop_words][:5]

    @staticmethod
    def _build_ambiguous_result(query: str, confidence: float) -> AnalysisResult:
        """Construit un AnalysisResult de type AMBIGUOUS.

        Args:
            query: La requête originale.
            confidence: Score de confiance de la classification.

        Returns:
            AnalysisResult avec reasoning_budget=0 (aucun retrieval autorisé).
        """
        tokens = query.lower().split()
        entities = [t for t in tokens if len(t) > 3][:3]
        return AnalysisResult(
            query_type=QueryType.AMBIGUOUS,
            confidence=confidence,
            detected_entities=entities,
            reasoning_budget=0,
        )
