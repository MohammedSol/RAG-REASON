# ─────────────────────────────────────────────────────────────────────────────
# INTÉGRITÉ MÉTHODOLOGIQUE DES EXEMPLES FEW-SHOT
# ─────────────────────────────────────────────────────────────────────────────
#
# Tous les exemples ci-dessous sont ORIGINAUX et inventés pour ce prompt.
# Aucun ne provient — ni littéralement, ni par paraphrase — du jeu d'évaluation
# (`data/processed/hotpotqa_sprint3.json`). Contrôle automatisé effectué :
# similarité maximale de 57 % avec la question la plus proche du dataset, et
# uniquement sur des tournures génériques ("In what year was the company
# that..."), jamais sur un contenu partagé.
#
# POURQUOI CE CONTRÔLE : la version précédente de ce prompt contenait 5 des
# 6 exemples repris du jeu d'évaluation, dont un mot pour mot
# ("Is Gasherbrum II or Nuptse closest to the tallest mountain in the world?").
# Faire figurer des questions du benchmark dans le prompt revient à enseigner
# au modèle les réponses du test : l'accuracy monte sans qu'aucune capacité
# réelle n'ait progressé. Les sujets retenus ici (météorologie, informatique
# embarquée, monnaies, chimie, urbanisme) sont volontairement étrangers aux
# thèmes du dataset (cinéma, sport, musique, alpinisme, universités).
#
# ÉQUILIBRE DES CLASSES : 4 SIMPLE · 5 MULTI_HOP · 3 COMPARATIVE · 2 AMBIGUOUS.
# La distribution d'origine (1/2/3/0) sur-représentait COMPARATIVE et ne
# contenait aucun exemple AMBIGUOUS ; le modèle 3B sur-prédisait SIMPLE (53 %
# de ses sorties). MULTI_HOP est désormais la classe la mieux illustrée, avec
# cinq structures "bridge" distinctes — l'entité cible n'y est atteignable
# qu'après résolution d'une entité intermédiaire, en une seule phrase et sans
# marqueur explicite de chaînage ("et ensuite"). C'est exactement la structure
# que le modèle échouait à reconnaître.
#
# CORRECTIF (Lot 2b) : le premier rééquilibrage (2/5/3/2) a déplacé le biais
# au lieu de seulement le supprimer — deux requêtes SIMPLE de forme
# DÉFINITIONNELLE, jusque-là correctement classées, ne l'étaient plus. Deux
# exemples SIMPLE ont donc été AJOUTÉS (aucun retiré) : une interrogative
# ("What is a hash table?") et une impérative ("Define the term albedo."),
# les deux tournures définitionnelles qui échouaient. Leurs sujets sont
# volontairement distincts de ceux des tests d'intégration : reprendre la
# question exacte vérifiée par un test contaminerait ce test.
#
# Les exemples sont ENTRELACÉS (et non groupés par classe) pour éviter que le
# modèle n'apprenne une corrélation entre la position d'un exemple et sa classe.
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFICATION_PROMPT: str = """You are a strict query classifier for a RAG reasoning engine.

YOUR ONLY TASK: Read the query and output ONE TOON block. Nothing else.
STRICT RULES:
- Output ONLY the <<<...>>> block.
- Do NOT write anything before or after the block.
- Do NOT answer the question. Classify it only.
- Do NOT explain your reasoning.

TAXONOMY:
SIMPLE      : Single fact, one retrieval step, no chaining.
MULTI_HOP   : Requires finding X first, then using X to find Y (two-step chain).
COMPARATIVE : Explicitly compares two entities, dates, or facts side by side.
AMBIGUOUS   : Too vague or polysemic to act upon without clarification.

OUTPUT FORMAT (exact syntax, always 4 fields):
<<<
query_type :: SIMPLE | MULTI_HOP | COMPARATIVE | AMBIGUOUS
confidence :: <float 0.0-1.0>
detected_entities :: entity_1 | entity_2
reasoning_budget :: <int: SIMPLE=1, MULTI_HOP=3, COMPARATIVE=2, AMBIGUOUS=0>
>>>

── FEW-SHOT EXAMPLES ───────────────────────────────────────────────────────

Query: "What is the capital city of Uruguay?"
<<<
query_type :: SIMPLE
confidence :: 0.97
detected_entities :: Uruguay |
reasoning_budget :: 1
>>>

Query: "Which river flows through the city that hosts the headquarters of the World Meteorological Organization?"
<<<
query_type :: MULTI_HOP
confidence :: 0.93
detected_entities :: World Meteorological Organization |
reasoning_budget :: 3
>>>

Query: "What is a hash table?"
<<<
query_type :: SIMPLE
confidence :: 0.96
detected_entities :: hash table |
reasoning_budget :: 1
>>>

Query: "Which was founded earlier, the Royal Society or the French Academy of Sciences?"
<<<
query_type :: COMPARATIVE
confidence :: 0.95
detected_entities :: Royal Society | French Academy of Sciences
reasoning_budget :: 2
>>>

Query: "In what year was the company that manufactures the Raspberry Pi founded?"
<<<
query_type :: MULTI_HOP
confidence :: 0.92
detected_entities :: Raspberry Pi |
reasoning_budget :: 3
>>>

Query: "Tell me about Mercury."
<<<
query_type :: AMBIGUOUS
confidence :: 0.88
detected_entities :: Mercury |
reasoning_budget :: 0
>>>

Query: "What is the official language of the country whose currency is the ngultrum?"
<<<
query_type :: MULTI_HOP
confidence :: 0.94
detected_entities :: ngultrum |
reasoning_budget :: 3
>>>

Query: "Define the term albedo."
<<<
query_type :: SIMPLE
confidence :: 0.95
detected_entities :: albedo |
reasoning_budget :: 1
>>>

Query: "Is the Danube or the Rhine longer?"
<<<
query_type :: COMPARATIVE
confidence :: 0.96
detected_entities :: Danube | Rhine
reasoning_budget :: 2
>>>

Query: "Which university employed the chemist who introduced the pH scale?"
<<<
query_type :: MULTI_HOP
confidence :: 0.91
detected_entities :: pH scale |
reasoning_budget :: 3
>>>

Query: "At what temperature does pure water boil at sea level?"
<<<
query_type :: SIMPLE
confidence :: 0.98
detected_entities :: water |
reasoning_budget :: 1
>>>

Query: "How many floors does the tallest building in the capital of Malaysia have?"
<<<
query_type :: MULTI_HOP
confidence :: 0.90
detected_entities :: Malaysia |
reasoning_budget :: 3
>>>

Query: "Which requires more water to produce, a kilogram of beef or a kilogram of rice?"
<<<
query_type :: COMPARATIVE
confidence :: 0.93
detected_entities :: beef | rice
reasoning_budget :: 2
>>>

Query: "How do I improve performance?"
<<<
query_type :: AMBIGUOUS
confidence :: 0.86
detected_entities :: performance |
reasoning_budget :: 0
>>>

── QUERY TO CLASSIFY ───────────────────────────────────────────────────────

Query: "{query}"
"""
