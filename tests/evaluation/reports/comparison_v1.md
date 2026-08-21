# Évaluation comparative — Sprint I5-B

Généré le 2026-08-21T01:15:47+00:00.

- **Juge** : `ollama/qwen2.5:7b` · **Embeddings** : `BAAI/bge-small-en-v1.5`
- **Questions** : 20 (baseline) / 20 (système complet)

> **Biais à connaître.** Le modèle qui génère les réponses est aussi
> celui qui les juge. Les valeurs absolues sont donc optimistes.
> L'écart entre les deux systèmes reste interprétable : le biais leur
> est commun.

## Global

| Métrique | Baseline naïf | Système complet | Écart (points) |
|---|---:|---:|---:|
| `faithfulness` | 0.75 | 0.7333 | **-1.7** |
| `answer_relevancy` | 0.5088 | 0.6716 | **+16.3** |
| `context_precision` | 0.2083 | 0.2144 | **+0.6** |
| `context_recall` | 0.75 | 0.8667 | **+11.7** |
| appels LLM / question | 1.0 | 5.55 | — |
| latence moyenne (s) | 37.6 | 321.6 | — |

## Bridge — multi-hop

| Métrique | Baseline naïf | Système complet | Écart (points) |
|---|---:|---:|---:|
| `faithfulness` | 0.7222 | 0.6833 | **-3.9** |
| `answer_relevancy` | 0.431 | 0.6547 | **+22.4** |
| `context_precision` | 0.2667 | 0.2955 | **+2.9** |
| `context_recall` | 0.6 | 0.8 | **+20.0** |
| appels LLM / question | 1.0 | 5.9 | — |
| latence moyenne (s) | 39.3 | 378.1 | — |

## Comparison

| Métrique | Baseline naïf | Système complet | Écart (points) |
|---|---:|---:|---:|
| `faithfulness` | 0.775 | 0.7833 | **+0.8** |
| `answer_relevancy` | 0.5866 | 0.6884 | **+10.2** |
| `context_precision` | 0.15 | 0.1333 | **-1.7** |
| `context_recall` | 0.9 | 0.9333 | **+3.3** |
| appels LLM / question | 1.0 | 5.2 | — |
| latence moyenne (s) | 36.0 | 265.2 | — |

## KPI principal

**Gain de faithfulness : -1.7 points** (cible indicative du cahier des charges : +15).
