"""
Package public du module Query Analyzer.

Expose l'interface publique minimale : seule la classe `QueryAnalyzer`
doit être importée par les consommateurs externes (nœuds LangGraph, tests).

Usage :
    from reasoning.analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer()
    result = analyzer.analyze("Qui a fondé OpenAI ?")
"""

from reasoning.analyzer.analyzer import QueryAnalyzer

__all__ = ["QueryAnalyzer"]
