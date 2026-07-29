"""Package Planner du module REASONING.

Expose la classe `Planner`, composant responsable de la décomposition
d'une requête complexe en un graphe acyclique dirigé (DAG) d'étapes
atomiques (ExecutionPlan), conformément à docs/planner_spec.md.
"""

from reasoning.planner.planner import Planner

__all__ = ["Planner"]
