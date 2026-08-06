"""Package Critic du module REASONING.

Expose la classe `Critic`, composant responsable de l'évaluation de la
qualité du contexte récupéré par le module ACTION pour chaque étape du
plan d'exécution, conformément à docs/critic_spec.md.

Usage :
    from reasoning.critic import Critic

    critic = Critic()
    evaluation = critic.evaluate(step, retrieval_response)
"""

from reasoning.critic.critic import Critic

__all__ = ["Critic"]
