"""Package Verifier du module REASONING.

Expose la classe `Verifier`, composant responsable de la vérification de
fidélité (Groundedness Check) de la réponse finale par rapport aux chunks
source récupérés par le module ACTION, conformément à docs/verifier_spec.md.

Usage :
    from reasoning.verifier import Verifier

    verifier = Verifier()
    result = verifier.verify(answer, sources)
"""

from reasoning.verifier.verifier import Verifier

__all__ = ["Verifier"]
