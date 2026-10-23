"""
Modulo: base (src/estimators)
Interfaccia astratta per qualsiasi oracolo che stimi il costo residuo di un bucket.
"""
from abc import ABC, abstractmethod


class BaseBucketEstimator(ABC):
    """Stimatore del numero medio di mosse necessarie per risolvere un bucket."""

    @abstractmethod
    def estimate_cost(self, solutions: list[str]) -> float:
        """
        PRE:  solutions è una lista non vuota di combinazioni.
        POST: costo atteso (float >= 1.0), contando anche la mossa che chiude la partita.
        """