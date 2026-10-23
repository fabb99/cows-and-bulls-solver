"""
Modulo: tablebase (src/estimators)
Pattern Decorator: se lo stato è nella tablebase restituisce il costo esatto in O(1),
altrimenti delega a un altro estimator.
"""
from src.core.state import state_id
from src.estimators.base import BaseBucketEstimator


class TablebaseBucketEstimator(BaseBucketEstimator):
    def __init__(self, tablebase: dict[str, float], fallback: BaseBucketEstimator) -> None:
        """
        PRE: tablebase = {state_id: costo_esatto}; fallback è un estimator valido.
        """
        self.tablebase = tablebase
        self.fallback = fallback

    def estimate_cost(self, solutions: list[str]) -> float:
        """
        PRE:  solutions non vuota.
        POST: costo esatto se lo stato è in tablebase, altrimenti la stima del fallback.
        """
        key = state_id(solutions)
        if key in self.tablebase:
            return self.tablebase[key]
        return self.fallback.estimate_cost(solutions)