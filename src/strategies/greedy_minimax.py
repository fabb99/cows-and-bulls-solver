"""
Modulo: greedy_minimax
Strategia che minimizza la dimensione del bucket peggiore (profondità 1).
"""
from src.core.engine import calculate_buckets
from src.strategies.base import BaseStrategy


class GreedyMinimaxStrategy(BaseStrategy):
    higher_is_better = False

    def evaluate_candidates(
        self, candidates_to_test: list[str], possible_solutions: list[str]
    ) -> dict[str, float]:
        """POST: {candidato: dimensione del bucket più grande}. Più basso è meglio."""
        return {
            c: float(
                max(len(b) for b in calculate_buckets(c, possible_solutions, self.lookup).values())
            )
            for c in candidates_to_test
        }