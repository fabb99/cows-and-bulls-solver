"""
Modulo: entropy
Strategia che massimizza l'entropia di Shannon della partizione (profondità 1).
"""
import math

from src.core.engine import calculate_buckets
from src.strategies.base import BaseStrategy


class EntropyStrategy(BaseStrategy):
    higher_is_better = True

    def evaluate_candidates(
        self, candidates_to_test: list[str], possible_solutions: list[str]
    ) -> dict[str, float]:
        """POST: {candidato: entropia (bit) della partizione}. Più alta è meglio."""
        n = len(possible_solutions)
        scores: dict[str, float] = {}
        for candidate in candidates_to_test:
            buckets = calculate_buckets(candidate, possible_solutions, self.lookup)
            scores[candidate] = -sum(
                (len(b) / n) * math.log2(len(b) / n) for b in buckets.values()
            )
        return scores