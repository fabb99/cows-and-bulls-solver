"""
Modulo: basic
Strategie di base: Naive e Max Frequenza. Giocano solo tra le soluzioni possibili.
"""
from collections import Counter

from src.core.constants import DIM
from src.strategies.base import BaseStrategy


class NaiveStrategy(BaseStrategy):
    """Gioca la prima soluzione ancora compatibile."""

    higher_is_better = True

    def candidate_pool(self, possible_solutions: list[str], history: list[str]) -> list[str]:
        return possible_solutions

    def evaluate_candidates(
        self, candidates_to_test: list[str], possible_solutions: list[str]
    ) -> dict[str, float]:
        """POST: 2 per la prima soluzione, 1 per le altre soluzioni, 0 per le non-soluzioni."""
        solutions = set(possible_solutions)
        return {
            c: 2.0 if c == possible_solutions[0] else 1.0 if c in solutions else 0.0
            for c in candidates_to_test
        }


class MaxFrequencyStrategy(BaseStrategy):
    """Tra le soluzioni possibili, sceglie quella che massimizza le frequenze posizionali."""

    higher_is_better = True

    def candidate_pool(self, possible_solutions: list[str], history: list[str]) -> list[str]:
        return possible_solutions

    @staticmethod
    def _positional_frequencies(solutions: list[str]) -> list[Counter]:
        """POST: per ogni posizione 0..DIM-1, il conteggio di ogni cifra in quella posizione."""
        return [Counter(s[pos] for s in solutions) for pos in range(DIM)]

    def evaluate_candidates(
        self, candidates_to_test: list[str], possible_solutions: list[str]
    ) -> dict[str, float]:
        """POST: {candidato: somma delle frequenze posizionali delle sue cifre}."""
        freqs = self._positional_frequencies(possible_solutions)
        return {
            c: float(sum(freqs[pos][c[pos]] for pos in range(DIM))) for c in candidates_to_test
        }