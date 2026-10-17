"""
Modulo: memoized
Wrapper con cache per stato di una strategia.

Valido per strategie la cui scelta dipende solo dall'insieme di soluzioni e, al Turno 1,
dalla history (pool ridotto a un rappresentante). Il Turno 1 non viene mai messo in cache.
"""
from src.core.state import state_hash
from src.strategies.base import BaseStrategy


class MemoizedStrategy(BaseStrategy):
    def __init__(self, inner: BaseStrategy) -> None:
        """PRE: inner è una strategia pronta. POST: stessa lookup, universo e verso dei punteggi."""
        super().__init__(inner.all_permutations, inner.lookup)
        self.inner = inner
        self.higher_is_better = inner.higher_is_better
        self._cache: dict[str, str] = {}
        self.hits = 0
        self.misses = 0

    def evaluate_candidates(
        self, candidates_to_test: list[str], possible_solutions: list[str]
    ) -> dict[str, float]:
        """POST: delega alla strategia interna (nessuna cache)."""
        return self.inner.evaluate_candidates(candidates_to_test, possible_solutions)

    def choose_guess(
        self, possible_solutions: list[str], history: list[str] | None = None
    ) -> str:
        """
        PRE:  possible_solutions non vuota.
        POST: stessa mossa della strategia interna. Con N <= 2 la prima soluzione;
              al Turno 1 (history vuota, universo intero) delega senza cache;
              altrimenti usa la cache per state_hash.
        """
        if not possible_solutions:
            raise ValueError("possible_solutions è vuota")
        if len(possible_solutions) <= 2:
            return possible_solutions[0]
        if not history and len(possible_solutions) == len(self.all_permutations):
            return self.inner.choose_guess(possible_solutions, history)

        key = state_hash(possible_solutions)
        guess = self._cache.get(key)
        if guess is None:
            self.misses += 1
            guess = self.inner.choose_guess(possible_solutions, history)
            self._cache[key] = guess
        else:
            self.hits += 1
        return guess