"""
Modulo: deep_evm
EVM ricorsiva pura: costo atteso minimo esatto (equazione di Bellman), senza approssimazioni.
Ottimizzata solo con memoization e Branch & Bound rigoroso.
"""
import math

from src.core.constants import WIN_FEEDBACK
from src.core.engine import calculate_buckets
from src.core.lookup_table import FeedbackLookup
from src.core.state import state_id
from src.strategies.base import BaseStrategy
from src.strategies.costs import expected_cost


class PureDeepEVMStrategy(BaseStrategy):
    higher_is_better = False

    def __init__(
        self,
        all_permutations: list[str],
        lookup: FeedbackLookup,
        warmup_strategy: BaseStrategy,
        precomputed_tablebase: dict[str, float] | None = None,
    ) -> None:
        """
        PRE:  warmup_strategy serve SOLO a trovare un primo upper bound reale
              (non altera la completezza della ricerca).
              precomputed_tablebase = {state_id: costo_esatto}.
        """
        super().__init__(all_permutations, lookup)
        self.warmup_strategy = warmup_strategy
        self.precomputed_tablebase = precomputed_tablebase or {}
        self.cost_cache: dict[frozenset[str], float] = {}

    def _compute_exact_cost(self, solutions: list[str]) -> float:
        """
        PRE:  solutions non vuota.
        POST: costo atteso minimo esatto per risolvere `solutions`.
              Esplora tutti i candidati; il warmup viene provato per primo per
              abbassare presto il bound; un candidato è scartato (Branch & Bound)
              appena il suo costo parziale raggiunge il miglior costo noto.
        """
        n = len(solutions)
        if n == 1:
            return 1.0
        if n == 2:
            return 1.5

        if self.precomputed_tablebase:
            known = self.precomputed_tablebase.get(state_id(solutions))
            if known is not None:
                return known

        key = frozenset(solutions)
        cached = self.cost_cache.get(key)
        if cached is not None:
            return cached

        warmup = self.warmup_strategy.choose_guess(solutions)
        candidates = [warmup] + [c for c in self.all_permutations if c != warmup]

        best = math.inf
        for candidate in candidates:
            buckets = calculate_buckets(candidate, solutions, self.lookup)
            if len(buckets) == 1 and WIN_FEEDBACK not in buckets:
                continue  # non partiziona: scartata (evita ricorsione infinita)

            score = 1.0
            pruned = False
            # bucket grandi per primi: fanno salire il costo parziale più in fretta
            for fb, bucket in sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True):
                if fb == WIN_FEEDBACK:
                    continue
                score += (len(bucket) / n) * self._compute_exact_cost(bucket)
                if score >= best:
                    pruned = True
                    break
            if not pruned and score < best:
                best = score

        self.cost_cache[key] = best
        return best

    def evaluate_candidates(
        self, candidates_to_test: list[str], possible_solutions: list[str]
    ) -> dict[str, float]:
        """POST: {candidato: costo atteso esatto}. I candidati che non partizionano ottengono inf."""
        n = len(possible_solutions)
        return {
            c: expected_cost(
                calculate_buckets(c, possible_solutions, self.lookup), n, self._compute_exact_cost
            )
            for c in candidates_to_test
        }

    def exact_cost(self, solutions: list[str]) -> float:
        """PRE: solutions non vuota. POST: costo atteso minimo esatto (con memoization)."""
        return self._compute_exact_cost(solutions)