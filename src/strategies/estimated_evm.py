"""
Modulo: estimated_evm
EVM stimata: costo atteso di un candidato = 1 + sum P(bucket) * stima(bucket),
dove la stima viene da un BaseBucketEstimator.
"""

from src.core.engine import calculate_buckets
from src.core.lookup_table import FeedbackLookup
from src.estimators.base import BaseBucketEstimator
from src.strategies.base import BaseStrategy
from src.strategies.costs import expected_cost


class EstimatedEVMStrategy(BaseStrategy):
    higher_is_better = False

    def __init__(
        self,
        all_permutations: list[str],
        lookup: FeedbackLookup,
        estimator: BaseBucketEstimator,
        exact_fallback_strategy: BaseStrategy | None = None,
        fallback_threshold: int = 0,
    ) -> None:
        """
        PRE:  se fallback_threshold > 0 serve exact_fallback_strategy.
        POST: con 2 < N < fallback_threshold la scelta è delegata alla strategia di fallback.
        """
        super().__init__(all_permutations, lookup)
        if fallback_threshold > 0 and exact_fallback_strategy is None:
            raise ValueError("fallback_threshold > 0 richiede exact_fallback_strategy")
        self.estimator = estimator
        self.exact_fallback_strategy = exact_fallback_strategy
        self.fallback_threshold = fallback_threshold

    def evaluate_candidates(
        self, candidates_to_test: list[str], possible_solutions: list[str]
    ) -> dict[str, float]:
        """
        POST: {candidato: costo atteso stimato}. Più basso è meglio.
              I candidati che non partizionano ottengono math.inf.
        """
        n = len(possible_solutions)
        return {
            c: expected_cost(
                calculate_buckets(c, possible_solutions, self.lookup),
                n,
                self.estimator.estimate_cost,
            )
            for c in candidates_to_test
        }

    def choose_guess(self, possible_solutions, history=None) -> str:
        if self.exact_fallback_strategy is not None and (
            2 < len(possible_solutions) < self.fallback_threshold
        ):
            return self.exact_fallback_strategy.choose_guess(possible_solutions, history)
        return super().choose_guess(possible_solutions, history)