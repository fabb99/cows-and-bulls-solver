import pytest

from experiments.oracle_lab.evaluate_regret import evaluate_state, summarize
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.estimators.base import BaseBucketEstimator
from src.strategies.deep_evm import PureDeepEVMStrategy
from src.strategies.entropy import EntropyStrategy


@pytest.fixture(scope="module")
def env():
    domain = generate_all_permutations()[:60]
    lookup = FeedbackLookup(domain)
    exact = PureDeepEVMStrategy(domain, lookup, EntropyStrategy(domain, lookup))
    return domain, lookup, exact


class ExactOracle(BaseBucketEstimator):
    def __init__(self, exact):
        self.exact = exact

    def estimate_cost(self, solutions):
        return self.exact.exact_cost(solutions)


class ConstantOracle(BaseBucketEstimator):
    def estimate_cost(self, solutions):
        return 2.0


def _states(domain):
    return [domain[i : i + 7] for i in range(0, 56, 7)]


def test_perfect_oracle_has_zero_regret(env):
    domain, lookup, exact = env
    for sols in _states(domain):
        r = evaluate_state(sols, ExactOracle(exact), lookup, domain, exact.exact_cost, exact.exact_cost(sols))
        assert r.regret == pytest.approx(0.0, abs=1e-9)


def test_regret_is_nonnegative_for_a_bad_oracle(env):
    domain, lookup, exact = env
    results = [
        evaluate_state(s, ConstantOracle(), lookup, domain, exact.exact_cost, exact.exact_cost(s))
        for s in _states(domain)
    ]
    assert all(r.regret >= 0 for r in results)
    rows = summarize(results)
    assert rows[-1]["bin"] == "TOTALE" and rows[-1]["states"] == len(results)


def test_impossible_negative_regret_raises(env):
    domain, lookup, exact = env
    sols = _states(domain)[0]
    with pytest.raises(RuntimeError):
        evaluate_state(sols, ExactOracle(exact), lookup, domain, exact.exact_cost, exact.exact_cost(sols) + 1.0)