"""Ogni strategia, contro ogni segreto di un dominio ridotto: mosse legali, progresso, terminazione."""
import pytest

from src.core.constants import WIN_FEEDBACK
from src.core.engine import filter_solutions
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.estimators.segmented import SegmentedLinearEstimator
from src.strategies.basic import MaxFrequencyStrategy, NaiveStrategy
from src.strategies.deep_evm import PureDeepEVMStrategy
from src.strategies.entropy import EntropyStrategy
from src.strategies.estimated_evm import EstimatedEVMStrategy
from src.strategies.greedy_minimax import GreedyMinimaxStrategy
from src.strategies.hybrid import HybridWeightedStrategy

NAMES = ["naive", "max_frequency", "minimax", "entropy", "estimated_evm", "hybrid", "deep_evm"]
MAX_TURNS = 12


@pytest.fixture(scope="module")
def domain():
    return generate_all_permutations()[:40]


@pytest.fixture(scope="module")
def lookup(domain):
    return FeedbackLookup(domain)


@pytest.fixture(scope="module")
def strategies(domain, lookup):
    entropy = EntropyStrategy(domain, lookup)
    minimax = GreedyMinimaxStrategy(domain, lookup)
    return {
        "naive": NaiveStrategy(domain, lookup),
        "max_frequency": MaxFrequencyStrategy(domain, lookup),
        "minimax": minimax,
        "entropy": entropy,
        "estimated_evm": EstimatedEVMStrategy(
            domain, lookup, SegmentedLinearEstimator.legacy_geometric()
        ),
        "hybrid": HybridWeightedStrategy(entropy, minimax, 0.94, 0.06),
        "deep_evm": PureDeepEVMStrategy(domain, lookup, entropy),
    }


@pytest.mark.parametrize("name", NAMES)
def test_strategy_solves_every_secret(name, strategies, domain, lookup):
    strategy = strategies[name]
    for secret in domain:
        solutions, history = list(domain), []
        for _ in range(MAX_TURNS):
            guess = strategy.choose_guess(solutions, history)
            assert guess in domain, "mossa illegale"
            history.append(guess)
            fb = lookup.get_feedback(guess, secret)
            if fb == WIN_FEEDBACK:
                break
            remaining = filter_solutions(guess, fb, solutions, lookup)
            assert secret in remaining
            assert len(remaining) < len(solutions), "nessun progresso: rischio di loop"
            solutions = remaining
        else:
            pytest.fail(f"{name} non ha risolto {secret} in {MAX_TURNS} mosse")


def test_deep_evm_is_never_worse_than_entropy(strategies, domain, lookup):
    """Il costo esatto è un minimo: sul dominio ridotto non può superare l'entropia."""
    from src.analysis.metrics import average_depth
    from src.analysis.tree_builder import build_decision_tree

    avg = {
        n: average_depth(build_decision_tree(strategies[n], domain))
        for n in ("entropy", "deep_evm")
    }
    assert avg["deep_evm"] <= avg["entropy"] + 1e-9