import pytest

from experiments.train_router import RouterSolver, extract_policy
from src.analysis.metrics import calculate_total_moves_and_leaves, max_depth
from src.analysis.tree_builder import build_decision_tree
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.core.state import state_hash, state_id
from src.strategies.entropy import EntropyStrategy
from src.strategies.frankenstein import RoutedStrategy
from src.strategies.greedy_minimax import GreedyMinimaxStrategy
from src.strategies.hybrid import HybridWeightedStrategy


@pytest.fixture(scope="module")
def setup():
    domain = generate_all_permutations()[:40]
    lookup = FeedbackLookup(domain)
    entropy = EntropyStrategy(domain, lookup)
    minimax = GreedyMinimaxStrategy(domain, lookup)
    competitors = {
        "entropy": entropy,
        "minimax": minimax,
        "hybrid": HybridWeightedStrategy(entropy, minimax, 0.94, 0.06),
    }
    solver = RouterSolver(competitors, lookup)
    plans = solver.solve(domain)
    return domain, lookup, entropy, competitors, solver, plans


def _tree(setup, objective):
    domain, lookup, entropy, _, solver, _ = setup
    policy = extract_policy(solver, domain, objective)
    return build_decision_tree(RoutedStrategy(domain, lookup, policy, entropy), domain)


def test_solver_values_match_built_trees(setup):
    plan_avg, plan_def = setup[5]
    tree_a, tree_d = _tree(setup, "avg"), _tree(setup, "def")
    assert calculate_total_moves_and_leaves(tree_a)[0] == plan_avg.total
    assert max_depth(tree_a) == plan_avg.depth
    assert calculate_total_moves_and_leaves(tree_d)[0] == plan_def.total
    assert max_depth(tree_d) == plan_def.depth


def test_router_never_worse_than_best_competitor(setup):
    domain, _, _, competitors, _, (plan_avg, plan_def) = setup
    for strategy in competitors.values():
        tree = build_decision_tree(strategy, domain)
        assert plan_avg.total <= calculate_total_moves_and_leaves(tree)[0]
        assert plan_def.depth <= max_depth(tree)


def test_routed_falls_back_outside_policy(setup):
    domain, lookup, entropy, *_ = setup
    routed = RoutedStrategy(domain, lookup, {}, entropy)
    sols = domain[:10]
    assert routed.choose_guess(sols, ["x"]) == entropy.choose_guess(sols, ["x"])
    assert routed.choose_guess(domain[:2], []) == domain[0]


def test_state_hash_is_order_independent_and_distinguishes_states():
    a, b = ["0123", "0132", "0213"], ["0123", "0132", "0231"]
    assert state_hash(a) == state_hash(list(reversed(a)))
    assert state_hash(a) != state_hash(b)
    assert state_id(a) == state_id(list(reversed(a)))