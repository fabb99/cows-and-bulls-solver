import pytest

from experiments.train_policy_router import RouterSolver, extract_policy, extract_recipe
from src.analysis.metrics import calculate_total_moves_and_leaves, max_depth
from src.analysis.tree_builder import build_decision_tree
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.strategies.entropy import EntropyStrategy
from src.strategies.frankenstein import FrankensteinStrategy, path_key
from src.strategies.greedy_minimax import GreedyMinimaxStrategy
from src.strategies.hybrid import HybridWeightedStrategy
from src.strategies.policy_router import PolicyRouterStrategy


@pytest.fixture(scope="module")
def env():
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
    solver.solve(domain)
    return domain, lookup, competitors, solver


def test_path_key():
    assert path_key([]) == "START"
    assert path_key([(0, 1), (1, 0)]) == "0,1|1,0"


@pytest.mark.parametrize("objective", ["avg", "def"])
def test_recipe_reproduces_policy_router(env, objective):
    domain, lookup, competitors, solver = env
    recipe = extract_recipe(solver, domain, objective, list(competitors))
    assert "START" in recipe
    frank = FrankensteinStrategy(domain, lookup, recipe, competitors, "entropy")
    policy = extract_policy(solver, domain, objective)
    router = PolicyRouterStrategy(domain, lookup, policy, competitors["entropy"])
    tree_f, tree_r = build_decision_tree(frank, domain), build_decision_tree(router, domain)
    assert calculate_total_moves_and_leaves(tree_f) == calculate_total_moves_and_leaves(tree_r)
    assert max_depth(tree_f) == max_depth(tree_r)


def test_recipe_works_with_any_first_move(env):
    domain, lookup, competitors, solver = env
    recipe = extract_recipe(solver, domain, "avg", list(competitors))
    frank = FrankensteinStrategy(domain, lookup, recipe, competitors, "entropy")
    tree = build_decision_tree(frank, domain, first_move=domain[17])
    assert tree["guess"] == domain[17]
    assert calculate_total_moves_and_leaves(tree)[1] == len(domain)


def test_unknown_strategy_in_recipe_rejected(env):
    domain, lookup, competitors, _ = env
    with pytest.raises(ValueError):
        FrankensteinStrategy(domain, lookup, {"START": "nope"}, competitors, "entropy")