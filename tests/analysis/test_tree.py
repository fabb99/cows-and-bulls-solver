import pytest

from src.analysis.metrics import average_depth, get_depth_distribution, max_depth
from src.analysis.tree_builder import build_decision_tree
from src.analysis.tree_io import load_tree_from_json, save_tree_to_json
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.strategies.entropy import EntropyStrategy


@pytest.fixture(scope="module")
def tree():
    domain = generate_all_permutations()[:40]
    return build_decision_tree(EntropyStrategy(domain, FeedbackLookup(domain)), domain)


def test_distribution_is_consistent(tree):
    dist = get_depth_distribution(tree)
    assert sum(dist.values()) == 40  # una partita per segreto
    assert max(dist) == max_depth(tree)
    assert average_depth(tree) == pytest.approx(sum(k * v for k, v in dist.items()) / 40)


def test_json_roundtrip(tree, tmp_path):
    path = tmp_path / "tree.json"
    save_tree_to_json(tree, str(path))
    assert load_tree_from_json(str(path)) == tree


def test_empty_solutions_rejected():
    domain = generate_all_permutations()[:5]
    with pytest.raises(ValueError):
        build_decision_tree(EntropyStrategy(domain, FeedbackLookup(domain)), [])

def test_first_move_is_forced_at_root_only(tree):
    domain = generate_all_permutations()[:40]
    forced = build_decision_tree(EntropyStrategy(domain, FeedbackLookup(domain)), domain, first_move=domain[9])
    assert forced["guess"] == domain[9]
    assert sum(1 for _ in forced["children"]) >= 1