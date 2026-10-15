import pytest

from src.core.generator import generate_all_permutations
from src.core.signatures import get_representatives
from src.strategies.hybrid import HybridWeightedStrategy
from src.strategies.selection import select_best


def test_tolerance_and_prefers_solution():
    scores = {"0123": 1.0, "0124": 1.0 + 1e-12, "0125": 2.0}
    assert select_best(scores, ["0124"], higher_is_better=False) == "0124"


def test_lexicographic_when_no_solution():
    scores = {"0125": 1.0, "0123": 1.0, "0124": 1.0}
    assert select_best(scores, [], higher_is_better=False) == "0123"


def test_higher_is_better():
    assert select_best({"0123": 1.0, "0124": 3.0}, [], higher_is_better=True) == "0124"


def test_turn_one_has_single_representative():
    perms = generate_all_permutations()
    assert get_representatives(perms, [], perms) == ["0123"]


def test_hybrid_normalization():
    lower = HybridWeightedStrategy._normalize_scores({"a": 2.0, "b": 4.0, "c": float("inf")}, False)
    assert lower == {"a": 1.0, "b": 0.5, "c": 0.0}
    higher = HybridWeightedStrategy._normalize_scores({"a": 0.0, "b": 0.0}, True)
    assert higher == {"a": 0.0, "b": 0.0}
    with pytest.raises(ValueError):
        HybridWeightedStrategy._normalize_scores({"a": 0.0, "b": 1.0}, False)