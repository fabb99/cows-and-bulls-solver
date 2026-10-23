import pytest

from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.strategies.entropy import EntropyStrategy
from src.strategies.memoized import MemoizedStrategy


@pytest.fixture(scope="module")
def env():
    domain = generate_all_permutations()[:120]
    lookup = FeedbackLookup(domain)
    return domain, lookup, EntropyStrategy(domain, lookup)


def test_same_choices_and_cache_hits(env):
    domain, _, inner = env
    memo = MemoizedStrategy(inner)
    for sols in (domain[:30], domain[40:70]):
        assert memo.choose_guess(sols, ["x"]) == inner.choose_guess(sols, ["x"])
    assert (memo.hits, memo.misses) == (0, 2)
    assert memo.choose_guess(list(reversed(domain[:30])), ["y"]) == inner.choose_guess(domain[:30], ["x"])
    assert memo.hits == 1


def test_turn_one_is_never_cached(env):
    domain, _, inner = env
    memo = MemoizedStrategy(inner)
    assert memo.choose_guess(domain, []) == inner.choose_guess(domain, [])
    assert (memo.hits, memo.misses) == (0, 0)


def test_small_states_and_empty(env):
    domain, _, inner = env
    memo = MemoizedStrategy(inner)
    assert memo.choose_guess(domain[:2], []) == domain[0]
    with pytest.raises(ValueError):
        memo.choose_guess([], [])
    assert memo.higher_is_better == inner.higher_is_better