import random

import pytest

from experiments.oracle_lab.augment_states import (
    EntropyCostOracle,
    augment,
    bin_index,
    play_epsilon_game,
)
from src.analysis.metrics import average_depth
from src.analysis.tree_builder import build_decision_tree
from src.core.constants import WIN_FEEDBACK
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.core.state import state_id
from src.strategies.entropy import EntropyStrategy
from src.strategies.memoized import MemoizedStrategy

BINS = [(3, 10), (11, 40), (41, 100)]


@pytest.fixture(scope="module")
def env():
    domain = generate_all_permutations()[:120]
    lookup = FeedbackLookup(domain)
    strategy = MemoizedStrategy(EntropyStrategy(domain, lookup))
    return domain, lookup, strategy, EntropyCostOracle(strategy, lookup)


def test_bin_index():
    assert bin_index(2) is None and bin_index(3) == 0 and bin_index(10) == 0
    assert bin_index(11) == 1 and bin_index(1440) == 4 and bin_index(1441) is None


def test_oracle_cost_equals_tree_average(env):
    domain, lookup, strategy, oracle = env
    for sols in (domain[:3], domain[10:25], domain[:80]):
        tree = build_decision_tree(strategy, sols)
        assert oracle.cost(sols) == pytest.approx(average_depth(tree))


def test_game_trajectory(env):
    domain, lookup, strategy, _ = env
    secret = domain[77]
    traj = play_epsilon_game(secret, strategy, domain, lookup, 0.0, random.Random(0))
    assert traj[0][1] == domain[0] and traj[0][0] == domain
    assert all(secret in sols for sols, _ in traj)
    assert lookup.get_feedback(traj[-1][1], secret) == WIN_FEEDBACK
    for sols, guess in traj[1:]:  # epsilon = 0: sempre la strategia
        assert guess == (sols[0] if len(sols) <= 2 else strategy.choose_guess(sols, ["-"]))


def _run(env, seed, m):
    domain, lookup, strategy, oracle = env
    raw = {}
    stats = augment(raw, strategy, oracle, domain, lookup, quotas=[20, 20, 20], bins=BINS,
                    epsilon=0.3, m_buckets=m, max_n=100, max_games=300, seed=seed)
    return raw, stats


def test_augment_quotas_ranges_and_determinism(env):
    raw1, stats = _run(env, 0, 3)
    raw2, _ = _run(env, 0, 3)
    assert sorted(raw1) == sorted(raw2)
    assert all(c <= 20 for c in stats.added_by_bin)
    assert sum(stats.added_by_bin) == len(raw1) == stats.added_a + stats.added_b
    assert stats.added_b > 0
    for sid, state in raw1.items():
        assert sid == state_id(state.solutions) and 3 <= state.n <= 100
        assert state.best_cost >= 1.0 and state.is_exact is False

def test_augment_skips_existing_states(env):
    domain, lookup, strategy, oracle = env
    raw, _ = _run(env, 1, 0)
    snapshot = {sid: (s.best_cost, s.source_tree) for sid, s in raw.items()}
    augment(raw, strategy, oracle, domain, lookup, quotas=[20, 20, 20], bins=BINS,
            epsilon=0.3, m_buckets=0, max_n=100, max_games=300, seed=1)
    # gli stati preesistenti restano identici (nessuna sovrascrittura)
    for sid, (cost, source) in snapshot.items():
        assert raw[sid].best_cost == cost and raw[sid].source_tree == source
    # e lo stato non cresce oltre le quote: al più 20 nuovi per fascia
    assert len(raw) - len(snapshot) <= 3 * 20
    assert len(raw) == len({state_id(s.solutions) for s in raw.values()})