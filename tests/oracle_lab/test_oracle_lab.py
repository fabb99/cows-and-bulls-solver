import math

import numpy as np
import pytest

from experiments.oracle_lab.config import EXACT_SOURCE
from experiments.oracle_lab.data_miner import mine_tree
from experiments.oracle_lab.find_optimal_splits import find_best_single_split
from experiments.oracle_lab.models import FeatureRecord, RawState
from experiments.oracle_lab.regression import cross_validate, fit_linear, predict
from experiments.oracle_lab.storage import load_raw_states, save_raw_states
from experiments.oracle_lab.train_model import train_segmented_model
from src.analysis.metrics import average_depth
from src.analysis.tree_builder import build_decision_tree
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.core.state import state_id
from src.strategies.entropy import EntropyStrategy


def test_legacy_raw_state_is_exact_only_for_exact_source():
    legacy = {"solutions": ["0123", "0132", "0213"], "best_cost": 2.0}
    assert RawState.from_dict({**legacy, "source_tree": EXACT_SOURCE}).is_exact is True
    assert RawState.from_dict({**legacy, "source_tree": "max_entropy"}).is_exact is False


def test_storage_roundtrip(tmp_path):
    states = {"a": RawState(["0123", "0132", "0213"], 2.0, "t", True)}
    path = tmp_path / "raw.json"
    save_raw_states(states, path)
    assert load_raw_states(path) == states
    assert load_raw_states(tmp_path / "missing.json") == {}


def test_mine_tree_keeps_first_source_on_tie():
    domain = generate_all_permutations()[:40]
    tree = build_decision_tree(EntropyStrategy(domain, FeedbackLookup(domain)), domain)
    raw: dict = {}
    mine_tree(tree, raw, "a", min_n=3, max_n=40)
    size = len(raw)
    assert raw[state_id(domain)].best_cost == pytest.approx(average_depth(tree))
    mine_tree(tree, raw, "b", min_n=3, max_n=40)
    assert len(raw) == size
    assert all(s.source_tree == "a" for s in raw.values())


@pytest.mark.parametrize("alpha", [0.5, 0.8])
def test_fit_recovers_noiseless_coefficients(alpha):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 1.5 + 0.7 * X[:, 0] - 0.3 * X[:, 1]
    assert fit_linear(X, y, alpha) == pytest.approx([1.5, 0.7, -0.3], abs=1e-8)


def test_expectile_overestimates_on_noisy_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 1))
    y = 1.0 + 2.0 * X[:, 0] + rng.normal(size=2000)
    mean_residual = lambda a: float(np.mean(y - predict(X, fit_linear(X, y, a))))
    assert abs(mean_residual(0.5)) < 1e-9
    assert mean_residual(0.8) < -0.2  # sovrastima: la sottostima costa di più


def test_cross_validate_is_deterministic_and_validates():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 2))
    y = X @ np.array([1.0, 2.0]) + rng.normal(scale=0.1, size=100)
    assert cross_validate(X, y, seed=3) == cross_validate(X, y, seed=3)
    assert cross_validate(X, y).r2 > 0.99
    with pytest.raises(ValueError):
        cross_validate(X[:1], y[:1])


def _synthetic_records():
    rng = np.random.default_rng(0)
    records = []
    for i in range(400):
        n = int(rng.integers(3, 120))
        feats = {"log_n": math.log2(n), "column_entropy": float(rng.uniform(0, 2))}
        cost = 1.0 + 0.3 * feats["log_n"] - 0.1 * feats["column_entropy"]
        records.append(FeatureRecord(f"s{i}", n, feats, cost, i % 2 == 0, "t"))
    return records


def test_train_segmented_model_end_to_end(tmp_path):
    model = train_segmented_model(
        _synthetic_records(), ["log_n", "column_entropy"], [13, 43], dataset_sha256="x"
    )
    assert len(model.splits) == 2 and len(model.segments) == 3
    for seg in model.segments:
        assert seg.features == ("log_n", "column_entropy")
        assert seg.coef == pytest.approx((0.3, -0.1), abs=1e-6)
    assert model.meta["segments"][0]["cv"]["r2"] > 0.999
    path = tmp_path / "m.json"
    model.save(path)
    assert path.exists()


def test_train_rejects_unknown_feature_and_tiny_segment():
    recs = _synthetic_records()
    with pytest.raises(ValueError):
        train_segmented_model(recs, ["nope"], [])
    with pytest.raises(ValueError):
        train_segmented_model(recs, ["log_n"], [1000])  # segmento vuoto


def test_find_best_single_split_finds_step():
    n = np.arange(1, 201)
    costs = np.where(n <= 80, 2.0, 5.0)
    threshold, gain, error = find_best_single_split(n, costs, min_leaf_samples=10)
    assert threshold == 80 and gain == pytest.approx(1.0) and error == pytest.approx(0.0)

def test_hierarchical_gains_are_relative_to_total_sse():
    from experiments.oracle_lab.find_optimal_splits import find_hierarchical_splits

    n = np.arange(1, 301)
    costs = np.where(n <= 100, 1.0, np.where(n <= 200, 3.0, 4.0))
    splits = find_hierarchical_splits(n, costs, max_splits=2)
    assert [t for t, _ in splits] == [100, 200]
    assert sum(g for _, g in splits) == pytest.approx(1.0)
    assert all(0 < g < 1 for _, g in splits)