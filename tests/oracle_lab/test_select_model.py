import math

import numpy as np
import pytest

from experiments.oracle_lab.models import FeatureRecord
from experiments.oracle_lab.select_model import (
    FEATURE_NAMES,
    MIXED_GROUP,
    CVTables,
    PartitionSearch,
    build_estimator,
    choose_k,
    make_folds,
    prepare,
    root_branch_group,
)
from src.core.feedback import calculate_feedback
from src.core.generator import generate_all_permutations
from src.features.bucket_features import FEATURES


def _group(record):
    return int(record.state_id[1:]) % 10


def _synthetic(seed=0):
    """Struttura vera: fase A (N<=42) usa column_entropy, B (42<N<=150) digit_std, C anagram_ratio."""
    rng = np.random.default_rng(seed)
    records, i = [], 0
    for n in range(6, 301, 6):
        for _ in range(12):
            feats = {name: float(rng.uniform(0, 2)) for name in FEATURES}
            feats["log_n"] = math.log2(n)
            if n <= 42:
                y = 1.0 + 0.5 * feats["column_entropy"]
            elif n <= 150:
                y = 2.0 + 1.0 * feats["digit_std"]
            else:
                y = 3.0 - 0.8 * feats["anagram_ratio"]
            y += float(rng.normal(scale=0.01))
            records.append(FeatureRecord(f"s{i}", n, feats, y, True, "t"))
            i += 1
    return records


@pytest.fixture(scope="module")
def synthetic():
    ds = prepare(_synthetic(), group_fn=_group)
    folds = make_folds(ds.groups, k=5, seed=0)
    return ds, folds


def test_dp_recovers_known_structure(synthetic):
    ds, folds = synthetic
    search = PartitionSearch(ds, folds, min_samples=50, n_candidates=50, max_features=2)
    partitions = search.best_partitions(4)
    k = choose_k(partitions, tol_k=0.02)
    assert k == 3
    part = partitions[k]
    assert [f.hi for f in part[:-1]] == [42, 150]
    assert [f.features for f in part] == [
        ("column_entropy",),
        ("digit_std",),
        ("anagram_ratio",),
    ]
    model = build_estimator(ds, part)
    assert model.splits == [42, 150]
    assert model.segments[0].coef == pytest.approx((0.5,), abs=0.02)
    assert model.segments[1].coef == pytest.approx((1.0,), abs=0.02)
    assert model.segments[2].coef == pytest.approx((-0.8,), abs=0.02)


def test_closed_form_matches_bruteforce(synthetic):
    ds, folds = synthetic
    tables = CVTables(ds, folds)
    subset = (0, 2)
    fast = tables.subset_sse(0, len(ds.y), [subset])[0]

    Z = np.column_stack([np.ones(len(ds.y)), ds.F[:, list(subset)]])
    brute = 0.0
    for f in range(tables.k):
        test = folds == f
        beta = np.linalg.lstsq(Z[~test], ds.y[~test], rcond=None)[0]
        brute += float(np.sum((ds.y[test] - Z[test] @ beta) ** 2))
    assert fast == pytest.approx(brute, rel=1e-4)


def test_make_folds_keeps_groups_together_and_is_deterministic():
    groups = ["a"] * 100 + ["b"] * 60 + ["c"] * 50 + ["d"] * 40 + ["e"] * 10
    folds = make_folds(groups, k=3, seed=1)
    assert (folds == make_folds(groups, k=3, seed=1)).all()
    assert set(folds) == {0, 1, 2}
    for g in set(groups):
        assert len({folds[i] for i, x in enumerate(groups) if x == g}) == 1
    with pytest.raises(ValueError):
        make_folds(["a"] * 10, k=3)


def test_root_branch_group():
    perms = generate_all_permutations()
    same = [p for p in perms if calculate_feedback("0123", p) == (0, 1)][:4]
    record = lambda sols: FeatureRecord(",".join(sorted(sols)), len(sols), {}, 1.0, False, "t")
    assert root_branch_group(record(same)) == (0, 1)
    assert root_branch_group(record(perms[:3])) == MIXED_GROUP


def test_feature_names_match_registry():
    assert FEATURE_NAMES == list(FEATURES)