import pytest

from src.estimators.segmented import SegmentedLinearEstimator

SOLS = ["0123", "0132", "0213", "0231", "0312"]


def _toy() -> SegmentedLinearEstimator:
    return SegmentedLinearEstimator(
        features=["log_n"],
        splits=[3],
        segments=[{"intercept": 10.0, "coef": [0.0]}, {"intercept": 20.0, "coef": [0.0]}],
    )


def test_exact_small_buckets():
    est = _toy()
    assert est.estimate_cost(["0123"]) == 1.0
    assert est.estimate_cost(["0123", "0132"]) == 1.5


def test_segment_boundaries():
    est = _toy()
    assert est.estimate_cost(SOLS[:3]) == 10.0  # N == soglia -> primo segmento
    assert est.estimate_cost(SOLS[:4]) == 20.0  # N > soglia -> secondo segmento


def test_json_roundtrip(tmp_path):
    est = SegmentedLinearEstimator.legacy_piecewise()
    path = tmp_path / "model.json"
    est.save(path)
    loaded = SegmentedLinearEstimator.load(path)
    assert loaded.to_dict() == est.to_dict()
    assert loaded.estimate_cost(SOLS) == est.estimate_cost(SOLS)


def test_floor_at_one():
    est = SegmentedLinearEstimator(["log_n"], [], [{"intercept": -5.0, "coef": [0.0]}])
    assert est.estimate_cost(SOLS) == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(features=["nope"], splits=[], segments=[{"intercept": 0, "coef": [0]}]),
        dict(features=["log_n"], splits=[5, 5], segments=[{"intercept": 0, "coef": [0]}] * 3),
        dict(features=["log_n"], splits=[5], segments=[{"intercept": 0, "coef": [0]}]),
        dict(features=["log_n"], splits=[], segments=[{"intercept": 0, "coef": [0, 1]}]),
    ],
)
def test_invalid_models_rejected(kwargs):
    with pytest.raises(ValueError):
        SegmentedLinearEstimator(**kwargs)