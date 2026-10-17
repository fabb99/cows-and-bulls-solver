import pytest

from src.estimators.segmented import Segment, SegmentedLinearEstimator
from src.features.bucket_features import FEATURES

SOLS = ["0123", "0132", "0213", "0231", "0312"]


def _toy() -> SegmentedLinearEstimator:
    return SegmentedLinearEstimator.from_common_features(
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


def test_each_segment_uses_its_own_features():
    est = SegmentedLinearEstimator(
        splits=[3],
        segments=[
            {"features": ["log_n"], "intercept": 1.0, "coef": [2.0]},
            {"features": ["column_entropy", "digit_std"], "intercept": 3.0, "coef": [0.5, -1.0]},
        ],
    )
    small, large = SOLS[:3], SOLS
    assert est.estimate_cost(small) == pytest.approx(1.0 + 2.0 * FEATURES["log_n"](small))
    expected = 3.0 + 0.5 * FEATURES["column_entropy"](large) - FEATURES["digit_std"](large)
    assert est.estimate_cost(large) == pytest.approx(max(1.0, expected))


def test_intercept_only_segment():
    est = SegmentedLinearEstimator(
        splits=[], segments=[{"features": [], "intercept": 2.5, "coef": []}]
    )
    assert est.estimate_cost(SOLS) == 2.5


def test_json_roundtrip_v2(tmp_path):
    est = SegmentedLinearEstimator.legacy_piecewise()
    path = tmp_path / "model.json"
    est.save(path)
    loaded = SegmentedLinearEstimator.load(path)
    assert loaded.to_dict() == est.to_dict()
    assert loaded.to_dict()["schema_version"] == 2
    assert loaded.estimate_cost(SOLS) == est.estimate_cost(SOLS)


def test_schema_v1_is_converted():
    v1 = {
        "schema_version": 1,
        "features": ["log_n", "column_entropy"],
        "splits": [3],
        "segments": [
            {"intercept": 1.0, "coef": [0.5, 0.1]},
            {"intercept": 2.0, "coef": [0.3, -0.2]},
        ],
        "alpha": 0.8,
        "meta": {},
    }
    est = SegmentedLinearEstimator.from_dict(v1)
    assert all(seg.features == ("log_n", "column_entropy") for seg in est.segments)
    assert est.alpha == 0.8
    expected = 2.0 + 0.3 * FEATURES["log_n"](SOLS) - 0.2 * FEATURES["column_entropy"](SOLS)
    assert est.estimate_cost(SOLS) == pytest.approx(expected)


def test_floor_at_one():
    est = SegmentedLinearEstimator.from_common_features(
        ["log_n"], [], [{"intercept": -5.0, "coef": [0.0]}]
    )
    assert est.estimate_cost(SOLS) == 1.0


def test_unsupported_schema_version_rejected():
    with pytest.raises(ValueError):
        SegmentedLinearEstimator.from_dict({"schema_version": 99})


def _seg(**over):
    base = {"features": ["log_n"], "intercept": 0.0, "coef": [0.0]}
    return {**base, **over}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"splits": [], "segments": [_seg(features=["nope"])]},  # feature sconosciuta
        {"splits": [], "segments": [_seg(features=["log_n", "log_n"], coef=[0, 0])]},  # duplicata
        {"splits": [5, 5], "segments": [_seg()] * 3},  # soglie non crescenti
        {"splits": [5], "segments": [_seg()]},  # segmenti insufficienti
        {"splits": [], "segments": [_seg(coef=[0, 1])]},  # coefficienti di troppo
    ],
)
def test_invalid_models_rejected(kwargs):
    with pytest.raises(ValueError):
        SegmentedLinearEstimator(**kwargs)


def test_segment_is_immutable():
    seg = Segment(("log_n",), 1.0, (0.5,))
    with pytest.raises(AttributeError):
        seg.intercept = 2.0  # type: ignore[misc]