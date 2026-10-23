import math

from src.features.bucket_features import FEATURES, compute_features

SOLS = ["0123", "0132"]


def test_known_values():
    assert FEATURES["log_n"](SOLS) == 1.0
    assert FEATURES["unique_digits_ratio"](SOLS) == 0.4
    assert FEATURES["column_entropy"](SOLS) == 0.5
    assert math.isclose(FEATURES["digit_std"](SOLS), math.sqrt(0.96))
    assert FEATURES["anagram_ratio"](SOLS) == 0.5
    assert FEATURES["internal_cohesion"](SOLS) == 0.75


def test_compute_features_preserves_order():
    assert compute_features(SOLS, ["column_entropy", "log_n"]) == [0.5, 1.0]