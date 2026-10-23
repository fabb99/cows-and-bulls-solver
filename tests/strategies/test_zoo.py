from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.estimators.segmented import SegmentedLinearEstimator
from src.strategies.memoized import MemoizedStrategy
from src.strategies.zoo import build_base_strategies

MODEL_DEPENDENT = {
    "estimated_evm_piecewise", "estimated_evm_geometric",
    "estimated_evm_selected", "hybrid_evm_minimax",
}


def _env():
    domain = generate_all_permutations()[:20]
    return domain, FeedbackLookup(domain)


def test_strict_without_models_has_no_model_strategies(tmp_path):
    domain, lookup = _env()
    s = build_base_strategies(domain, lookup, tmp_path, strict=True)
    assert {"max_entropy", "greedy_minimax", "estimated_evm_legacy"} <= set(s)
    assert not MODEL_DEPENDENT & set(s)


def test_non_strict_falls_back_to_legacy_but_never_for_selected(tmp_path):
    domain, lookup = _env()
    s = build_base_strategies(domain, lookup, tmp_path, strict=False)
    assert {"estimated_evm_piecewise", "estimated_evm_geometric", "hybrid_evm_minimax"} <= set(s)
    assert "estimated_evm_selected" not in s


def test_selected_model_is_loaded_and_memoized(tmp_path):
    domain, lookup = _env()
    model = SegmentedLinearEstimator.legacy_piecewise()
    model.save(tmp_path / "selected.json")
    s = build_base_strategies(domain, lookup, tmp_path, strict=True, memoize=True)
    assert all(isinstance(v, MemoizedStrategy) for v in s.values())
    assert s["estimated_evm_selected"].inner.estimator.to_dict() == model.to_dict()