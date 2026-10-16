"""
Modulo: zoo
Costruzione centralizzata delle strategie con nome, usata da generate_trees,
train_router, tune_hybrid_weights, investigate_branch e main.
"""
import sys
from pathlib import Path

from src.core.lookup_table import FeedbackLookup
from src.estimators.segmented import SegmentedLinearEstimator
from src.strategies.base import BaseStrategy
from src.strategies.basic import MaxFrequencyStrategy, NaiveStrategy
from src.strategies.entropy import EntropyStrategy
from src.strategies.estimated_evm import EstimatedEVMStrategy
from src.strategies.frankenstein import RoutedStrategy
from src.strategies.greedy_minimax import GreedyMinimaxStrategy
from src.strategies.hybrid import HybridWeightedStrategy

MODEL_FILES = {"geometric": "geometric.json", "piecewise": "piecewise.json"}
ENTROPY_MINIMAX_WEIGHTS = (0.94, 0.06)
EVM_MINIMAX_WEIGHTS = (0.944, 0.056)
ROUTED_NAMES = {
    "frankenstein_a_strategy": "best_average",
    "frankenstein_d_strategy": "pareto_defensive",
}


def load_estimator(models_dir: Path, kind: str) -> SegmentedLinearEstimator:
    """
    PRE:  kind in MODEL_FILES.
    POST: il modello di models_dir/<kind>.json; se manca, i coefficienti legacy_v1
          (con avviso su stderr).
    """
    path = models_dir / MODEL_FILES[kind]
    if path.exists():
        return SegmentedLinearEstimator.load(path)
    print(f"ATTENZIONE: {path} assente, uso i coefficienti legacy_v1", file=sys.stderr)
    if kind == "geometric":
        return SegmentedLinearEstimator.legacy_geometric()
    return SegmentedLinearEstimator.legacy_piecewise()


def build_base_strategies(
    all_perms: list[str], lookup: FeedbackLookup, models_dir: Path
) -> dict[str, BaseStrategy]:
    """POST: {nome: strategia} per tutte le strategie non instradate."""
    entropy = EntropyStrategy(all_perms, lookup)
    minimax = GreedyMinimaxStrategy(all_perms, lookup)
    evm_piecewise = EstimatedEVMStrategy(
        all_perms, lookup, load_estimator(models_dir, "piecewise")
    )
    evm_geometric = EstimatedEVMStrategy(
        all_perms, lookup, load_estimator(models_dir, "geometric")
    )
    return {
        "naive": NaiveStrategy(all_perms, lookup),
        "max_frequency": MaxFrequencyStrategy(all_perms, lookup),
        "greedy_minimax": minimax,
        "max_entropy": entropy,
        "estimated_evm_piecewise": evm_piecewise,
        "estimated_evm_geometric": evm_geometric,
        "hybrid_entropy_minimax": HybridWeightedStrategy(entropy, minimax, *ENTROPY_MINIMAX_WEIGHTS),
        "hybrid_evm_minimax": HybridWeightedStrategy(evm_piecewise, minimax, *EVM_MINIMAX_WEIGHTS),
    }


def build_routed_strategies(
    all_perms: list[str],
    lookup: FeedbackLookup,
    base: dict[str, BaseStrategy],
    policy_path: Path,
) -> dict[str, BaseStrategy]:
    """POST: le due strategie Frankenstein, oppure {} se la policy non esiste."""
    if not policy_path.exists():
        return {}
    return {
        name: RoutedStrategy.from_file(
            policy_path, style, all_perms, lookup, fallback=base["max_entropy"]
        )
        for name, style in ROUTED_NAMES.items()
    }