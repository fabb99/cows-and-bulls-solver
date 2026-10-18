"""
Modulo: zoo
Costruzione centralizzata delle strategie con nome, usata da generate_trees,
dataset_refiner, train_router, tune_hybrid_weights, investigate_branch e main.

Modalità `strict` (usata dalla pipeline): una strategia che dipende da un modello
viene costruita SOLO se il file del modello esiste; nessun fallback sui coefficienti legacy.
"""
import json
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
from src.strategies.memoized import MemoizedStrategy

MODEL_FILES = {
    "geometric": "geometric.json",
    "piecewise": "piecewise.json",  # soglie fisse 13/43
    "selected": "selected.json",  # output di select_model (soglie e feature per fase)
}
ENTROPY_MINIMAX_WEIGHTS = (0.94, 0.06)
EVM_MINIMAX_WEIGHTS = (0.944, 0.056)
GEOMETRIC_MINIMAX_WEIGHTS = (0.962, 0.038)
ROUTED_NAMES = {
    "frankenstein_a_strategy": "best_average",
    "frankenstein_d_strategy": "pareto_defensive",
}


def load_estimator(models_dir: Path, kind: str) -> SegmentedLinearEstimator:
    """
    PRE:  kind in MODEL_FILES.
    POST: il modello di models_dir/<kind>.json. Se manca: per 'geometric' e 'piecewise'
          i coefficienti legacy_v1 (con avviso su stderr); per 'selected' FileNotFoundError.
    """
    path = models_dir / MODEL_FILES[kind]
    if path.exists():
        return SegmentedLinearEstimator.load(path)
    if kind == "selected":
        raise FileNotFoundError(path)
    print(f"ATTENZIONE: {path} assente, uso i coefficienti legacy_v1", file=sys.stderr)
    if kind == "geometric":
        return SegmentedLinearEstimator.legacy_geometric()
    return SegmentedLinearEstimator.legacy_piecewise()


def _try_load(models_dir: Path, kind: str, strict: bool) -> SegmentedLinearEstimator | None:
    """POST: il modello se il file esiste; altrimenti None (strict o 'selected') o legacy."""
    path = models_dir / MODEL_FILES[kind]
    if path.exists():
        return SegmentedLinearEstimator.load(path)
    if strict or kind == "selected":
        return None
    return load_estimator(models_dir, kind)


def build_base_strategies(
    all_perms: list[str],
    lookup: FeedbackLookup,
    models_dir: Path,
    strict: bool = False,
    memoize: bool = False,
) -> dict[str, BaseStrategy]:
    """
    POST: {nome: strategia} per le strategie non instradate.
          Sempre presenti: naive, max_frequency, greedy_minimax, max_entropy,
          hybrid_entropy_minimax, estimated_evm_legacy.
          Dipendenti da modello (solo se il file esiste, o se non strict per i legacy):
          estimated_evm_{piecewise,geometric,selected}, hybrid_evm_minimax.
          memoize=True avvolge ogni strategia in MemoizedStrategy.
    """
    entropy = EntropyStrategy(all_perms, lookup)
    minimax = GreedyMinimaxStrategy(all_perms, lookup)
    strategies: dict[str, BaseStrategy] = {
        "naive": NaiveStrategy(all_perms, lookup),
        "max_frequency": MaxFrequencyStrategy(all_perms, lookup),
        "greedy_minimax": minimax,
        "max_entropy": entropy,
        "hybrid_entropy_minimax": HybridWeightedStrategy(entropy, minimax, *ENTROPY_MINIMAX_WEIGHTS),
        "estimated_evm_legacy": EstimatedEVMStrategy(
            all_perms, lookup, SegmentedLinearEstimator.legacy_piecewise()
        ),
    }
    for kind in MODEL_FILES:
        estimator = _try_load(models_dir, kind, strict)
        if estimator is None:
            continue
        evm = EstimatedEVMStrategy(all_perms, lookup, estimator)
        strategies[f"estimated_evm_{kind}"] = evm
        if kind == "piecewise":
            strategies["hybrid_evm_minimax"] = HybridWeightedStrategy(
                evm, minimax, *EVM_MINIMAX_WEIGHTS
            )
        if kind == "geometric":
            strategies["hybrid_geometric_minimax"] = HybridWeightedStrategy(
                evm, minimax, *GEOMETRIC_MINIMAX_WEIGHTS
            )
    hybrid_spec = models_dir / "hybrid_best.json"
    if hybrid_spec.exists():
        spec = json.loads(hybrid_spec.read_text(encoding="utf-8"))
        base_evm = strategies.get(f"estimated_evm_{spec.get('base_model')}")
        if not spec.get("skipped") and base_evm is not None:
            strategies["hybrid_best_minimax"] = HybridWeightedStrategy(
                base_evm, minimax, spec["weight_a"], spec["weight_b"]
            )
    if memoize:
        strategies = {name: MemoizedStrategy(s) for name, s in strategies.items()}
    return strategies


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