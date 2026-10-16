"""
Modulo: tune_hybrid_weights (Script)
Ricerca adattiva (coarse -> zoom -> micro) del peso ottimale di una HybridWeightedStrategy
tra due strategie. Trova due campioni:
  1. Media assoluta (minima media di mosse)
  2. Difensivo di Pareto (minimo caso peggiore, a parità la media più bassa)
Ogni valutazione costruisce l'albero completo: è lenta.

Uso: python -m experiments.tune_hybrid_weights --a max_entropy --b greedy_minimax
"""
import argparse
import json
from typing import Any

import numpy as np
from tqdm import tqdm

from src.analysis.metrics import average_depth, max_depth
from src.analysis.tree_builder import build_decision_tree
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.paths import MODELS_DIR
from src.strategies.base import BaseStrategy
from src.strategies.hybrid import HybridWeightedStrategy
from src.strategies.zoo import build_base_strategies

TUNABLE = ["max_entropy", "greedy_minimax", "estimated_evm_piecewise", "estimated_evm_geometric"]


def evaluate_weight(
    a: BaseStrategy, b: BaseStrategy, weight_a: float, all_perms: list[str]
) -> tuple[int, float]:
    """PRE: 0 <= weight_a <= 1. POST: (caso peggiore, media mosse) dell'ibrida (w, 1-w)."""
    hybrid = HybridWeightedStrategy(a, b, weight_a, round(1.0 - weight_a, 4))
    tree = build_decision_tree(hybrid, all_perms)
    return max_depth(tree), average_depth(tree)


def extract_champions(tested: dict[float, tuple[int, float]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """PRE: tested non vuoto. POST: (campione della media, campione difensivo)."""

    def describe(w: float) -> dict[str, Any]:
        worst, avg = tested[w]
        return {"weight_a": w, "weight_b": round(1.0 - w, 4), "worst_case": worst, "avg_moves": round(avg, 5)}

    best_avg = min(tested, key=lambda w: (tested[w][1], w))
    min_worst = min(worst for worst, _ in tested.values())
    best_def = min((w for w, (worst, _) in tested.items() if worst == min_worst), key=lambda w: (tested[w][1], w))
    return describe(best_avg), describe(best_def)


def run_adaptive_search(
    a: BaseStrategy, b: BaseStrategy, all_perms: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """POST: (campione della media, campione difensivo) dopo 3 livelli di zoom."""
    tested: dict[float, tuple[int, float]] = {}

    def run_phase(title: str, weights) -> None:
        todo = sorted({round(float(w), 4) for w in weights} - set(tested))
        with tqdm(todo, desc=title) as pbar:
            for w in pbar:
                tested[w] = evaluate_weight(a, b, w, all_perms)
                pbar.set_postfix_str(f"w_a={w:.3f} worst={tested[w][0]} avg={tested[w][1]:.4f}")

    def around(step: float, radius: float):
        points = []
        for champ in extract_champions(tested):
            lo, hi = max(0.5, champ["weight_a"] - radius), min(1.0, champ["weight_a"] + radius)
            points.extend(np.arange(lo, hi + step / 10, step))
        return points

    run_phase("Fase 1: coarse (0.05)", np.arange(0.50, 1.0001, 0.05))
    run_phase("Fase 2: zoom (0.01)", around(0.01, 0.04))
    run_phase("Fase 3: micro (0.002)", around(0.002, 0.008))
    return extract_champions(tested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ottimizza i pesi di una strategia ibrida.")
    parser.add_argument("--a", choices=TUNABLE, required=True)
    parser.add_argument("--b", choices=TUNABLE, required=True)
    args = parser.parse_args()
    if args.a == args.b:
        raise SystemExit("--a e --b devono essere diverse")

    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    base = build_base_strategies(all_perms, lookup, MODELS_DIR)

    champ_avg, champ_def = run_adaptive_search(base[args.a], base[args.b], all_perms)

    for title, champ in (("CAMPIONE MEDIO", champ_avg), ("CAMPIONE DIFENSIVO", champ_def)):
        print(f"\n{title}: peso {args.a} = {champ['weight_a']:.4f}, peso {args.b} = {champ['weight_b']:.4f}")
        print(f"  media {champ['avg_moves']:.5f} | caso peggiore {champ['worst_case']}")

    out = MODELS_DIR / f"hybrid_{args.a}__{args.b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump(
            {"strategy_a": args.a, "strategy_b": args.b,
             "champion_best_average": champ_avg, "champion_pareto_defensive": champ_def},
            file, indent=4,
        )
    print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()