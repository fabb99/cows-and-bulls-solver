"""
Modulo: investigate_branch (Script)
Diagnostica di un ramo del primo turno: confronta, per ogni albero, la mossa del Turno 2,
come partiziona le soluzioni del ramo, cosa prevedeva l'oracolo e come finiscono le partite.

Uso: python -m experiments.investigate_branch --feedback=- [--trees A B ...] [--duel A B]
     (feedback: '+' toro, '-' mucca, '0' nessuno; sempre con '=' per i valori che iniziano per '-')
"""
import argparse
import math
from pathlib import Path
from typing import Any, Optional

from src.analysis.metrics import average_depth, get_depth_distribution
from src.analysis.tree_io import load_tree_from_json
from src.cli.game_loop import fb_tuple_to_string, parse_feedback
from src.core.engine import calculate_buckets
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.estimators.base import BaseBucketEstimator
from src.paths import MODELS_DIR, TREES_DIR
from src.strategies.costs import expected_cost
from src.strategies.zoo import load_estimator


def analyze_guess_partition(
    guess: str, solutions: list[str], lookup: FeedbackLookup, estimator: BaseBucketEstimator
) -> dict[str, Any]:
    """
    PRE:  guess e solutions appartengono al dominio di lookup.
    POST: entropia, N_max, numero di bucket, costo previsto dall'oracolo e bucket.
    """
    n = len(solutions)
    buckets = calculate_buckets(guess, solutions, lookup)
    entropy = -sum((len(b) / n) * math.log2(len(b) / n) for b in buckets.values())
    return {
        "guess": guess,
        "is_valid": guess in set(solutions),
        "entropy": entropy,
        "n_max": max(len(b) for b in buckets.values()),
        "total_buckets": len(buckets),
        "predicted_cost": expected_cost(buckets, n, estimator.estimate_cost),
        "buckets": buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Autopsia di un ramo del primo turno.")
    parser.add_argument("--feedback", required=True, help="es. --feedback=- oppure --feedback=+-")
    parser.add_argument("--trees", nargs="+", default=None)
    parser.add_argument("--dir", type=Path, default=TREES_DIR)
    parser.add_argument("--duel", nargs=2, metavar=("A", "B"), default=None)
    args = parser.parse_args()

    target = parse_feedback(args.feedback)
    if target is None:
        raise SystemExit(f"Feedback non valido: {args.feedback!r}")
    label = fb_tuple_to_string(target) or "0"

    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    estimator = load_estimator(MODELS_DIR, "piecewise")

    results: dict[str, dict[str, Any]] = {}
    n_solutions: Optional[int] = None
    for path in sorted(args.dir.glob("*.json")):
        if args.trees and path.stem not in args.trees:
            continue
        child = load_tree_from_json(str(path))["children"].get(target)
        if child is None:
            continue
        solutions = child["possible_solutions"]
        n_solutions = n_solutions or len(solutions)
        results[path.stem] = {
            "real_avg": average_depth(child),
            "dist": get_depth_distribution(child),
            "partition": analyze_guess_partition(child["guess"], solutions, lookup, estimator),
        }
    if not results:
        raise SystemExit(f"Nessun albero con il ramo {label!r} in {args.dir}")

    width = 96
    print("=" * width)
    print(f" RAMO {target} [feedback: {label}] - {n_solutions} SOLUZIONI")
    print("=" * width)
    print(f"{'Strategia':<28} | {'Mossa T2':<8} | {'Valida':<6} | {'Entropia':<8} | {'N_max':<5} | {'Stima':<8} | {'Media reale':<11}")
    print("-" * width)
    for name, r in results.items():
        p = r["partition"]
        print(
            f"{name:<28} | {p['guess']:<8} | {'sì' if p['is_valid'] else 'no':<6} | "
            f"{p['entropy']:<8.4f} | {p['n_max']:<5} | {p['predicted_cost']:<8.4f} | {r['real_avg']:<11.5f}"
        )

    max_rel = max(max(r["dist"]) for r in results.values())
    print("\n" + "=" * width)
    print(" DISTRIBUZIONE DELLE PARTITE DEL RAMO (mosse dopo il primo turno)")
    print("=" * width)
    print(f"{'Strategia':<28} | " + " | ".join(f"+{i:<5}" for i in range(1, max_rel + 1)))
    for name, r in results.items():
        print(f"{name:<28} | " + " | ".join(f"{r['dist'].get(i, 0):<6}" for i in range(1, max_rel + 1)))

    if args.duel:
        a, b = args.duel
        if a not in results or b not in results:
            raise SystemExit(f"Duello impossibile: {a} o {b} non presenti tra i risultati")
        ba, bb = results[a]["partition"]["buckets"], results[b]["partition"]["buckets"]
        print(f"\nDUELLO BUCKET: {a} ({results[a]['partition']['guess']}) vs {b} ({results[b]['partition']['guess']})")
        print(f"{'Feedback':<10} | {a[:18]:<18} | {b[:18]:<18} | Differenza")
        print("-" * 62)
        for fb in sorted(set(ba) | set(bb)):
            la, lb = len(ba.get(fb, [])), len(bb.get(fb, []))
            print(f"{fb_tuple_to_string(fb) or '0':<10} | {la:<18} | {lb:<18} | {lb - la:+d}")


if __name__ == "__main__":
    main()