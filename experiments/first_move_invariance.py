"""
Modulo: first_move_invariance (Script)
Verifica empirica: la prestazione di una strategia dipende dalla prima mossa?
Per simmetria (rinominare le cifre) i bucket sono isomorfi e le strategie basate su
quantità invarianti dovrebbero dare la stessa media. Il tie-break lessicografico di
select_best NON è equivariante: eventuali differenze piccole vanno attribuite a lui.

Uso: python -m experiments.first_move_invariance --strategies max_entropy --random 4
"""
import argparse
import random

from src.analysis.metrics import average_depth, get_depth_distribution, max_depth
from src.analysis.tree_builder import build_decision_tree
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.paths import MODELS_DIR
from src.strategies.zoo import build_base_strategies


def main() -> None:
    parser = argparse.ArgumentParser(description="Invarianza rispetto alla prima mossa.")
    parser.add_argument("--strategies", nargs="+", default=["max_entropy"])
    parser.add_argument("--first-moves", nargs="+", default=["0123", "8901", "4567", "9876"])
    parser.add_argument("--random", type=int, default=0, help="prime mosse casuali aggiuntive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    base = build_base_strategies(all_perms, lookup, MODELS_DIR.__class__(args.models_dir),
                                 strict=True, memoize=True)
    unknown = [s for s in args.strategies if s not in base]
    if unknown:
        raise SystemExit(f"Strategie non disponibili: {unknown}. Disponibili: {sorted(base)}")

    moves = list(args.first_moves)
    moves += random.Random(args.seed).sample(all_perms, args.random)

    for name in args.strategies:
        print(f"\n=== {name} ===")
        print(f"{'Prima mossa':<12} | {'Media':<9} | {'Peggiore':<8} | Distribuzione")
        averages = []
        for move in moves:
            tree = build_decision_tree(base[name], all_perms, first_move=move)
            avg = average_depth(tree)
            averages.append(avg)
            dist = get_depth_distribution(tree)
            print(f"{move:<12} | {avg:<9.5f} | {max_depth(tree):<8} | {list(dist.values())}")
        print(f"Differenza max tra medie: {max(averages) - min(averages):.5f}")


if __name__ == "__main__":
    main()