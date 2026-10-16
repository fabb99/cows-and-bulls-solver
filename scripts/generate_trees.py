"""
Modulo: generate_trees (Script)
Genera e salva gli alberi decisionali delle strategie scelte in TREES_DIR.

Uso: python -m scripts.generate_trees --strategies max_entropy greedy_minimax [--force]
"""
import argparse
import time

from tqdm import tqdm

from src.analysis.metrics import average_depth, max_depth
from src.analysis.tree_builder import build_decision_tree
from src.analysis.tree_io import save_tree_to_json
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.paths import MODELS_DIR, ROUTING_POLICY_FILEPATH, TREES_DIR
from src.strategies.zoo import (
    ROUTED_NAMES,
    build_base_strategies,
    build_routed_strategies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera gli alberi decisionali.")
    parser.add_argument("--strategies", nargs="+", required=True)
    parser.add_argument("--force", action="store_true", help="rigenera anche se esiste già")
    args = parser.parse_args()

    start = time.time()
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    print(f"Sistema pronto ({time.time() - start:.1f} s)\n")

    available = build_base_strategies(all_perms, lookup, MODELS_DIR)
    available.update(build_routed_strategies(all_perms, lookup, available, ROUTING_POLICY_FILEPATH))
    unknown = [n for n in args.strategies if n not in available]
    if unknown:
        hint = " (esegui prima train_router)" if set(unknown) & set(ROUTED_NAMES) else ""
        raise SystemExit(f"Strategie non disponibili: {unknown}{hint}. Disponibili: {sorted(available)}")

    TREES_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.strategies:
        path = TREES_DIR / f"{name}.json"
        if path.exists() and not args.force:
            print(f"{name}: già presente, salto (usa --force)")
            continue
        with tqdm(total=len(all_perms), desc=name, unit="partite") as bar:
            tree = build_decision_tree(
                available[name], all_perms, on_victory_callback=bar.update
            )
        save_tree_to_json(tree, str(path))
        print(f"{name}: media {average_depth(tree):.5f} | caso peggiore {max_depth(tree)} -> {path}")


if __name__ == "__main__":
    main()