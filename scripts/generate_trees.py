"""
Modulo: generate_trees (Script)
Genera e salva gli alberi decisionali delle strategie scelte in TREES_DIR.

Uso: python -m scripts.generate_trees --strategies max_entropy greedy_minimax [--force]
"""
import argparse
import time
from pathlib import Path

from tqdm import tqdm

from src.analysis.metrics import average_depth, max_depth
from src.analysis.tree_builder import build_decision_tree
from src.analysis.tree_io import save_tree_to_json
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.paths import MODELS_DIR, RECIPE_FILEPATH, ROUTING_POLICY_FILEPATH, TREES_DIR
from src.strategies.zoo import (
    POLICY_ROUTER_NAMES,
    RECIPE_NAMES,
    build_base_strategies,
    build_policy_routers,
    build_recipe_strategies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera gli alberi decisionali.")
    parser.add_argument("--strategies", nargs="+", required=True)
    parser.add_argument("--force", action="store_true", help="rigenera anche se esiste già")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--strict-models", action="store_true",
                        help="nessun fallback legacy: i modelli assenti rendono la strategia non disponibile")
    args = parser.parse_args()

    start = time.time()
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    print(f"Sistema pronto ({time.time() - start:.1f} s)\n")

    available = build_base_strategies(
        all_perms, lookup, args.models_dir, strict=args.strict_models, memoize=True
    )
    available.update(build_policy_routers(all_perms, lookup, available, ROUTING_POLICY_FILEPATH))
    available.update(build_recipe_strategies(all_perms, lookup, available, RECIPE_FILEPATH))
    unknown = [n for n in args.strategies if n not in available]
    if unknown:
        needs = set(POLICY_ROUTER_NAMES) | set(RECIPE_NAMES)
        hint = " (esegui prima train_policy_router)" if set(unknown) & needs else ""
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