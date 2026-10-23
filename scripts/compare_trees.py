"""
Modulo: compare_trees (Script)
Confronta gli alberi salvati: media, caso peggiore e distribuzione dei turni.
Con --branches ripete il confronto per ogni feedback del primo turno (turni RELATIVI).

Uso: python -m scripts.compare_trees [--trees A B ...] [--dir PERCORSO] [--branches]
"""
import argparse
from pathlib import Path
from typing import Any

from src.analysis.metrics import average_depth, get_depth_distribution, max_depth
from src.analysis.tree_builder import TreeNode
from src.analysis.tree_io import load_tree_from_json
from src.cli.game_loop import fb_tuple_to_string
from src.core.constants import WIN_FEEDBACK
from src.core.generator import generate_all_feedbacks
from src.paths import TREES_DIR

Report = dict[str, Any]


def analyze_tree(tree: TreeNode) -> Report:
    """PRE: tree valido. POST: {'avg', 'worst', 'dist'}."""
    return {
        "avg": average_depth(tree),
        "worst": max_depth(tree),
        "dist": get_depth_distribution(tree),
    }


def print_dashboard(title: str, reports: dict[str, Report], prefix: str = "T") -> None:
    """PRE: reports = {nome: Report} (può essere vuoto). POST: stampa le tabelle."""
    if not reports:
        return
    width = 82
    max_turns = max(r["worst"] for r in reports.values())
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)
    print(f"{'Strategia':<28} | {'Caso peggiore':<13} | {'Media mosse':<11}")
    print("-" * 58)
    for name, r in sorted(reports.items(), key=lambda kv: kv[1]["avg"]):
        print(f"{name:<28} | {r['worst']:<13} | {r['avg']:<11.5f}")
    header = f"\n{'Strategia':<28} | " + " | ".join(f"{prefix}{i:<4}" for i in range(1, max_turns + 1))
    print(header)
    print("-" * len(header))
    for name, r in reports.items():
        row = " | ".join(f"{r['dist'].get(i, 0):<5}" for i in range(1, max_turns + 1))
        print(f"{name:<28} | {row}")


def load_trees(directory: Path, names: list[str] | None) -> dict[str, TreeNode]:
    """POST: {nome: albero} per i file richiesti (o tutti) in `directory`."""
    paths = sorted(directory.glob("*.json"))
    if names:
        paths = [p for p in paths if p.stem in names]
        missing = set(names) - {p.stem for p in paths}
        if missing:
            print(f"Alberi non trovati in {directory}: {sorted(missing)}")
    return {p.stem: load_tree_from_json(str(p)) for p in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description="Confronta gli alberi decisionali.")
    parser.add_argument("--trees", nargs="+", default=None)
    parser.add_argument("--dir", type=Path, default=TREES_DIR)
    parser.add_argument("--branches", action="store_true")
    args = parser.parse_args()

    if not args.dir.exists():
        raise SystemExit(f"Cartella {args.dir} inesistente")
    trees = load_trees(args.dir, args.trees)
    if not trees:
        raise SystemExit(f"Nessun albero in {args.dir}")

    print_dashboard("CONFRONTO STRATEGIE (turni assoluti)", {n: analyze_tree(t) for n, t in trees.items()})

    if args.branches:
        for fb in generate_all_feedbacks():
            if fb == WIN_FEEDBACK:
                continue
            reports = {
                name: analyze_tree(tree["children"][fb])
                for name, tree in trees.items()
                if tree["children"].get(fb) is not None
            }
            label = fb_tuple_to_string(fb) or "0"
            print_dashboard(f"RAMO '{label}' dopo il primo turno (turni relativi)", reports, prefix="+")


if __name__ == "__main__":
    main()