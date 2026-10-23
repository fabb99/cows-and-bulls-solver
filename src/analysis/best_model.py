"""
Modulo: best_model
Statistiche di un albero e scelta del miglior modello EVM di un'iterazione.
"""
from pathlib import Path

from src.analysis.metrics import calculate_total_moves_and_leaves, max_depth
from src.analysis.tree_io import load_tree_from_json

MODEL_KINDS = ("selected", "geometric", "piecewise")


def tree_stats(path: Path) -> dict:
    """PRE: path è un albero JSON. POST: {'total_moves', 'games', 'avg', 'worst'}."""
    tree = load_tree_from_json(str(path))
    moves, games = calculate_total_moves_and_leaves(tree)
    return {"total_moves": moves, "games": games, "avg": moves / games, "worst": max_depth(tree)}


def pick_best_model(trees_dir: Path) -> tuple[str, dict]:
    """
    PRE:  trees_dir contiene almeno un estimated_evm_<kind>.json con kind in MODEL_KINDS.
    POST: (kind, statistiche) dell'albero con meno mosse totali; a parità il kind in ordine
          alfabetico. Solleva FileNotFoundError se non c'è nessun albero.
    """
    stats = {
        kind: tree_stats(trees_dir / f"estimated_evm_{kind}.json")
        for kind in MODEL_KINDS
        if (trees_dir / f"estimated_evm_{kind}.json").exists()
    }
    if not stats:
        raise FileNotFoundError(f"Nessun albero estimated_evm_* in {trees_dir}")
    kind = min(stats, key=lambda k: (stats[k]["total_moves"], k))
    return kind, stats[kind]