"""
Modulo: metrics
Metriche di performance di un albero decisionale. Nessun I/O.
Convenzione: una foglia è None e vale "partita vinta con la mossa del padre".
"""
from typing import Optional

from src.analysis.tree_builder import TreeNode
from src.core.feedback import Feedback


def max_depth(tree: Optional[TreeNode]) -> int:
    """
    PRE:  tree è un nodo o None.
    POST: numero di mosse nel caso peggiore (0 per None).
    """
    if tree is None:
        return 0
    return 1 + max(max_depth(child) for child in tree["children"].values())


def calculate_total_moves_and_leaves(tree: Optional[TreeNode]) -> tuple[int, int]:
    """
    PRE:  tree è un nodo o None.
    POST: (somma delle mosse usate da tutte le partite, numero di partite).
          Una foglia vale (0, 1); ogni livello aggiunge 1 mossa a ciascuna partita sottostante.
    """
    if tree is None:
        return (0, 1)
    moves = leaves = 0
    for child in tree["children"].values():
        child_moves, child_leaves = calculate_total_moves_and_leaves(child)
        moves += child_moves
        leaves += child_leaves
    return (moves + leaves, leaves)


def average_depth(tree: Optional[TreeNode]) -> float:
    """PRE: tree valido. POST: numero medio di mosse per vincere (tutti i segreti equiprobabili)."""
    moves, games = calculate_total_moves_and_leaves(tree)
    return moves / games


def get_depth_distribution(tree: Optional[TreeNode]) -> dict[int, int]:
    """
    PRE:  tree valido.
    POST: {numero_di_mosse: partite concluse con quel numero di mosse}, ordinato per chiave.
          Es. {1: 1, 2: 14, 3: 156, ...}
    """
    distribution: dict[int, int] = {}
    _fill_distribution(tree, 1, distribution)
    return dict(sorted(distribution.items()))


def _fill_distribution(
    node: Optional[TreeNode], depth: int, distribution: dict[int, int]
) -> None:
    if node is None:
        return
    for child in node["children"].values():
        if child is None:
            distribution[depth] = distribution.get(depth, 0) + 1
        else:
            _fill_distribution(child, depth + 1, distribution)


def get_depth_distribution_next_move(
    tree: TreeNode,
) -> dict[Feedback, dict[int, int]]:
    """
    PRE:  tree è un nodo non nullo.
    POST: per ogni feedback non vincente del primo turno, la distribuzione dei turni
          RELATIVI al sotto-albero (1 = si vince alla mossa successiva alla prima).
    """
    return {
        fb: get_depth_distribution(child)
        for fb, child in tree["children"].items()
        if child is not None
    }