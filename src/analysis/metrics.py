"""
Modulo: metrics
Contiene le funzioni matematiche per calcolare le performance di un albero decisionale.
Nessuna operazione di I/O (nessun print).
"""
from typing import Dict, Optional, Tuple
from src.analysis.tree_builder import TreeNode
from src.core.constants import DIM

def max_depth(tree: Optional[TreeNode]) -> int:
    """
    PRE: tree è un nodo dell'albero o None.
    POST: Ritorna la profondità massima (il numero di mosse nel caso peggiore).
    """
    if tree is not None:
        return 1 + max(max_depth(x) for x in tree["children"].values())
    else:
        return 0
    
def calculate_total_moves_and_leaves(tree: Optional[TreeNode]) -> Tuple[int, int]:
    """
    Funzione di supporto per calcolare la media.
    PRE: tree valido.
    POST: Ritorna (somma_totale_mosse_per_vincere, numero_di_partite_possibili).
    """
    if tree is not None:
        num_moves = 0
        num_leaves = 0 
        for next_tree in tree["children"].values():
            (num_moves_child, num_leaves_child) = calculate_total_moves_and_leaves(next_tree)
            num_moves  += num_moves_child  
            num_leaves += num_leaves_child
        return (num_moves + num_leaves, num_leaves)
    else: 
        return (0, 1)

def average_depth(tree: Optional[TreeNode]) -> float:
    """
    PRE: tree valido.
    POST: Ritorna il numero medio di mosse necessarie per vincere.
    """
    (moves, games) = calculate_total_moves_and_leaves(tree)
    return moves/games

def get_depth_distribution(tree: Optional[TreeNode]) -> Dict[int, int]:
    """
    PRE: tree valido.
    POST: Ritorna un dizionario {numero_di_mosse: quante_partite_finiscono_in_quelle_mosse}.
          Es: {1: 1, 2: 14, 3: 156, 4: 980, ...}
    """
    distribuition = {}
    _fill_distribuition(tree, 1, distribuition)
    return dict(sorted(distribuition.items()))

def _fill_distribuition(node: Optional[TreeNode], current_depth: int, distribuition: Dict[int, int]):
    if node is None:
        return

    for child_node in node["children"].values():
        if child_node is None:
            distribuition[current_depth] = distribuition.get(current_depth, 0) + 1
        else:
            _fill_distribuition(child_node, current_depth+1, distribuition)

def get_depth_distribution_next_move(tree: Optional[TreeNode]) -> Dict[tuple[int,int], Dict[int, int]]:
    distributions = {}
    for fb, sub_tree in tree["children"].items():
        if not sub_tree is None:
            distributions[fb] = get_depth_distribution(sub_tree)
    return distributions
    
    