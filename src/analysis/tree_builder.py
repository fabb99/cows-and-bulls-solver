"""
Modulo: tree_builder
Costruzione dell'albero decisionale completo di una strategia contro tutti i segreti possibili.
"""
from typing import Any, Callable, Optional

from src.core.constants import WIN_FEEDBACK
from src.core.engine import calculate_buckets
from src.core.feedback import Feedback
from src.strategies.base import BaseStrategy

# Nodo: {"guess": str, "possible_solutions": list[str], "children": {Feedback: Nodo | None}}
# Un figlio None è una foglia: il feedback di vittoria (partita conclusa).
TreeNode = dict[str, Any]


def build_decision_tree(
    strategy: BaseStrategy,
    possible_solutions: list[str],
    history: Optional[list[str]] = None,
    on_victory_callback: Optional[Callable[[int], None]] = None,
) -> TreeNode:
    """
    PRE:  strategy è inizializzata; possible_solutions è non vuota.
          history è lo storico delle mosse già giocate (vuoto alla radice).
    POST: albero di tutte le partite possibili. Con 1 o 2 soluzioni la mossa è la prima
          soluzione (con 2 è già ottimo); altrimenti la sceglie la strategia.
          Ogni foglia (None) corrisponde a una vittoria; `on_victory_callback(1)`
          è chiamata una volta per foglia.
    """
    if not possible_solutions:
        raise ValueError("possible_solutions è vuota")
    history = history or []

    if len(possible_solutions) <= 2:
        guess = possible_solutions[0]
    else:
        guess = strategy.choose_guess(possible_solutions, history)

    next_history = history + [guess]
    children: dict[Feedback, Optional[TreeNode]] = {}
    for fb, bucket in calculate_buckets(guess, possible_solutions, strategy.lookup).items():
        if fb == WIN_FEEDBACK:
            children[fb] = None
            if on_victory_callback is not None:
                on_victory_callback(1)
        else:
            children[fb] = build_decision_tree(strategy, bucket, next_history, on_victory_callback)

    return {"guess": guess, "possible_solutions": possible_solutions, "children": children}