"""
Modulo: tree_builder
Contiene la logica per generare l'albero decisionale completo di una specifica strategia
giocata contro tutte le configurazioni possibili.
"""
from typing import List, Dict, Any, Callable,Optional
from src.strategies.base import BaseStrategy
from src.core.engine import calculate_buckets
from src.core.constants import DIM

# Un tipo custom (Type Alias) per semplificare la firma dei dizionari.
# Un Nodo sarà: {"guess": "0123", "solutions_count": 5040, "children": {"++": NodoFiglio, ...}}
TreeNode = Dict[str, Any]

def build_decision_tree(
    strategy: BaseStrategy, 
    possible_solutions: List[str], 
    history: List[str] = None,
    on_victory_callback: Optional[Callable[[int], None]] = None
) -> TreeNode:
    """
    Costruisce ricorsivamente l'albero di tutte le partite possibili per la data strategia.
    
    PRE: strategy è un'istanza valida. possible_solutions è la lista di soluzioni iniziali (es. 5040).
    POST: Restituisce un dizionario annidato che rappresenta i percorsi dell'algoritmo.
          Ogni foglia si ferma quando la valutazione è "++++".
    """
    if history is None:
        history = []

    num_solutions = len(possible_solutions) 

    if num_solutions == 1:
        guess = possible_solutions[0]
    elif num_solutions == 2:
        guess = possible_solutions[0]
    else:
        guess = strategy.choose_guess(possible_solutions, history)

    bucket = calculate_buckets(guess, possible_solutions, strategy.lookup)

    decision_tree = {}
    decision_tree["guess"] = guess
    decision_tree["possible_solutions"] = possible_solutions
    decision_tree["children"] = {}
    next_history = history + [guess]

    
    for feedback , next_possible_solutions in bucket.items():
        if not next_possible_solutions:
            continue
        if feedback == (DIM,0):
            decision_tree["children"][feedback] = None 
            if on_victory_callback is not None:
                on_victory_callback(1) 
        else: 
            decision_tree["children"][feedback] = build_decision_tree(strategy, next_possible_solutions, next_history, on_victory_callback)

    
    return decision_tree