"""
Modulo: selection
Scelta deterministica della mossa migliore a partire dai punteggi.
"""
from collections.abc import Iterable

TIE_TOLERANCE: float = 1e-9


def select_best(
    scores: dict[str, float],
    possible_solutions: Iterable[str],
    higher_is_better: bool,
    tol: float = TIE_TOLERANCE,
) -> str:
    """
    PRE:  scores non vuoto, senza NaN.
    POST: tra i candidati entro `tol` dal punteggio migliore sceglie, in ordine:
          1) una soluzione ancora possibile (chance di vincere subito);
          2) la stringa lessicograficamente minore.
          Il risultato non dipende dall'ordine di inserimento né da errori di arrotondamento.
    """
    if not scores:
        raise ValueError("scores è vuoto")
    best = max(scores.values()) if higher_is_better else min(scores.values())
    tied = [c for c, s in scores.items() if abs(s - best) <= tol]
    solutions = set(possible_solutions)
    return min(tied, key=lambda c: (c not in solutions, c))