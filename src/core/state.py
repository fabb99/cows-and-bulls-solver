"""
Modulo: state
Identità canonica di uno stato di gioco (insieme di soluzioni ancora possibili).
Unica implementazione, usata da dataset, estimator, tablebase e DeepEVM.
"""
from typing import Iterable


def state_id(solutions: Iterable[str]) -> str:
    """
    PRE:  solutions è un insieme non vuoto di combinazioni.
    POST: chiave stringa indipendente dall'ordine (soluzioni ordinate, unite da virgola).
    """
    return ",".join(sorted(solutions))