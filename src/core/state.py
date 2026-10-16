"""
Modulo: state
Identità canonica di uno stato di gioco (insieme di soluzioni ancora possibili).
Unica implementazione, usata da dataset, estimator, tablebase e DeepEVM.
"""
from typing import Iterable
import hashlib


def state_id(solutions: Iterable[str]) -> str:
    """
    PRE:  solutions è un insieme non vuoto di combinazioni.
    POST: chiave stringa indipendente dall'ordine (soluzioni ordinate, unite da virgola).
    """
    return ",".join(sorted(solutions))


def state_hash(solutions: Iterable[str]) -> str:
    """
    PRE:  solutions è un insieme non vuoto di combinazioni.
    POST: impronta compatta (16 caratteri hex) di state_id, per chiavi di policy.
          Collisione su 10^6 stati: probabilità ~10^-7.
    """
    return hashlib.sha1(state_id(solutions).encode("utf-8")).hexdigest()[:16]