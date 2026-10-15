"""
Modulo: costs
Valore atteso del numero di mosse di un candidato, dato il costo di ogni bucket residuo.
"""
import math
from typing import Callable

from src.core.constants import WIN_FEEDBACK
from src.core.feedback import Feedback


def expected_cost(
    buckets: dict[Feedback, list[str]],
    n_solutions: int,
    cost_of: Callable[[list[str]], float],
) -> float:
    """
    PRE:  buckets è la partizione di n_solutions soluzioni prodotta da un candidato.
    POST: 1 + sum_{bucket != vittoria} P(bucket) * cost_of(bucket).
          Se il candidato non partiziona nulla (un solo bucket, non vincente)
          ritorna math.inf: una mossa che non fa progressi non va mai scelta.
    """
    if len(buckets) == 1 and WIN_FEEDBACK not in buckets:
        return math.inf
    total = 1.0
    for fb, solutions in buckets.items():
        if fb != WIN_FEEDBACK:
            total += (len(solutions) / n_solutions) * cost_of(solutions)
    return total