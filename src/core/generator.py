"""
Modulo: generator
Generazione dell'universo delle soluzioni e dei feedback possibili.
"""
from itertools import permutations

from src.core.constants import BASE, DIM
from src.core.feedback import Feedback


def generate_all_permutations(dim: int = DIM, base: int = BASE) -> list[str]:
    """
    PRE:  dim <= base.
    POST: tutte le stringhe di `dim` cifre distinte in '0'..str(base-1), in ordine lessicografico.
          Con i default: 10*9*8*7 = 5040 elementi.
    """
    digits = [str(d) for d in range(base)]
    return ["".join(p) for p in permutations(digits, dim)]


def generate_all_feedbacks(dim: int = DIM) -> list[Feedback]:
    """
    PRE:  dim >= 1.
    POST: tutti i feedback (bulls, cows) realizzabili con `dim` cifre distinte.
          Esclude (dim-1, 1), impossibile: se dim-1 cifre sono al posto giusto,
          l'ultima non può essere "giusta ma fuori posto".
    """
    return [
        (b, c)
        for b in range(dim + 1)
        for c in range(dim - b + 1)
        if not (b == dim - 1 and c == 1)
    ]