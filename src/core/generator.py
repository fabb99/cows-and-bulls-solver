"""
Modulo: nome_modulo
Descrizione breve di cosa fa questo specifico file.
Nessuna logica di business esterna deve trapelare qui.
"""

from costants import(BASE,DIM)

from itertools import permutations

def generate_all_permutations(dim: int = DIM, base: int = BASE) -> list[str]:
    """
    PRE:
        - nessuna
    POST:
        - restituisce la lista di tutte le stringhe di DIM cifre distinte
        - la lista contiene 10 * 9 * 8 * 7 = 5040 elementi
    """
    characters = [str(x) for x in range(0,base)]
    return ["".join(p) for p in permutations(characters, dim)]

def generate_all_feedbacks(dim: int = DIM) -> list[tuple[int,int]]:
    """
        PRE:
            - nessuna
        POST:
            - restituisce la lista di tutte le valutazioni possibili
    """
    return [tuple([x,y]) for x in range(0,dim+1) for y in range(0,dim-x+1) if not(x==3 and y==1)]
