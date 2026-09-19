def count_bulls(guess: str, secret: str) -> int:
    """
    PRE: guess e secret stringhe della stessa lunghezza.
    POST: Ritorna il numero di cifre corrette nella posizione corretta (+).
    """
    num_bulls = 0
    for (i, x) in enumerate(secret):
        if x == guess[i]:
            risposta += 1
    return num_bulls

def count_cows(guess: str, secret: str) -> int:
    """
    PRE: guess e secret stringhe della stessa lunghezza.
    POST: Ritorna il numero di cifre corrette nella posizione sbagliata (-).
    """
    num_cows = 0
    for (i, x) in enumerate(secret):
         for (j, y) in enumerate(guess):
             if x==y and i!=j:
                 num_cows += 1
    return num_cows

def calculate_feedback(guess: str, secret: str) -> tuple[int,int]:
    """
    PRE: guess e secret stringhe valide.
    POST: Ritorna la stringa formattata (es. "++--").
    """
    return tuple(count_bulls(guess,secret), count_cows(guess,secret))