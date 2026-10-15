"""
Modulo: feedback
Calcolo del feedback Cows and Bulls tra un'ipotesi e un segreto.
"""

# (bulls, cows): bulls = cifra giusta al posto giusto ('+'), cows = cifra giusta al posto sbagliato ('-')
Feedback = tuple[int, int]


def count_bulls(guess: str, secret: str) -> int:
    """
    PRE:  guess e secret hanno la stessa lunghezza.
    POST: numero di posizioni in cui guess e secret hanno la stessa cifra.
    """
    return sum(g == s for g, s in zip(guess, secret))


def count_cows(guess: str, secret: str) -> int:
    """
    PRE:  guess e secret hanno la stessa lunghezza e cifre distinte al loro interno.
    POST: numero di cifre di guess presenti in secret ma in posizione diversa.
    """
    return sum(g in secret and g != s for g, s in zip(guess, secret))


def calculate_feedback(guess: str, secret: str) -> Feedback:
    """
    PRE:  guess e secret sono stringhe valide (stessa lunghezza, cifre distinte).
    POST: tupla (bulls, cows).
    """
    return (count_bulls(guess, secret), count_cows(guess, secret))