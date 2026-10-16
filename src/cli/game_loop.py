"""
Modulo: game_loop
Interazione da terminale e simulazione di partite.
"""

from src.core.constants import WIN_FEEDBACK
from src.core.engine import filter_solutions
from src.core.feedback import Feedback
from src.core.generator import generate_all_feedbacks
from src.strategies.base import BaseStrategy

NO_MATCH_SYMBOL = "0"  # stringa digitata per il feedback (0, 0)
QUIT_COMMANDS = {"q", "quit", "esci"}
MAX_TURNS = 20  # rete di sicurezza contro loop infiniti


def fb_tuple_to_string(fb: Feedback) -> str:
    """POST: '+' per ogni bull e '-' per ogni cow, es. (2, 1) -> '++-'. (0, 0) -> ''."""
    return "+" * fb[0] + "-" * fb[1]


def parse_feedback(text: str) -> Feedback | None:
    """
    PRE:  text è l'input grezzo dell'utente.
    POST: (bulls, cows) se text è un feedback realizzabile, altrimenti None.
          Accetta solo '+' e '-' (ordine libero); '0' indica nessuna corrispondenza.
          La stringa vuota è rifiutata (quasi sempre un invio per errore).
          Feedback impossibili (es. '+++-') sono rifiutati.
    """
    text = text.strip()
    if text == NO_MATCH_SYMBOL:
        return (0, 0)
    if not text or set(text) - {"+", "-"}:
        return None
    fb = (text.count("+"), text.count("-"))
    return fb if fb in set(generate_all_feedbacks()) else None


def play_interactive(strategy: BaseStrategy, possible_solutions: list[str]) -> int | None:
    """
    PRE:  strategy inizializzata; possible_solutions è l'insieme iniziale (non vuoto).
    POST: stampa l'ipotesi della strategia e legge il feedback dell'utente finché non è
          '++++'. Un feedback malformato viene richiesto; uno incoerente con le risposte
          precedenti (nessuna soluzione compatibile) viene rifiutato senza perdere la partita.
          Ritorna i tentativi usati, oppure None se l'utente esce.
    """
    solutions = list(possible_solutions)
    history: list[str] = []

    while len(history) < MAX_TURNS:
        guess = strategy.choose_guess(solutions, history)
        print(f"Ipotesi: {guess}   (candidati rimasti: {len(solutions)})")

        while True:
            raw = input("Feedback (+ = toro, - = mucca, 0 = niente, q = esci): ")
            if raw.strip().lower() in QUIT_COMMANDS:
                return None
            fb = parse_feedback(raw)
            if fb is None:
                print("Feedback non valido o impossibile: riprova.")
                continue
            remaining = [] if fb == WIN_FEEDBACK else filter_solutions(
                guess, fb, solutions, strategy.lookup
            )
            if fb != WIN_FEEDBACK and not remaining:
                print("Nessuna combinazione è compatibile con questo feedback: ricontrolla.")
                continue
            break

        history.append(guess)
        if fb == WIN_FEEDBACK:
            print(f"Risolto in {len(history)} tentativi")
            return len(history)
        solutions = remaining

    print("Limite di turni raggiunto.")
    return None


def play_autoplay(
    strategy: BaseStrategy,
    secret: str,
    possible_solutions: list[str],
    verbose: bool = True,
) -> int:
    """
    PRE:  secret appartiene a possible_solutions (e al dominio della lookup).
    POST: simula la partita; se verbose stampa ogni passo. Ritorna i turni usati.
          Solleva ValueError se il segreto non è tra le soluzioni.
    """
    if secret not in possible_solutions:
        raise ValueError(f"Il segreto {secret!r} non è tra le soluzioni possibili")

    solutions = list(possible_solutions)
    history: list[str] = []

    while True:
        guess = strategy.choose_guess(solutions, history)
        history.append(guess)
        fb = strategy.lookup.get_feedback(guess, secret)
        if verbose:
            print(f"{len(history)}. {guess} -> {fb_tuple_to_string(fb) or NO_MATCH_SYMBOL}")
        if fb == WIN_FEEDBACK:
            if verbose:
                print(f"Risolto in {len(history)} tentativi")
            return len(history)
        solutions = filter_solutions(guess, fb, solutions, strategy.lookup)