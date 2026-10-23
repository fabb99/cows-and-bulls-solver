"""
Modulo: engine
Filtraggio delle soluzioni e partizionamento in bucket per feedback.
"""
from src.core.feedback import Feedback
from src.core.lookup_table import FeedbackLookup


def filter_solutions(
    guess: str, feedback: Feedback, candidates: list[str], lookup: FeedbackLookup
) -> list[str]:
    """
    PRE:  guess e candidates appartengono al dominio di lookup.
    POST: sottolista di candidates (stesso ordine) compatibili con `feedback` rispetto a `guess`.
    """
    return [c for c in candidates if lookup.get_feedback(guess, c) == feedback]


def calculate_buckets(
    guess: str, candidates: list[str], lookup: FeedbackLookup
) -> dict[Feedback, list[str]]:
    """
    PRE:  guess e candidates appartengono al dominio di lookup.
    POST: partizione di candidates {feedback: [soluzioni con quel feedback]}.
          Nessun bucket è vuoto; l'ordine relativo dei candidati è preservato.
    """
    buckets: dict[Feedback, list[str]] = {}
    for c in candidates:
        buckets.setdefault(lookup.get_feedback(guess, c), []).append(c)
    return buckets