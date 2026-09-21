from src.core.lookup_table import(FeedbackLookup)

def filter_solutions(guess: str, feedback: str, candidates: list[str], lookup: FeedbackLookup) -> list[str]:
    """
    PRE: guess e feedback validi. candidates lista di soluzioni rimaste. lookup inizializzato.
    POST: Ritorna la sottolista di candidates compatibili.
    """
    return [c for c in candidates if lookup.get_feedback(guess, c) == feedback]


def calculate_buckets(guess: str, candidates: list[str], lookup: FeedbackLookup) -> dict[tuple[int, int], list[str]]:
    """
    PRE: guess valida, candidates lista di soluzioni rimaste.
    POST: Ritorna dizionario { feedback: [soluzioni_compatibili] }.
    """
    bucket = {}
    for c in candidates:
        fb = lookup.get_feedback(guess,c)
        if fb not in bucket:
            bucket[fb]=[]
        bucket[fb].append(c)
    return bucket
