"""
Modulo: signatures
Classi di equivalenza (isomorfismi) tra ipotesi, usate per ridurre la ricerca al Turno 1.
Due candidate con la stessa "firma" rispetto allo storico sono intercambiabili.
"""
#from typing import Hashable

# 0 = cifra mai giocata, 1 = giocata in posizione diversa, 2 = giocata nella stessa posizione
_NEW, _MISPLACED, _SAME_POS = 0, 1, 2

Signature = tuple[tuple[int, ...], ...]


def calculate_signature(candidate: str, history: list[str]) -> Signature:
    """
    PRE:  candidate è una combinazione; history la lista delle mosse già giocate.
    POST: tupla ordinata che descrive, per ogni cifra di candidate, come è comparsa nelle
          mosse dello storico (0=nuova, 1=presente ma in altra posizione, 2=stessa posizione).
          L'ordinamento rende la firma indipendente dalla posizione delle cifre.
    """
    signature = []
    for i, digit in enumerate(candidate):
        per_move = []
        for move in history:
            if digit not in move:
                per_move.append(_NEW)
            elif move[i] == digit:
                per_move.append(_SAME_POS)
            else:
                per_move.append(_MISPLACED)
        signature.append(tuple(per_move))
    return tuple(sorted(signature))


def group_candidates_by_signature(
    candidates: list[str], history: list[str]
) -> dict[Signature, list[str]]:
    """
    PRE:  candidates e history validi.
    POST: {firma: [candidati con quella firma]}, ordine dei candidati preservato.
    """
    groups: dict[Signature, list[str]] = {}
    for c in candidates:
        groups.setdefault(calculate_signature(c, history), []).append(c)
    return groups


def get_representatives(
    candidates: list[str], history: list[str], possible_solutions: list[str]
) -> list[str]:
    """
    PRE:  candidates e history validi.
    POST: un rappresentante per firma, preferendo (se esiste) una soluzione ancora possibile.
          Con history vuota (Turno 1) restituisce un solo rappresentante: tutte le
          combinazioni sono equivalenti per simmetria.
    """
    solutions = set(possible_solutions)
    representatives = []
    for group in group_candidates_by_signature(candidates, history).values():
        in_solutions = [c for c in group if c in solutions]
        representatives.append(in_solutions[0] if in_solutions else group[0])
    return representatives