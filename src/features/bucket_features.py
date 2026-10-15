"""
Modulo: bucket_features
Registry unico delle feature geometriche di un bucket (insieme di soluzioni).
Usato sia dal training (experiments/) sia dall'inferenza (estimators/): una sola
implementazione per ogni formula, quindi nessuno skew possibile tra le due fasi.
"""
import math
from collections import Counter
from typing import Callable

from src.core.constants import BASE, DIM

Feature = Callable[[list[str]], float]


def log_n(solutions: list[str]) -> float:
    """PRE: solutions non vuota. POST: log2(N)."""
    return math.log2(len(solutions))


def unique_digits_ratio(solutions: list[str]) -> float:
    """POST: (numero di cifre distinte usate nell'insieme) / BASE, in [DIM/BASE, 1]."""
    return len({c for s in solutions for c in s}) / BASE


def _digit_frequencies(solutions: list[str]) -> list[int]:
    counts = Counter("".join(solutions))
    return [counts.get(str(d), 0) for d in range(BASE)]


def digit_std(solutions: list[str]) -> float:
    """POST: deviazione standard (popolazione) delle frequenze assolute delle BASE cifre."""
    freqs = _digit_frequencies(solutions)
    mean = sum(freqs) / BASE
    return math.sqrt(sum((f - mean) ** 2 for f in freqs) / BASE)


def column_entropy(solutions: list[str]) -> float:
    """POST: entropia di Shannon (bit) media sulle DIM colonne."""
    n = len(solutions)
    total = 0.0
    for col in range(DIM):
        counts = Counter(s[col] for s in solutions)
        total -= sum((c / n) * math.log2(c / n) for c in counts.values())
    return total / DIM


def anagram_ratio(solutions: list[str]) -> float:
    """POST: (insiemi di cifre distinti, ignorando l'ordine) / N, in (0, 1]."""
    return len({"".join(sorted(s)) for s in solutions}) / len(solutions)


def internal_cohesion(solutions: list[str]) -> float:
    """
    POST: allineamento medio delle soluzioni con il profilo per colonna del bucket,
          in (0, 1]. Equivale a mean_s( mean_col( P(col, s[col]) ) ).
    """
    n = len(solutions)
    col_counts = [Counter(s[col] for s in solutions) for col in range(DIM)]
    total = sum(col_counts[col][s[col]] for s in solutions for col in range(DIM))
    return total / (n * n * DIM)


FEATURES: dict[str, Feature] = {
    "log_n": log_n,
    "unique_digits_ratio": unique_digits_ratio,
    "digit_std": digit_std,
    "column_entropy": column_entropy,
    "anagram_ratio": anagram_ratio,
    "internal_cohesion": internal_cohesion,
}


def compute_features(solutions: list[str], names: list[str]) -> list[float]:
    """
    PRE:  solutions non vuota; names sono chiavi di FEATURES.
    POST: valori delle feature richieste, nello stesso ordine di `names`.
    """
    return [FEATURES[name](solutions) for name in names]