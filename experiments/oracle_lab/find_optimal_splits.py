"""
Modulo: find_optimal_splits (Script)
Algoritmo CART 1D: cerca le soglie su N che minimizzano la somma degli errori quadratici
(SSE) del Costo Reale rispetto alla media di ogni scaglione.

NOTA (DA STUDIARE prima dell'ultimo commit del refactoring; per ora il comportamento
resta quello originale):
  1. La "varianza eliminata" del secondo taglio è relativa al solo sotto-insieme destro
     (N > T1), non al totale: il numero stampato non è confrontabile con il primo.
  2. Le soglie sono scelte su SSE in-sample, senza cross-validation.
  3. Il target mescola costi esatti e upper bound, e gli stati non sono un campione
     uniforme (vengono dagli alberi). Le soglie dipendono dal dataset.
  4. Il taglio è solo su N: nessuna verifica che scaglioni a gradini siano meglio
     di un modello continuo (es. spline in log2(N)).

Uso: python -m experiments.oracle_lab.find_optimal_splits
"""

import numpy as np

from experiments.oracle_lab.config import FEATURES_DATASET_FILEPATH
from experiments.oracle_lab.storage import load_features


def compute_sum_squared_errors(y: np.ndarray) -> float:
    """
    PRE:  y è un array 1D.
    POST: sum((y - mean(y))^2). Se len(y) <= 1 ritorna 0.0.
    """
    if len(y) <= 1:
        return 0.0
    return float(np.sum((y - np.mean(y)) ** 2))


def find_best_single_split(
    n_values: np.ndarray,
    costs: np.ndarray,
    min_leaf_samples: int = 50,
    ss_ref: float | None = None,
) -> tuple[int | None, float, float]:
    """
    PRE:  n_values e costs hanno la stessa lunghezza.
          ss_ref: SSE di riferimento per normalizzare (default: SSE dei costs passati).
    POST: (T*, riduzione di SSE / ss_ref, SSE minima dopo il taglio).
          T* è None se nessuna soglia lascia almeno min_leaf_samples campioni per lato.
    """
    ss_tot = compute_sum_squared_errors(costs)
    ss_ref = ss_tot if ss_ref is None else ss_ref
    best_threshold: int | None = None
    min_total_error = np.inf

    for t in sorted({int(x) for x in n_values}):
        left = n_values <= t
        right = n_values > t
        if left.sum() < min_leaf_samples or right.sum() < min_leaf_samples:
            continue
        total_error = compute_sum_squared_errors(costs[left]) + compute_sum_squared_errors(costs[right])
        if total_error < min_total_error:
            min_total_error = total_error
            best_threshold = t

    gain = (ss_tot - min_total_error) / ss_ref if ss_ref > 0 else 0.0
    return best_threshold, gain, float(min_total_error)

def find_hierarchical_splits(
    n_values: np.ndarray,
    costs: np.ndarray,
    max_splits: int = 2,
    ss_ref: float | None = None,
) -> list[tuple[int, float]]:
    """
    POST: lista ordinata di (soglia, riduzione di SSE rispetto all'SSE TOTALE del dataset).
    """
    if max_splits <= 0 or len(n_values) < 100:
        return []
    ss_ref = compute_sum_squared_errors(costs) if ss_ref is None else ss_ref

    t, gain, _ = find_best_single_split(n_values, costs, ss_ref=ss_ref)
    if t is None:
        return []
    if max_splits == 1:
        return [(t, gain)]

    right = n_values > t
    sub_splits = find_hierarchical_splits(n_values[right], costs[right], max_splits - 1, ss_ref)
    sub_splits = [(t2, g2) for t2, g2 in sub_splits if g2 > 1e-9]
    return sorted([(t, gain)] + sub_splits, key=lambda x: x[0])

def main() -> None:
    dataset = load_features(FEATURES_DATASET_FILEPATH)
    if not dataset:
        print(f"{FEATURES_DATASET_FILEPATH} assente o vuoto: esegui prima feature_builder.")
        return
    n_values = np.array([r.n for r in dataset])
    costs = np.array([r.target_cost for r in dataset])

    splits = find_hierarchical_splits(n_values, costs, max_splits=2)

    print("\n" + "=" * 65)
    print(" SOGLIE E RIDUZIONE DI SSE (CART)")
    print("=" * 65)
    for i, (threshold, gain) in enumerate(splits, start=1):
        print(f"Taglio {i}: soglia N = {threshold:<4} | SSE eliminata: {gain * 100:.2f}%")
    print("-" * 65)
    if len(splits) >= 2:
        t1, t2 = splits[0][0], splits[1][0]
        print(f"1. ENDGAME : N <= {t1}")
        print(f"2. MIDGAME : {t1} < N <= {t2}")
        print(f"3. OPENING : N > {t2}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()