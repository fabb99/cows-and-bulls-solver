"""
Modulo: exploratory_analysis (Script)
Fase 3: EDA globale e per scaglione (soglie in config.DEFAULT_SPLITS):
statistiche descrittive e matrice di correlazione di Pearson tra n, feature e costo.

Uso: python -m experiments.oracle_lab.exploratory_analysis
"""
import numpy as np

from experiments.oracle_lab.config import DEFAULT_SPLITS, FEATURES_DATASET_FILEPATH
from experiments.oracle_lab.models import FeatureRecord
from experiments.oracle_lab.storage import load_features
from src.features.bucket_features import FEATURES

COLUMNS = ["n", *FEATURES, "target_cost"]


def build_data_matrix(records: list[FeatureRecord]) -> np.ndarray:
    """PRE: records non vuota. POST: matrice (len(records), len(COLUMNS)) nell'ordine di COLUMNS."""
    return np.array(
        [[r.n, *(r.features[f] for f in FEATURES), r.target_cost] for r in records], dtype=float
    )


def print_statistical_summary(matrix: np.ndarray) -> None:
    """POST: stampa media, deviazione standard, minimo e massimo di ogni colonna."""
    print("-" * 70)
    print(f"{'Colonna':<22} | {'Media':>9} | {'Dev.Std':>9} | {'Min':>8} | {'Max':>8}")
    print("-" * 70)
    for i, name in enumerate(COLUMNS):
        col = matrix[:, i]
        print(f"{name:<22} | {col.mean():>9.3f} | {col.std():>9.3f} | {col.min():>8.2f} | {col.max():>8.2f}")
    print("-" * 70)


def print_correlation_matrix(matrix: np.ndarray) -> None:
    """POST: stampa la matrice di Pearson ('N/A' per colonne costanti)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix, rowvar=False)
    header = " | ".join(f"{name[:8]:>8}" for name in COLUMNS)
    print(f"\n{'':<22} | {header}")
    for i, name in enumerate(COLUMNS):
        cells = " | ".join(
            f"{'N/A':>8}" if np.isnan(v) else f"{v:>+8.3f}" for v in corr[i]
        )
        print(f"{name:<22} | {cells}")
    print()


def analyze_segment(title: str, records: list[FeatureRecord]) -> None:
    """POST: stampa riepilogo e correlazioni del sottoinsieme (o un avviso se vuoto)."""
    if not records:
        print(f"\nNessun record per {title}")
        return
    n_exact = sum(r.is_exact for r in records)
    print("\n" + "=" * 70)
    print(f" {title}: {len(records)} stati ({n_exact} con costo esatto)")
    print("=" * 70)
    matrix = build_data_matrix(records)
    print_statistical_summary(matrix)
    print_correlation_matrix(matrix)


def segment_records(records: list[FeatureRecord], splits: list[int]) -> list[tuple[str, list[FeatureRecord]]]:
    """POST: [(titolo, record)] per gli scaglioni definiti da splits (N <= s0, s0 < N <= s1, ...)."""
    bounds = [0, *splits, float("inf")]
    return [
        (f"SCAGLIONE {lo} < N <= {hi}", [r for r in records if lo < r.n <= hi])
        for lo, hi in zip(bounds, bounds[1:])
    ]


def main() -> None:
    records = load_features(FEATURES_DATASET_FILEPATH)
    if not records:
        print(f"{FEATURES_DATASET_FILEPATH} assente o vuoto: esegui prima feature_builder.")
        return
    analyze_segment("ANALISI GLOBALE", records)
    for title, subset in segment_records(records, DEFAULT_SPLITS):
        analyze_segment(title, subset)


if __name__ == "__main__":
    main()