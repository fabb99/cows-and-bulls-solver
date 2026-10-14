"""
Modulo: 4_exploratory_analysis (Script - experiments/oracle_lab/)
Analisi Esplorativa dei Dati (EDA) Globale e a Scaglioni (CART):
1. Statistica descrittiva e Matrice di Correlazione globale (4.781 stati).
2. Matrici di Correlazione segmentate per i 3 regimi fisici del gioco:
   - Endgame: N <= 13
   - Midgame: 13 < N <= 40
   - Opening: N > 40
"""
from typing import List, Tuple
from pathlib import Path
import json
import numpy as np

from experiments.oracle_lab.models import FeatureRecord

FEATURES_DATASET_FILEPATH = Path("data/features_dataset.json")

SPLIT_ENDGAME: int = 13
SPLIT_MIDGAME: int = 43

FEATURE_NAMES = [
    "Dimensione N",
    "log2(N)",
    "Rapporto Cifre",
    "Entropia Colonne",
    "Rapporto Anagrammi",
    "Varianza Cifre",
    "Coesione Interna",
    "Costo Reale M"
]

SHORT_NAMES = [
    "N",
    "log2(N)",
    "Dig_R",
    "Col_Ent",
    "Ana_R",
    "Dig_Var",
    "Cohes",
    "Costo_M"
]


def load_dataset(filepath: Path) -> List[FeatureRecord]:
    """Carica i dati arricchiti dal file JSON."""
    if not filepath.exists():
        raise FileNotFoundError(f"Errore: {filepath} non trovato. Esegui prima 3_feature_builder.py!")
    with open(filepath, "r", encoding="utf-8") as file:
        dataset = json.load(file)     
    return [FeatureRecord.from_dict(record) for record in dataset]


def build_data_matrix(records: List[FeatureRecord]) -> np.ndarray:
    """Trasforma una lista di FeatureRecord in una matrice NumPy 2D (N_samples x 8)."""
    return np.column_stack([
        np.array([r.n for r in records]),
        np.array([r.log_n for r in records]),
        np.array([r.unique_digits_ratio for r in records]),
        np.array([r.column_entropy for r in records]),
        np.array([r.anagram_ratio for r in records]),
        np.array([r.digit_variance  for r in records]),
        np.array([r.internal_cohesion for r in records]),
        np.array([r.target_cost for r in records])
    ])


def print_statistical_summary(feature_names: List[str], data_matrix: np.ndarray) -> None:
    """Stampa la tabella con Media, Deviazione Standard, Minimo e Massimo."""
    print("-" * 78)
    print(f"{'Feature':<22} | {'Media':<9} | {'Dev.Std':<9} | {'Min':<8} | {'Max':<8}")
    print("-" * 78)
    for i, name in enumerate(feature_names):
        col = data_matrix[:, i]
        print(f"{name:<22} | {np.mean(col):<9.3f} | {np.std(col):<9.3f} | {np.min(col):<8.2f} | {np.max(col):<8.2f}")
    print("-" * 78)


def print_correlation_matrix(feature_names: List[str], short_names: List[str], data_matrix: np.ndarray) -> None:
    """Calcola e stampa la Matrice di Correlazione di Pearson (K x K)."""
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)

    print("\n" + "-" * 105)
    header_cols = " | ".join([f"{name:>8}" for name in short_names])
    print(f"{'Feature':<22} | {header_cols}")
    print("-" * 105)

    for i, name in enumerate(feature_names):
        row_vals = []
        for j in range(len(short_names)):
            val = corr_matrix[i, j]
            # Gestione sicura per evitare crash se la deviazione standard è 0 (NaN)
            val_str = f"{val:>+8.3f}" if not np.isnan(val) else f"{'N/A':>8}"
            row_vals.append(val_str)
        print(f"{name:<22} | " + " | ".join(row_vals))
    print("-" * 105 + "\n")


def analyze_segment(title: str, records: List[FeatureRecord]) -> None:
    """Esegue EDA e Matrice per un sottoinsieme di dati."""
    if not records:
        print(f"\nNessun record per {title}!")
        return

    print("\n" + "=" * 105)
    print(f" {title} - {len(records)} STATI")
    print("=" * 105)

    matrix = build_data_matrix(records)
    print_statistical_summary(FEATURE_NAMES, matrix)
    print_correlation_matrix(FEATURE_NAMES, SHORT_NAMES, matrix)


def main():
    print("Caricamento dataset in corso...")
    dataset = load_dataset(FEATURES_DATASET_FILEPATH)

    # 1. ANALISI GLOBALE COMPLETA
    analyze_segment("ANALISI GLOBALE COMPLETA", dataset)

    # 2. PARTIZIONE NEI 3 SCAGLIONI CART
    endgame_records = [r for r in dataset if r.n <= SPLIT_ENDGAME]
    midgame_records = [r for r in dataset if SPLIT_ENDGAME < r.n <= SPLIT_MIDGAME]
    opening_records = [r for r in dataset if r.n > SPLIT_MIDGAME]

    # 3. ANALISI SEPARATA PER SCAGLIONE
    analyze_segment(f"SCAGLIONE 1: ENDGAME (N <= {SPLIT_ENDGAME})", endgame_records)
    analyze_segment(f"SCAGLIONE 2: MIDGAME ({SPLIT_ENDGAME} < N <= {SPLIT_MIDGAME})", midgame_records)
    analyze_segment(f"SCAGLIONE 3: OPENING (N > {SPLIT_MIDGAME})", opening_records)


if __name__ == "__main__":
    main()