"""
Modulo: 4_exploratory_analysis (Script - experiments/oracle_lab/)
Analisi Esplorativa dei Dati (EDA) Avanzata:
1. Calcola la tabella statistica (Media, Dev.Std, Min, Max) per ciascuna feature.
2. Calcola e stampa la Matrice di Correlazione di Pearson Completa (K x K).
3. Evidenzia la multicollinearità tra le feature per guidare la selezione del modello.
"""
from typing import List, Tuple
from pathlib import Path
import json
import numpy as np

from experiments.oracle_lab.models import FeatureRecord

FEATURES_DATASET_FILEPATH = Path("data/features_dataset.json")


def load_dataset(filepath: Path) -> List[FeatureRecord]:
    """Carica i dati arricchiti dal file JSON."""
    if not filepath.exists():
        raise FileNotFoundError(f"Errore: {filepath} non trovato. Esegui prima 3_feature_builder.py!")
    with open(filepath, "r", encoding="utf-8") as file:
        dataset = json.load(file)     
    return [FeatureRecord.from_dict(record) for record in dataset]


def print_statistical_summary(feature_names: List[str], data_matrix: np.ndarray) -> None:
    """Stampa la tabella con Media, Deviazione Standard, Minimo e Massimo."""
    print("\n" + "=" * 78)
    print(f" STATISTICA DESCRITTIVA DEI BUCKET - {data_matrix.shape[0]} STATI ANALIZZATI")
    print("=" * 78)
    print(f"{'Feature':<22} | {'Media':<9} | {'Dev.Std':<9} | {'Min':<8} | {'Max':<8}")
    print("-" * 78)
    
    for i, name in enumerate(feature_names):
        col = data_matrix[:, i]
        print(f"{name:<22} | {np.mean(col):<9.3f} | {np.std(col):<9.3f} | {np.min(col):<8.2f} | {np.max(col):<8.2f}")
    print("=" * 78 + "\n")


def print_correlation_matrix(feature_names: List[str], short_names: List[str], data_matrix: np.ndarray) -> None:
    """
    Calcola e stampa la Matrice di Correlazione di Pearson Completa (K x K).
    """
    # np.corrcoef con rowvar=False tratta ogni colonna come una variabile!
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)

    print("=" * 105)
    print(" MATRICE DI CORRELAZIONE DI PEARSON COMPLETA (K x K)")
    print("=" * 105)
    
    # Intestazione Colonne
    header_cols = " | ".join([f"{name:>8}" for name in short_names])
    print(f"{'Feature':<22} | {header_cols}")
    print("-" * 105)

    # Righe della matrice
    for i, name in enumerate(feature_names):
        row_vals = []
        for j in range(len(short_names)):
            val = corr_matrix[i, j]
            # Formattazione con segno esplicito
            row_vals.append(f"{val:>+8.3f}")
        print(f"{name:<22} | " + " | ".join(row_vals))

    print("=" * 105)

    # Mini guida all'interpretazione
    print("\n Guida rapida di lettura:")
    print(" - Sulla diagonale c'è sempre +1.000 (ogni variabile con se stessa).")
    print(" - L'ultima colonna/riga (Costo_M) mostra quanto ogni feature predice il target.")
    print(" - Valori > +0.70 o < -0.70 tra due feature indicano forte RIDONDANZA (Multicollinearità)!\n")


def main():
    print("Caricamento dataset in corso...")
    dataset = load_dataset(FEATURES_DATASET_FILEPATH)

    # Nomi estesi e abbreviazioni per la griglia della matrice
    feature_names = [
        "Dimensione N",
        "log2(N)",
        "Rapporto Cifre",
        "Entropia Colonne",
        "Rapporto Anagrammi",
        "Varianza Cifre",
        "Coesione Interna",
        "Costo Reale M"
    ]
    
    short_names = [
        "N",
        "log2(N)",
        "Dig_R",
        "Col_Ent",
        "Ana_R",
        "Dig_Var",
        "Cohes",
        "Costo_M"
    ]

    # Matrice dati completa (N_samples x K_features)
    data_matrix = np.column_stack([
        np.array([r.n for r in dataset]),
        np.array([r.log_n for r in dataset]),
        np.array([r.unique_digits_ratio for r in dataset]),
        np.array([r.column_entropy for r in dataset]),
        np.array([r.anagram_ratio for r in dataset]),
        np.array([r.digit_variance for r in dataset]),
        np.array([r.internal_cohesion for r in dataset]),
        np.array([r.target_cost for r in dataset])
    ])

    # 1. Statistica descrittiva
    print_statistical_summary(feature_names, data_matrix)

    # 2. Matrice di correlazione completa
    print_correlation_matrix(feature_names, short_names, data_matrix)


if __name__ == "__main__":
    main()