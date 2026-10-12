"""
Modulo: 4_exploratory_analysis.py (Script)
Fase 1 della Pipeline:
Stampa la tabella statistica (Media, Dev.Std, Min, Max) e la matrice
di correlazione di Pearson per tutte le variabili geometriche
"""
from typing import List, Tuple
from pathlib import Path
import json
import numpy as np

from experiments.oracle_lab.models import FeatureRecord

FEATURES_DATASET_FILEPATH = Path("data/features_dataset.json")

def load_dataset(filepath: Path) -> List[FeatureRecord]:
    """Carica i dati arricchiti dal file JSON."""
    with open(filepath, "r", encoding="utf-8") as file:
        dataset = json.load(file)     
    return [FeatureRecord.from_dict(record) for record in dataset]

def print_exploratory_analysis(dataset: List[FeatureRecord]) -> None:
    # 1. Estrai i vettori base
    n_vals = np.array([r.n for r in dataset])
    log_n = np.array([r.log_n for r in dataset])
    digits_ratio = np.array([r.unique_digits_ratio for r in dataset])
    col_entropy = np.array([r.column_entropy for r in dataset])
    ana_ratio = np.array([r.anagram_ratio for r in dataset])
    dig_var = np.array([r.digit_variance for r in dataset])
    int_cohesion = np.array([r.internal_cohesion for r in dataset])

    target_m = np.array([r.target_cost for r in dataset])

    # 2. Stampa la tabella
    print("\n" + "=" * 75)
    print(f"ANALISI ESPLORATIVA DEI DATI DEI BUCKET - {len(dataset)} STATI")
    print("=" * 75)
    print(f"{'Feature':<25} | {'Media':<8} | {'Dev.Std':<8} | {'Min':<6} | {'Max':<6}")
    print("-" * 75)
    print(f"{'Dimensione N':<25} | {np.mean(n_vals):<8.2f} | {np.std(n_vals):<8.2f} | {np.min(n_vals):<6} | {np.max(n_vals):<6}")
    print(f"{'log2(N)':<25} | {np.mean(log_n):<8.3f} | {np.std(log_n):<8.3f} | {np.min(log_n):<6.2f} | {np.max(log_n):<6.2f}")
    print(f"{'Rapporto Cifre':<25} | {np.mean(digits_ratio):<8.3f} | {np.std(digits_ratio):<8.3f} | {np.min(digits_ratio):<6.2f} | {np.max(digits_ratio):<6.2f}")
    print(f"{'Entropia Colonne':<25} | {np.mean(col_entropy):<8.3f} | {np.std(col_entropy):<8.3f} | {np.min(col_entropy):<6.2f} | {np.max(col_entropy):<6.2f}")
    print(f"{'Rapporto Anagrammi':<25} | {np.mean(ana_ratio):<8.3f} | {np.std(ana_ratio):<8.3f} | {np.min(ana_ratio):<6.2f} | {np.max(ana_ratio):<6.2f}")
    print(f"{'Varianza Cifre':<25} | {np.mean(dig_var):<8.3f} | {np.std(dig_var):<8.3f} | {np.min(dig_var):<6.2f} | {np.max(dig_var):<6.2f}")
    print(f"{'Coesione Interna':<25} | {np.mean(int_cohesion):<8.3f} | {np.std(int_cohesion):<8.3f} | {np.min(int_cohesion):<6.2f} | {np.max(int_cohesion):<6.2f}")
    print(f"{'Costo Reale M':<25} | {np.mean(target_m):<8.3f} | {np.std(target_m):<8.3f} | {np.min(target_m):<6.2f} | {np.max(target_m):<6.2f}")

    # 3. Correlazioni di Pearson
    print("\n--- Correlazione di Pearson con il Costo Reale M ---")
    print(f"- log2(N)            -> Costo M : {np.corrcoef(log_n, target_m)[0, 1]:+.4f}")
    print(f"- Rapporto Cifre     -> Costo M : {np.corrcoef(digits_ratio, target_m)[0, 1]:+.4f} (Atteso -)")
    print(f"- Entropia Colonne   -> Costo M : {np.corrcoef(col_entropy, target_m)[0, 1]:+.4f} (Atteso -)")
    print(f"- Rapporto Anagrammi -> Costo M : {np.corrcoef(ana_ratio, target_m)[0, 1]:+.4f} (Atteso -)")
    print(f"- Varianza Cifre     -> Costo M : {np.corrcoef(dig_var, target_m)[0, 1]:+.4f} (Atteso +)")
    print(f"- Coesione Interna   -> Costo M : {np.corrcoef(int_cohesion, target_m)[0, 1]:+.4f} (Atteso +)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    dataset = load_dataset(FEATURES_DATASET_FILEPATH)
    print_exploratory_analysis(dataset)