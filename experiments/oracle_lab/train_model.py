"""
Modulo: 3_train_model (Script)
Fase 3 della Pipeline:
Analisi statistica (EDA), Correlazione, Training a minimi quadrati (NumPy) 
e Valutazione dell'errore (MAE, RMSE, R2) per estrarre le costanti ottimali.
"""
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np

DATASET_FILEPATH = Path("data/features_dataset.json")


def load_dataset(filepath: Path) -> List[Dict[str, Any]]:
    """Carica il dataset arricchito."""
    if not filepath.exists():
        raise FileNotFoundError(f"Errore: {filepath} non trovato. Esegui prima 2_feature_builder.py!")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def print_eda(dataset: List[Dict[str, Any]]) -> None:
    """Stampa l'Analisi Esplorativa dei Dati e la Matrice di Correlazione."""
    n_samples = len(dataset)
    n_vals = np.array([d["n"] for d in dataset])
    res_ent = np.array([d["residual_entropy"] for d in dataset])
    n_max = np.array([d["n_max"] for d in dataset])
    target_m = np.array([d["target_m"] for d in dataset])

    print("\n" + "=" * 70)
    print(f" ANALISI ESPLORATIVA DEI DATI (EDA) - {n_samples} STATI TOTALI")
    print("=" * 70)
    print(f"{'Feature':<25} | {'Media':<8} | {'Dev.Std':<8} | {'Min':<6} | {'Max':<6}")
    print("-" * 70)
    print(f"{'Dimensione N':<25} | {np.mean(n_vals):<8.2f} | {np.std(n_vals):<8.2f} | {np.min(n_vals):<6} | {np.max(n_vals):<6}")
    print(f"{'Entropia Residua':<25} | {np.mean(res_ent):<8.3f} | {np.std(res_ent):<8.3f} | {np.min(res_ent):<6.2f} | {np.max(res_ent):<6.2f}")
    print(f"{'N_max (Caso Peggiore)':<25} | {np.mean(n_max):<8.2f} | {np.std(n_max):<8.2f} | {np.min(n_max):<6} | {np.max(n_max):<6}")
    print(f"{'Costo Reale M':<25} | {np.mean(target_m):<8.3f} | {np.std(target_m):<8.3f} | {np.min(target_m):<6.2f} | {np.max(target_m):<6.2f}")

    # Correlazione di Pearson con il Costo M
    corr_ent = np.corrcoef(res_ent, target_m)[0, 1]
    corr_nmax = np.corrcoef(n_max, target_m)[0, 1]

    print("\n--- Correlazione di Pearson con il Costo Reale M ---")
    print(f"- Entropia Residua -> Costo M : {corr_ent:+.4f} (Più è vicino a +1, più è predittivo)")
    print(f"- N_max            -> Costo M : {corr_nmax:+.4f} (Più è vicino a +1, più è predittivo)")
    print("=" * 70 + "\n")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Calcola MAE (Errore Medio Assoluto), RMSE e R^2 (Bontà di Adattamento)."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_residual / ss_total)
    return mae, rmse, r2


def main():
    print("Caricamento dataset arricchito...")
    dataset = load_dataset(DATASET_FILEPATH)
    
    # 1. Stampa statistiche e correlazioni
    print_eda(dataset)

    # 2. Preparazione Matrici per NumPy
    # Target: Y = M - 1.0 (perché 1.0 è la mossa corrente)
    Y = np.array([d["target_m"] - 1.0 for d in dataset])
    X_ent = np.array([d["residual_entropy"] for d in dataset])
    X_nmax = np.array([d["n_max"] for d in dataset])

    # ----------------------------------------------------
    # MODELLO 1: Solo Entropia Residua (Y = K1 * Ent)
    # ----------------------------------------------------
    X_mod1 = X_ent[:, np.newaxis]
    k1_only, _, _, _ = np.linalg.lstsq(X_mod1, Y, rcond=None)
    pred_m1 = 1.0 + (k1_only[0] * X_ent)
    mae_1, rmse_1, r2_1 = compute_metrics(np.array([d["target_m"] for d in dataset]), pred_m1)

    Y_real = np.array([d["target_m"] for d in dataset])
    
    # Creiamo la matrice con una colonna di 1 (per K0) e la colonna dell'Entropia (per K1)
    X_with_intercept = np.column_stack([np.ones_like(X_ent), X_ent])
    
    # Risolviamo
    coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, Y_real, rcond=None)
    k0 = coeffs[0]
    k1 = coeffs[1]
    
    pred_real = k0 + (k1 * X_ent)
    mae, rmse, r2 = compute_metrics(Y_real, pred_real)
    
    print("\n" + "=" * 70)
    print("MODELLO OTTIMIZZATO CON INTERCETTA (K0 + K1 * Ent)")
    print("=" * 70)
    print(f"MAE (Errore Medio)  : ±{mae:.3f} mosse")
    print(f"R² (Accuratezza)    : {r2 * 100:.2f}%")
    print(f"\nCostante Base (K0)  : {k0:.4f}")
    print(f"Moltiplicatore (K1) : {k1:.4f}")
    print(f"\nFormula: Costo = {k0:.4f} + ({k1:.4f} * residual_entropy)")
    print("=" * 70)




if __name__ == "__main__":
    main()