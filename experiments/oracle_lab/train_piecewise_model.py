"""
Modulo: train_model (Script)
Fase 4 della Pipeline:
Analisi descrittiva (EDA) e Regressione Asimmetrica (Asymmetric Least Squares - IRLS).
Penalizza severamente le sottostime dei bucket difficili per evitare che l'IA 
scelga mosse rischiose che finiscono al Turno 7.
"""
from typing import List, Tuple
from pathlib import Path
import json
import numpy as np

from experiments.oracle_lab.models import FeatureRecord

FEATURES_DATASET_FILEPATH = Path("data/features_dataset.json")
OUTPUT_FILEPATH = Path("data/models/piecewise_model.json")

# ---------------------------------------------------------------------------
# CONFIGURAZIONE ASIMMETRIA (La manopola di controllo)
# ---------------------------------------------------------------------------
# alpha = 0.50 -> Regressione standard simmetrica (quella di prima)
# alpha = 0.80 -> Punisce le sottostime 4 volte più delle sovrastime (0.80 / 0.20 = 4x)
# alpha = 0.85 -> Punisce le sottostime 5.6 volte più delle sovrastime (molto conservativo)
ALPHA_PENALTY: float = 0.50

SPLIT_ENDGAME: int = 13
SPLIT_MIDGAME: int = 43

def load_dataset(filepath: Path) -> List[FeatureRecord]:
    """Carica i dati arricchiti dal file JSON."""
    with open(filepath, "r", encoding="utf-8") as file:
        dataset = json.load(file)     
    return [FeatureRecord.from_dict(record) for record in dataset]

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Calcola MAE, RMSE e R²."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    return mae, rmse, r2

def train_linear_model(
    X_features: np.ndarray, 
    y_targets: np.ndarray,
    alpha: float = ALPHA_PENALTY,
    max_iter: int = 15
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Risolve il sistema lineare. Se alpha != 0.5, applica la regressione asimmetrica
    tramite Minimi Quadrati Ponderati Iterati (IRLS).
    """
    # 1. Aggiungiamo la colonna di 1 per l'intercetta K0
    X = np.column_stack([np.ones_like(y_targets), X_features])
    
    # 2. Se alpha è 0.50, eseguiamo la classica regressione OLS simmetrica
    if alpha == 0.50:
        coeffs = np.linalg.lstsq(X, y_targets, rcond=None)[0]
        y_pred = X @ coeffs
        metrics = compute_regression_metrics(y_targets, y_pred)
        return coeffs, metrics

    # 3. REGRESSIONE ASIMMETRICA (IRLS - Iteratively Reweighted Least Squares)
    # Partiamo da una stima OLS iniziale
    coeffs = np.linalg.lstsq(X, y_targets, rcond=None)[0]

    for _ in range(max_iter):
        y_pred = X @ coeffs
        residuals = y_targets - y_pred  # > 0 significa sottostima (la realtà è peggiore della stima)
        
        # Pesi asimmetrici: se sottostimiamo applichiamo alpha, altrimenti (1 - alpha)
        weights = np.where(residuals > 0, alpha, 1.0 - alpha)
        
        # Risoluzione pesata (Weighted Least Squares)
        sqrt_w = np.sqrt(weights)
        X_weighted = X * sqrt_w[:, np.newaxis]
        y_weighted = y_targets * sqrt_w
        
        coeffs = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]

    # Previsioni finali sui target originali
    y_pred_final = X @ coeffs
    metrics = compute_regression_metrics(y_targets, y_pred_final)
    return coeffs, metrics

def train_segment(
    segment_name: str, 
    records: List[FeatureRecord]
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    
    # 1. Estrazione di tutti i vettori
    log_n = [r.log_n for r in records]
    col_entropy = [r.column_entropy for r in records]   
    dig_var = [r.digit_variance for r in records]
    
    target_m = [r.target_cost for r in records]

    # 2. La Matrice con le 3 Feature (NumPy farà la magia)
    X = np.column_stack([log_n, col_entropy, dig_var])

    coeff, metrics = train_linear_model(X, target_m)

    print(f"{'Numero':<12} | {'MAE (Errore)':<12} | {'R² (Accuratezza)':<15}")
    print(f"{len(records):<12} | ±{metrics[0]:<11.3f} | {metrics[2] * 100:<14.2f}%")

    # Estrazione dei 4 coefficienti (K0 intercetta + K1...K4)
    k0, k1, k2, k3 =coeff[0], coeff[1], coeff[2], coeff[3]
    
    print("\n" + "#" * 70)
    print(f" COEFFICIENTI DEFINITIVI PER {segment_name}")
    print("#" * 70)
    print(f"K0_INTERCEPT: float = {k0:.4f}")
    print(f"K1_LOG_N: float     = {k1:.4f}")
    print(f"K2_COLUMNS: float    = {k2:.4f}")
    print(f"K3_DIG_VAR: float   = {k3:.4f}")
    print("#" * 70 + "\n")

    # In train_segment:
    segment_dict = {
        "k0": float(k0),
        "k1_log_n": float(k1),
        "k2_columns": float(k2),
        "k3_digit_std": float(k3)
    }
    return segment_dict, metrics
    
def main():
    dataset = load_dataset(FEATURES_DATASET_FILEPATH)
    
    # Fasi
    dataset_endgame = [d for d in dataset if d.n <= SPLIT_ENDGAME]
    dataset_midgame = [d for d in dataset if SPLIT_ENDGAME < d.n <= SPLIT_MIDGAME]
    dataset_opening = [d for d in dataset if d.n > SPLIT_MIDGAME]

    segments = {
        "ENDGAME": dataset_endgame,
        "MIDGAME": dataset_midgame,
        "OPENING": dataset_opening
    }

    print("\n" + "=" * 70)
    print(f"TRAINING A FASI CON REGRESSIONE ASIMMETRICA (ALPHA = {ALPHA_PENALTY})")
    print("=" * 70)

    coeffs_to_save = {
        "split_endgame": SPLIT_ENDGAME,
        "split_midgame": SPLIT_MIDGAME
    }
    
    # Mappiamo i nomi corretti attesi dall'estimatore
    name_mapping = {
        "ENDGAME": "k_endgame",
        "MIDGAME": "k_midgame",
        "OPENING": "k_opening"
    }

    for name, subset in segments.items():
        segment_dict, _ = train_segment(name, subset)
        coeffs_to_save[name_mapping[name]] = segment_dict

    OUTPUT_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILEPATH, "w", encoding="utf-8") as file:
        json.dump(coeffs_to_save, file, indent=4)
        
    print(f"Modello a scaglioni salvato con successo in {OUTPUT_FILEPATH}!")


if __name__ == "__main__":
    main()