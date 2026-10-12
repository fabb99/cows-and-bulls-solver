"""
Modulo: train_model (Script)
Fase 4 della Pipeline (Training del Regressore di Difficoltà del Bucket):
1. Carica il dataset geometrico tipizzato da data/features_dataset.json.
2. Analisi Esplorativa (EDA): calcola medie, deviazioni standard e le correlazioni
   di Pearson tra le feature geometriche del bucket e il costo reale M.
3. Addestra un modelli lineare con Intercetta (NumPy lstsq):
    Modello: log2(N) + Column Entropy + Digit std
4. Misura MAE, RMSE e R^2 per decretare il modello vincitore.
5. Stampa la formula matematica finale pronta per EstimatedEVMStrategy.
"""
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np

from experiments.oracle_lab.models import FeatureRecord

# Percorso del dataset arricchito
FEATURES_DATASET_FILEPATH = Path("data/features_dataset.json")

SPLIT_ENDGAME: int = 13
SPLIT_MIDGAME: int = 40

def load_dataset(filepath: Path) -> List[FeatureRecord]:
    """
    Carica i dati arricchiti dal file JSON e ricostruisce gli oggetti FeatureRecord.
    
    PRE: filepath punta a un file data/features_dataset.json esistente e valido.
    POST: Ritorna una lista di istanze FeatureRecord usando from_dict.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        dataset = json.load(file)     
    return [FeatureRecord.from_dict(record) for record in dataset]

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcola le metriche oggettive di errore e di spiegazione della varianza.
    
    PRE: y_true e y_pred sono array NumPy 1D della stessa dimensione.
    POST: Ritorna la tripla (MAE, RMSE, R2):
          - MAE: Errore medio assoluto in turni (np.mean(np.abs(y_true - y_pred)))
          - RMSE: Radice dell'errore quadratico medio (np.sqrt(np.mean((y_true - y_pred)**2)))
          - R2: Coefficiente di determinazione (1.0 - (SS_res / SS_tot))
    """
    # 1. Calcola MAE (L1 loss)
    mae = np.mean(np.abs(y_true - y_pred))
    # 2. Calcola RMSE (L2 loss)
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))
    # 3. Calcola la somma dei quadrati degli errori residui
    ss_res = np.sum((y_true - y_pred)**2)
    # 4. Calcola la somma dei quadrati della varianza totale rispetto alla media
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    # 5. Calcola R2 = 1.0 - (SS_res / SS_tot)
    r2 = 1.0 - (ss_res/ss_tot)
    # 6. Ritorna (mae, rmse, r2)
    return mae, rmse, r2

def train_linear_model(
    X_features: np.ndarray, 
    y_targets: np.ndarray
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Risolve il sistema OLS con intercetta automatica K0 tramite NumPy SVD.
    
    PRE: 
        - X_features è una matrice NumPy (N_samples, N_features).
        - y_targets è un vettore 1D con i costi reali M.
    POST: Ritorna (coefficienti, (mae, rmse, r2)).
    """
    # 1. Crea la colonna di 1 con np.ones_like(y_targets)
    # 2. Incolla con np.column_stack([colonna_uno, X_features])
    x = np.column_stack([np.ones_like(y_targets), X_features])
    # 3. Risolvi con np.linalg.lstsq(..., rcond=None)[0]
    coeffs = np.linalg.lstsq(x, y_targets, rcond=None)[0]
    # 4. Calcola le predizioni con l'operatore matriciale @
    y_pred = x @ coeffs
    # 5. Calcola le metriche con compute_regression_metrics(y_targets, y_pred)
    metrics = compute_regression_metrics(y_targets, y_pred)
    # 6. Ritorna (coeffs, metrics)
    return (coeffs, metrics)

def train_segment(
    segment_name: str, 
    records: List[FeatureRecord]
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Allena e valuta il Modello Geometrico (Modello 3) su uno specifico scaglione di dati.
    
    PRE: segment_name è una stringa descrittiva. records è una lista non vuota di FeatureRecord.
    POST: Ritorna (coefficienti_K, metriche_(mae, rmse, r2)) e stampa a terminale 
          la scheda di performance dello scaglione.
    """
    log_n = [r.log_n for r in records]
    digits_std = [r.digit_variance for r in records]
    col_entropy = [r.column_entropy for r in records]
    target_m = [r.target_cost for r in records]

    X = np.column_stack([log_n, digits_std, col_entropy])

    coeff, metrics = train_linear_model(X,target_m)
    
    # Stampa tabella con: Numero Campioni, MAE, R², e i 4 coefficienti:
    print(f"{'Numero':<12} | {'MAE (Errore)':<12} | {'R² (Accuratezza)':<15}")
    print(f"{len(records):<12} | ±{metrics[0]:<11.3f} | {metrics[2] * 100:<14.2f}%")

    k0, k1, k2, k3 = coeff[0], coeff[1], coeff[2], coeff[3]
    print("\n" + "#" * 70)
    print(f" COEFFICIENTI DEFINITIVI PER {segment_name}")
    print("#" * 70)
    print(f"K0_INTERCEPT: float = {k0:.4f}")
    print(f"K1_LOG_N: float     = {k1:.4f}")
    print(f"K2_DIGITS: float    = {k2:.4f}")
    print(f"K3_COLUMNS: float   = {k3:.4f}")
    print("\nFormula: Costo = K0 + (K1 * log2(n)) + (K2 * digits_ratio) + (K3 * col_entropy)")
    print("#" * 70 + "\n")
    # 5. Ritorna (coeffs, metrics)
    return coeff, metrics


def main():
    """
    Punto di ingresso principale della Fase 4:
    1. Carica il dataset da FEATURES_DATASET_FILEPATH.
    2. Stampa l'Analisi Esplorativa (EDA) completa con le correlazioni.
    3. Estrai i vettori di feature: log_n, digits_ratio, col_entropy e y_targets.
    4. Allena il MODELLO 1: solo log_n.
    5. Allena il MODELLO 2: np.column_stack([log_n, digits_ratio]).
    6. Allena il MODELLO 3: np.column_stack([log_n, digits_ratio, col_entropy]).
    7. Stampa la tabella comparativa che mette a confronto MAE e R² dei 3 modelli.
    8. Stampa l'equazione definitiva del modello vincitore con i coefficienti trovati!
    """
    dataset = load_dataset(FEATURES_DATASET_FILEPATH) 



    x_log_n = np.array([state.log_n for state in dataset])
    digits_radio = np.array([state.unique_digits_ratio for state in dataset])
    x_col_entropy = np.array([state.column_entropy for state in dataset])
    
    y_targets = np.array([state.target_cost for state in dataset])


    #MODELLO 1
    coeff1, metrics1 = train_linear_model(x_log_n, y_targets)

    #MODELLO 2
    coeff2, metrics2 = train_linear_model(np.column_stack([x_log_n, digits_radio]), y_targets)

    #MODELLO 3
    coeff3, metrics3 = train_linear_model(np.column_stack([x_log_n, digits_radio, x_col_entropy]), y_targets)

    # 7. Tabella Comparativa dei 3 Modelli
    print("=" * 70)
    print(" TABELLA PERFORMANCE")
    print("=" * 70)
    print(f"{'Modello':<35} | {'MAE (Errore)':<12} | {'R² (Accuratezza)':<15}")
    print(f"{'log2(N) + Column Entropy + Digit std':<35} | ±{metrics3[0]:<11.3f} | {metrics3[2] * 100:<14.2f}%")
    print("=" * 70)

    # 8. Formula Definitiva per il Modello Completo (Modello 3)
    k0, k1, k2, k3 = coeff3[0], coeff3[1], coeff3[2], coeff3[3]
    print("\n" + "#" * 70)
    print(" COEFFICIENTI DEFINITIVI")
    print("#" * 70)
    print(f"K0_INTERCEPT: float = {k0:.4f}")
    print(f"K1_LOG_N: float     = {k1:.4f}")
    print(f"K2_DIGITS: float    = {k2:.4f}")
    print(f"K3_COLUMNS: float   = {k3:.4f}")
    print("\nFormula: Costo = K0 + (K1 * log2(n)) + (K2 * digits_ratio) + (K3 * col_entropy)")
    print("#" * 70 + "\n")

    dataset_opening = [d for d in dataset if d.n > SPLIT_MIDGAME]  
    dataset_midgame = [d for d in dataset if SPLIT_ENDGAME < d.n <= SPLIT_MIDGAME] 
    dataset_endgame = [d for d in dataset if d.n <= SPLIT_ENDGAME] 

    dict = {"OPENING" : dataset_opening, "MIDGAME": dataset_midgame, "ENDGAME": dataset_endgame} 

    for k, v in dict.items():
        train_segment(k, v)


if __name__ == "__main__":
    main()