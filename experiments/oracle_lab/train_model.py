"""
Modulo: train_model 
"""
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np

from experiments.oracle_lab.models import FeatureRecord

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
    # 3. Calcola la somma dei quadrati degli errori residui: SS_res = np.sum((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred)**2)
    # 4. Calcola la somma dei quadrati della varianza totale rispetto alla media: SS_tot = np.sum((y_true - np.mean(y_true)) ** 2)
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
