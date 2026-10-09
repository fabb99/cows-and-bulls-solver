"""
Modulo: train_model (Script)
Fase 4 della Pipeline (Training del Regressore di Difficoltà del Bucket):
1. Carica il dataset geometrico tipizzato da data/features_dataset.json.
2. Analisi Esplorativa (EDA): calcola medie, deviazioni standard e le correlazioni
   di Pearson tra le feature geometriche del bucket e il costo reale M.
3. Addestra e confronta 3 modelli lineari con Intercetta (NumPy lstsq):
   - Modello 1: log2(N)
   - Modello 2: log2(N) + Unique Digits Ratio
   - Modello 3: log2(N) + Unique Digits Ratio + Column Entropy
4. Misura MAE, RMSE e R² per decretare il modello vincitore.
5. Stampa la formula matematica finale pronta per EstimatedEVMStrategy.
"""
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np

from experiments.oracle_lab.models import FeatureRecord

# Percorso del dataset arricchito
FEATURES_DATASET_FILEPATH = Path("data/features_dataset.json")

def load_dataset(filepath: Path) -> List[FeatureRecord]:
    """
    Carica i dati arricchiti dal file JSON e ricostruisce gli oggetti FeatureRecord.
    
    PRE: filepath punta a un file data/features_dataset.json esistente e valido.
    POST: Ritorna una lista di istanze FeatureRecord usando from_dict.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        dataset = json.load(file)     
    return [FeatureRecord.from_dict(record) for record in dataset]

def print_exploratory_analysis(dataset: List[FeatureRecord]) -> None:
    """
    Stampa la tabella statistica e le correlazioni di Pearson con il Costo Reale M.
    
    PRE: dataset non vuoto.
    POST: Stampa:
          - Media, Std, Min e Max per: N, log_n, unique_digits_ratio, column_entropy, target_cost.
          - La correlazione np.corrcoef tra ciascuna feature e target_cost.
    """
    # 1. Estrai i vettori NumPy per ciascun campo del nuovo FeatureRecord:
    #    - n_vals = np.array([r.n for r in dataset])
    #    - log_n = np.array([r.log_n for r in dataset])
    #    - digits_ratio = np.array([r.unique_digits_ratio for r in dataset])
    #    - col_entropy = np.array([r.column_entropy for r in dataset])
    #    - target_m = np.array([r.target_cost for r in dataset])
    
    # 2. Stampa la tabella formattata con np.mean, np.std, np.min, np.max.
    
    # 3. Calcola le correlazioni di Pearson rispetto a target_m usando np.corrcoef:
    #    - corr_log_n = np.corrcoef(log_n, target_m)[0, 1]
    #    - corr_digits = np.corrcoef(digits_ratio, target_m)[0, 1]
    #    - corr_col_ent = np.corrcoef(col_entropy, target_m)[0, 1]
    
    # 4. Stampa le correlazioni con l'interpretazione intuitiva.
    # 1. Estrai i vettori
    n_vals = np.array([r.n for r in dataset])
    log_n = np.array([r.log_n for r in dataset])
    digits_ratio = np.array([r.unique_digits_ratio for r in dataset])
    col_entropy = np.array([r.column_entropy for r in dataset])
    target_m = np.array([r.target_cost for r in dataset])

    # 2. Stampa la tabella formattata
    print("\n" + "=" * 70)
    print(f" ANALISI ESPLORATIVA DEI DATI DEI BUCKET - {len(dataset)} STATI")
    print("=" * 70)
    print(f"{'Feature':<25} | {'Media':<8} | {'Dev.Std':<8} | {'Min':<6} | {'Max':<6}")
    print("-" * 70)
    print(f"{'Dimensione N':<25} | {np.mean(n_vals):<8.2f} | {np.std(n_vals):<8.2f} | {np.min(n_vals):<6} | {np.max(n_vals):<6}")
    print(f"{'log2(N)':<25} | {np.mean(log_n):<8.3f} | {np.std(log_n):<8.3f} | {np.min(log_n):<6.2f} | {np.max(log_n):<6.2f}")
    print(f"{'Rapporto Cifre':<25} | {np.mean(digits_ratio):<8.3f} | {np.std(digits_ratio):<8.3f} | {np.min(digits_ratio):<6.2f} | {np.max(digits_ratio):<6.2f}")
    print(f"{'Entropia Colonne':<25} | {np.mean(col_entropy):<8.3f} | {np.std(col_entropy):<8.3f} | {np.min(col_entropy):<6.2f} | {np.max(col_entropy):<6.2f}")
    print(f"{'Costo Reale M':<25} | {np.mean(target_m):<8.3f} | {np.std(target_m):<8.3f} | {np.min(target_m):<6.2f} | {np.max(target_m):<6.2f}")

    # 3. Correlazione di Pearson con il Costo M
    print("\n--- Correlazione di Pearson con il Costo Reale M ---")
    print(f"- log2(N)          -> Costo M : {np.corrcoef(log_n, target_m)[0, 1]:+.4f}")
    print(f"- Rapporto Cifre   -> Costo M : {np.corrcoef(digits_ratio, target_m)[0, 1]:+.4f} (Ci aspettiamo un segno NEGATIVO!)")
    print(f"- Entropia Colonne -> Costo M : {np.corrcoef(col_entropy, target_m)[0, 1]:+.4f} (Ci aspettiamo un segno NEGATIVO!)")
    print("=" * 70 + "\n")

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



def main():
    """
    Punto di ingresso principale della Fase 4:
    1. Carica il dataset tipizzato da FEATURES_DATASET_FILEPATH con load_dataset.
    2. Stampa l'EDA completa chiamando print_exploratory_analysis.
    3. Prepara il vettore target Y (i costi reali target_cost).
    4. Allena il MODELLO 1 (Solo Entropia Residua con Intercetta: Costo = K0 + K1 * Delta_H).
    5. Allena il MODELLO 2 (Entropia Residua + N_max con Intercetta: Costo = K0 + K1 * Delta_H + K2 * N_max).
    6. Stampa una tabella comparativa affiancando i due modelli con le metriche MAE, RMSE e R².
    7. Stampa la formula matematica definitiva pronta da copiare dentro EstimatedEVMStrategy.
    """
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

    print_exploratory_analysis(dataset)

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
    print(" CONFRONTO TRA I 3 MODELLI GEOMETRICI DEL BUCKET")
    print("=" * 70)
    print(f"{'Modello':<35} | {'MAE (Errore)':<12} | {'R² (Accuratezza)':<15}")
    print("-" * 70)
    print(f"{'1. Solo log2(N)':<35} | ±{metrics1[0]:<11.3f} | {metrics1[2] * 100:<14.2f}%")
    print(f"{'2. log2(N) + Rapporto Cifre':<35} | ±{metrics2[0]:<11.3f} | {metrics2[2] * 100:<14.2f}%")
    print(f"{'3. Geometria Completa (Tutte e 3)':<35} | ±{metrics3[0]:<11.3f} | {metrics3[2] * 100:<14.2f}%")
    print("=" * 70)

    # 8. Formula Definitiva per il Modello Completo (Modello 3)
    k0, k1, k2, k3 = coeff3[0], coeff3[1], coeff3[2], coeff3[3]
    print("\n" + "#" * 70)
    print(" COEFFICIENTI DEFINITIVI PER _estimate_bucket_cost")
    print("#" * 70)
    print(f"K0_INTERCEPT: float = {k0:.4f}")
    print(f"K1_LOG_N: float     = {k1:.4f}")
    print(f"K2_DIGITS: float    = {k2:.4f}")
    print(f"K3_COLUMNS: float   = {k3:.4f}")
    print("\nFormula: Costo = K0 + (K1 * log2(n)) + (K2 * digits_ratio) + (K3 * col_entropy)")
    print("#" * 70 + "\n")



if __name__ == "__main__":
    main()