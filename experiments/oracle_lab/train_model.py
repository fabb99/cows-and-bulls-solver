"""
Modulo: train_evaluator (Script)
Fase 4 della Pipeline (Training e Valutazione del Modello):
1. Carica il dataset tipizzato (List[FeatureRecord]) da data/features_dataset.json.
2. Esegue l'Analisi Esplorativa dei Dati (EDA) e calcola le correlazioni di Pearson.
3. Addestra e confronta due modelli lineari tramite Decomposizione a Valori Singolari (NumPy SVD):
   - Modello A: Intercetta + Entropia Residua
   - Modello B: Intercetta + Entropia Residua + Rischio N_max
4. Misura le performance su scala reale tramite MAE, RMSE e R² (Coefficiente di Determinazione).
5. Stampa la formula matematica definitiva pronta da incollare in EstimatedEVMStrategy.
"""
from typing import List, Tuple
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
    Stampa l'Analisi Esplorativa dei Dati (EDA) e le matrici di correlazione lineare.
    
    PRE: dataset è una lista non vuota di FeatureRecord.
    POST: Stampa a terminale una tabella ASCII con:
          - Media, Deviazione Standard, Minimo e Massimo per ciascuna variabile (N, Delta H, N_max, M).
          - La Correlazione di Pearson (np.corrcoef) tra ciascuna feature e il costo reale target_cost.
    """
    # 1. Estrai gli array NumPy 1D per ciascun campo:
    #    - n_vals = np.array([r.n for r in dataset])
    #    - res_ent = np.array([r.residual_entropy for r in dataset])
    #    - n_max = np.array([r.n_max for r in dataset])
    #    - target_m = np.array([r.target_cost for r in dataset])
    n_vals = np.array([np.log2(r.n) for r in dataset])
    res_ent = np.array([r.residual_entropy for r in dataset])
    n_max = np.array([np.log2(r.n_max) for r in dataset])
    target_m = np.array([r.target_cost for r in dataset])

    # 2. Stampa la tabella formattata calcolando np.mean, np.std, np.min, np.max per ciascun vettore.
    print("-" * 70)
    print(f"{'Feature':<25} | {'Media':<8} | {'Dev.Std':<8} | {'Min':<6} | {'Max':<6}")
    print("-" * 70)
    print(f"{'Dimensione N':<25} | {np.mean(n_vals):<8.2f} | {np.std(n_vals):<8.2f} | {np.min(n_vals):<6} | {np.max(n_vals):<6}")
    print(f"{'Entropia Residua':<25} | {np.mean(res_ent):<8.3f} | {np.std(res_ent):<8.3f} | {np.min(res_ent):<6.2f} | {np.max(res_ent):<6.2f}")
    print(f"{'N_max (Caso Peggiore)':<25} | {np.mean(n_max):<8.2f} | {np.std(n_max):<8.2f} | {np.min(n_max):<6} | {np.max(n_max):<6}")
    print(f"{'Costo Reale M':<25} | {np.mean(target_m):<8.3f} | {np.std(target_m):<8.3f} | {np.min(target_m):<6.2f} | {np.max(target_m):<6.2f}")
    print("-" * 70)

    # 3. Calcola la correlazione di Pearson (r) tra res_ent e target_m, e tra n_max e target_m usando np.corrcoef.
    #    Ricorda: np.corrcoef(X, Y)[0, 1] estrae il valore scalare tra -1 e +1.
    
    perason_corr_n = np.corrcoef(n_vals, target_m)[0,1]
    pearson_corr_res_ent = np.corrcoef(res_ent, target_m)[0,1]
    perason_corr_n_max = np.corrcoef(n_max, target_m)[0,1]
    
    # 4. Stampa le correlazioni con una breve interpretazione qualitativa.
    print("-" * 70)
    print("Correlazione di Pearson con il Costo Reale M (Più è vicino a +1, più è predittivo)")
    print(f"- N                -> Costo M : {perason_corr_n:+.4f} ")
    print(f"- Entropia Residua -> Costo M : {pearson_corr_res_ent:+.4f} ")
    print(f"- N_max            -> Costo M : {perason_corr_n_max:+.4f} ")
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
    y_targets: np.ndarray, 
    include_intercept: bool = True
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Risolve il sistema lineare ai minimi quadrati (OLS) tramite Decomposizione a Valori Singolari (SVD).
    
    PRE: 
        - X_features è una matrice NumPy (N_samples, N_features).
        - y_targets è un vettore NumPy 1D con i valori target reali (M).
        - include_intercept specifica se inserire il termine di bias (K0).
    POST: Ritorna una tupla contenente:
          - Il vettore dei coefficienti ottimali trovati da NumPy.
          - La tupla delle metriche (MAE, RMSE, R2).
    """
    # 1. Se include_intercept è True:
    #    Crea una colonna di soli '1' della stessa lunghezza di X (es. np.ones_like...)
    #    Concatena orizzontalmente la colonna di 1 alle feature usando np.column_stack.
    #    Se False, usa direttamente X_features.
    if include_intercept:
        array_ones = np.ones_like(y_targets)
        x_features = np.column_stack([array_ones, X_features])
    else: 
        x_features = X_features
    # 2. Risolvi il sistema ai minimi quadrati:
    #    coeffs = np.linalg.lstsq(X_con_intercetta, y_targets, rcond=None)[0]
    coeff= np.linalg.lstsq(x_features,y_targets, rcond=None)[0]
    # 3. Calcola le previsioni del modello moltiplicando la matrice per i coefficienti trovati (prodotto scalare).
    y_pred = x_features @ coeff
    # 4. Calcola le metriche chiamando compute_regression_metrics(y_targets, y_pred).
    metrics = compute_regression_metrics(y_targets, y_pred)
    # 5. Ritorna (coeffs, metrics)
    return coeff, metrics


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
    dataset = load_dataset(FEATURES_DATASET_FILEPATH) 

    print_exploratory_analysis(dataset)

    x_n = np.array([np.log2(state.n) for state in dataset])
    x_n_max = np.array([np.log2(state.n_max) for state in dataset])
    x_residual_entropy = np.array([state.residual_entropy for state in dataset])
    
    y_targets = np.array([state.target_cost for state in dataset])

    #MODELLO per ora solo uno
    coeff, metrics = train_linear_model(np.column_stack([x_n, x_n_max,x_residual_entropy]), y_targets)

    print(f"Coefficenti =  {coeff}")
    print(f"metriche = {metrics}" )



if __name__ == "__main__":
    main()