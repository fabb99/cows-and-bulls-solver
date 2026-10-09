"""
Modulo: find_optimal_splits (Script)
Fase Analitica del Laboratorio (Algoritmo CART 1D per Trovare i Punti di Taglio):
Analizza la dispersione del Costo Reale (M) rispetto alla Dimensione del Bucket (N).
Esegue una scansione esaustiva di tutti i possibili valori di soglia T ed estrae 
matematicamente i confini che minimizzano la varianza residua (Sum of Squared Errors).
Output: Le soglie ottimali T1, T2 per dividere il gioco in scaglioni matematicamente giustificati.
"""
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np

from experiments.oracle_lab.models import FeatureRecord

# Percorso del dataset con i dati geometrici e i costi reali
FEATURES_DATASET_FILEPATH = Path("data/features_dataset.json")


def load_dataset(filepath: Path) -> List[FeatureRecord]:
    """
    Carica i FeatureRecord da data/features_dataset.json.
    
    PRE: filepath punta a un file JSON valido.
    POST: Ritorna una lista di istanze FeatureRecord.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        dataset = json.load(file)     
    return [FeatureRecord.from_dict(record) for record in dataset]

def compute_sum_squared_errors(y: np.ndarray) -> float:
    """
    Calcola la Dispersione Interna (Somma degli Errori al Quadrato rispetto alla Media):
    SS = sum((y - mean(y))^2)
    
    PRE: y è un array NumPy 1D di valori numerici.
    POST: Ritorna la somma degli scarti quadratici (float >= 0.0). Se len(y) <= 1, ritorna 0.0.
    """
    y_pred = np.mean(y)
    return np.sum((y - y_pred)**2)

def find_best_single_split(
    n_values: np.ndarray, 
    costs: np.ndarray, 
    min_leaf_samples: int = 50
) -> Tuple[Optional[int], float, float]:
    """
    Esegue l'algoritmo CART per trovare la migliore soglia singola T* che divide i dati in due stanze.
    
    PRE: 
        - n_values: array 1D con le dimensioni dei bucket (N).
        - costs: array 1D con i costi reali corrispondenti (M).
        - min_leaf_samples: numero minimo di campioni che devono rimanere in ciascuna stanza.
    POST: Ritorna (miglior_soglia_T, percentuale_varianza_spiegata, errore_minimo_trovato):
          - miglior_soglia_T: il valore intero di N che minimizza la somma degli errori.
          - percentuale_varianza_spiegata: quanto disordine viene eliminato da questo taglio.
    """
    # 1. Calcola l'errore totale iniziale senza nessun taglio: SS_tot = compute_sum_squared_errors(costs)
    ss_tot = compute_sum_squared_errors(costs)
    # 2. Estrai tutti i valori unici e ordinati di N presenti nei dati (i candidati di taglio)
    possible_th = set(x for x in n_values)
    # 3. Inizializza best_threshold = None, min_total_error = infinito
    best_threshold = None
    min_total_error = np.inf
    # 4. Per ogni valore candidato T:
    #    a. Crea la maschera booleana: sinistra = (n_values <= T), destra = (n_values > T)
    #    b. Verifica che sia la stanza sinistra che quella destra abbiano almeno min_leaf_samples campioni.
    #    c. Calcola l'errore della stanza sinistra: SS_sinistra = compute_sum_squared_errors(costs[sinistra])
    #    d. Calcola l'errore della stanza destra: SS_destra = compute_sum_squared_errors(costs[destra])
    #    e. Errore totale del taglio = SS_sinistra + SS_destra
    #    f. Se errore_totale < min_total_error:
    #       - Aggiorna min_total_error = errore_totale
    #       - Aggiorna best_threshold = T
    for t in possible_th:
        left = (n_values <= t)
        
        right = (n_values > t)
        
        if len(costs[left]) >= min_leaf_samples and len(costs[right]) >= min_leaf_samples:
            ss_left = compute_sum_squared_errors(costs[left])
            ss_right = compute_sum_squared_errors(costs[right])
            total_error = ss_left + ss_right
            if total_error < min_total_error:
                min_total_error = total_error
                best_threshold = t

    # 5. Calcola la riduzione percentuale di errore: guadagno = 1.0 - (min_total_error / SS_tot)
    profit = 1.0 - (min_total_error/ss_tot)

    # 6. Ritorna (best_threshold, guadagno, min_total_error)
    return (best_threshold, profit, min_total_error)


def find_hierarchical_splits(
    n_values: np.ndarray, 
    costs: np.ndarray, 
    max_splits: int = 2
) -> List[Tuple[int, float]]:
    """
    Applica l'algoritmo di taglio ricorsivamente per trovare 2 o più confini naturali.
    
    PRE: max_splits indica quanti confini cercare (es. 2 tagli creano 3 scaglioni).
    POST: Ritorna una lista ordinata di interi con le soglie ottimali [T1, T2, ...].
    """
    # Caso base: se non ci sono più tagli da fare o se i dati sono troppo pochi
    if max_splits <= 0 or len(n_values) < 100:
        return []
        
    # 1. Trova il taglio migliore per i dati correnti
    t, profit, _ = find_best_single_split(n_values, costs)
    if t is None:
        return []
        
    # Se volevamo solo 1 taglio, abbiamo finito!
    if max_splits == 1:
        return [(t, profit)]
        
    # 2. Se vogliamo più tagli, prendiamo la parte destra (N > t) e cerchiamo il prossimo confine!
    right_mask = (n_values > t)
    sub_splits = find_hierarchical_splits(n_values[right_mask], costs[right_mask], max_splits - 1)
    
    # 3. Uniamo il taglio t con i tagli successivi trovati a destra
    return sorted([(t, profit)] + sub_splits,  key=lambda x: x[0])


def main():
    """
    Punto di ingresso principale:
    1. Carica il dataset da FEATURES_DATASET_FILEPATH.
    2. Estrai gli array n_values e costs.
    3. Esegui la scansione per trovare la prima soglia principale T1 e stampa il guadagno di varianza.
    4. Esegui la scansione gerarchica per trovare 2 o 3 soglie ottimali (es. Endgame, Midgame, Opening).
    5. Stampa la mappa definitiva degli scaglioni:
       - Scaglione 1 (Endgame): N <= T1
       - Scaglione 2 (Midgame): T1 < N <= T2
       - Scaglione 3 (Opening): N > T2
    6. Mostra per ciascuno scaglione quanti campioni contiene e il costo medio.
    """
    dataset = load_dataset(FEATURES_DATASET_FILEPATH) 

    n_values = np.array([state.n for state in dataset]) 

    y_targets = np.array([state.target_cost for state in dataset])


    splits = find_hierarchical_splits(n_values, y_targets, max_splits=4)

    print("\n" + "=" * 65)
    print(" SOGLIE MATEMATICHE E GUADAGNO DI VARIANZA (CART)")
    print("=" * 65)
    
    for i, (threshold, gain) in enumerate(splits, start=1):
        print(f"Taglio {i}: Soglia a N = {threshold:<4} | Varianza Eliminata: {gain * 100:.2f}%")
        
    print("-" * 65)
    
    if len(splits) >= 2:
        t1 = splits[0][0]
        t2 = splits[1][0]
        print(f"1. SCAGLIONE ENDGAME : N <= {t1}")
        print(f"2. SCAGLIONE MIDGAME : {t1} < N <= {t2}")
        print(f"3. SCAGLIONE OPENING : N > {t2}")
        
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()