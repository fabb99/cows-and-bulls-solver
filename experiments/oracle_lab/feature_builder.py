"""
Modulo: feature_builder (Script)
Fase 2 della Pipeline (Feature Engineering):
Legge i bivi da data/raw_states.json (usando RawState) e per ciascuno calcola
le feature matematiche della Teoria dell'Informazione:
- Incertezza Residua (log2(N) - H)
- Rischio Coda (N_max)
- Target reale (best_cost)
Salva il dataset finale tipizzato (FeatureRecord) in data/features_dataset.json.
"""
from typing import List, Dict, Any, Set
from pathlib import Path
import json
import math
from tqdm import tqdm

from experiments.oracle_lab.models import RawState, FeatureRecord
from experiments.oracle_lab.data_miner import load_raw_states

from src.core.constants import DIM , BASE

# Percorsi e configurazioni
RAW_STATES_FILEPATH = Path("data/raw_states.json")
FEATURES_DATASET_FILEPATH = Path("data/features_dataset.json")
SAVE_INTERVAL: int = 100      # Salva su disco ogni N stati per sicurezza


def load_features_dataset(filepath: Path) -> List[FeatureRecord]:
    """
    Carica la lista dei FeatureRecord già calcolati da file JSON (se esiste).
    
    PRE: filepath punta all'eventuale file data/features_dataset.json.
    POST: Ritorna una lista di istanze FeatureRecord usando from_dict, altrimenti lista vuota [].
    """
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as file:
            dataset = json.load(file)     
        return [FeatureRecord.from_dict(record) for record in dataset]
    return []

def save_features_dataset(dataset: List[FeatureRecord], filepath: Path) -> None:
    """
    Serializza la lista di FeatureRecord in formato JSON.
    
    PRE: dataset è una lista di oggetti FeatureRecord.
    POST: Sovrascrive il file JSON convertendo ogni elemento con .to_dict() e indent=4.
    """
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump([record.to_dict() for record in dataset] , file, indent=4)

def compute_unique_digits_ratio(solutions: List[str]) -> float:
    """
    Calcola il rapporto di cifre distinte presenti nell'intero insieme di soluzioni.
    
    PRE: solutions è una lista non vuota di combinazioni a 4 cifre.
    POST: Ritorna float tra 0.4 e 1.0 (len(set_cifre) / 10.0).
    """
    set_sol = {c for s in solutions for c in s}
    return len(set_sol) / BASE

def compute_column_entropy(solutions: List[str], dim: int = 4) -> float:
    """
    Calcola l'entropia media delle 4 colonne del bucket.
    
    PRE: solutions lista non vuota. dim = 4.
    POST: Ritorna la media aritmetica dell'entropia di Shannon calcolata colonna per colonna.
    """
    # 1. Inizializza una lista o accumulatore per le entropie delle 4 colonne
    # 2. Per ogni posizione da 0 a dim-1:
    #    a. Estrai la lista di caratteri della colonna: [s[col] for s in solutions]
    #    b. Calcola le frequenze delle cifre presenti
    #    c. Calcola p_d = frequenza / len(solutions)
    #    d. Calcola H_col = -sum(p * log2(p) for p in probabilities if p > 0)
    # 3. Ritorna la media delle entropie (somma / dim)
    acc_entropy = 0
    n_sol = len(solutions)
    for col in range(0,DIM):
        val_col = [s[col] for s in solutions]
        frequencies = calculate__frequencies(val_col)
        probabilities = [frequency/n_sol for frequency in frequencies.values()]
        h_col = -sum(p * math.log2(p) for p in probabilities if p > 0)
        acc_entropy += h_col
    return acc_entropy / DIM

def calculate__frequencies(col: list[str]) -> dict[str, int]:
        """
        Calcola le frequenza delle cifre in una colonna.

        """
        frequencies = {}
        for n in range(0,BASE):
            c = str(n)
            frequencies[c] =col.count(c)
        return frequencies

def build_feature_record(raw_state: RawState) -> FeatureRecord:
    """
    Costruisce l'oggetto FeatureRecord calcolando tutte le metriche intrinseche di un RawState.
    
    PRE: raw_state è un'istanza valida.
    POST: Ritorna FeatureRecord con log_n, unique_digits_ratio e column_entropy calcolati.
    """
    # 1. Calcola log_n = math.log2(raw_state.n)
    # 2. Calcola unique_digits_ratio tramite compute_unique_digits_ratio
    # 3. Calcola column_entropy tramite compute_column_entropy
    # 4. Istanzia e ritorna FeatureRecord con tutti i campi
    return FeatureRecord(
        state_id=raw_state.state_id,
        n=raw_state.n,
        log_n=math.log2(raw_state.n),
        unique_digits_ratio=compute_unique_digits_ratio(raw_state.solutions),
        column_entropy=compute_column_entropy(raw_state.solutions),
        target_cost= raw_state.best_cost,
        source_tree= raw_state.source_tree
    )

def main():
    """
    1. Carica raw_states da RAW_STATES_FILEPATH.
    2. Carica features_cache da FEATURES_DATASET_FILEPATH (se esiste).
    3. Cicla con tqdm su tutti i raw_states:
       - Se lo state_id è già in cache: aggiorna target_cost e source_tree se migliorati.
       - Se è nuovo: costruisci il nuovo record con build_feature_record e aggiungilo.
    4. Salva la lista aggiornata su data/features_dataset.json.
    5. Stampa riepilogo con il totale di record sincronizzati.
    """
    raw_states = load_raw_states(RAW_STATES_FILEPATH)

    features = load_features_dataset(FEATURES_DATASET_FILEPATH)

    cache_features = {record.state_id: record for record in features}

    for i, (state_id, state_data) in enumerate(tqdm(raw_states.items(), desc="Calcolo Feature")):
        if state_id in cache_features.keys():
            if state_data.best_cost < cache_features[state_id].target_cost:
                cache_features[state_id].target_cost = state_data.best_cost
                cache_features[state_id].source_tree = state_data.source_tree
        else:
            new_record = build_feature_record(state_data)
            cache_features[state_id] = new_record

        if (i + 1) % SAVE_INTERVAL == 0:
            save_features_dataset(list(cache_features.values()), FEATURES_DATASET_FILEPATH)


    save_features_dataset(list(cache_features.values()), FEATURES_DATASET_FILEPATH)
    print(f"\nDataset finale salvato con {len(cache_features)} campioni in {FEATURES_DATASET_FILEPATH}!")

if __name__ == "__main__":
    main() 