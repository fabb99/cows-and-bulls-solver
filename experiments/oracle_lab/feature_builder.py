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

from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.core.engine import calculate_buckets
from src.strategies.entropy import EntropyStrategy

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



def extract_features_for_state(
    raw_state: RawState, 
    entropy_strategy: EntropyStrategy, 
    lookup: FeedbackLookup
) -> FeatureRecord:
    """
    Calcola le metriche della Teoria dell'Informazione per un singolo RawState.
    
    PRE: 
        - raw_state è un'istanza valida di RawState.
        - entropy_strategy e lookup sono istanze funzionanti del motore di gioco.
    POST: Ritorna un'istanza FeatureRecord contenente:
          - state_id = raw_state.state_id
          - n = raw_state.n
          - residual_entropy = log2(N) - H
          - n_max = dimensione del bucket peggiore
          - target_cost = raw_state.best_cost
          - source_tree = raw_state.source_tree
    """
    # 1. Trova la mossa migliore: entropy_strategy.choose_guess(raw_state.solutions, history=["bypass"])
    entropy_move = entropy_strategy.choose_guess(raw_state.solutions, history=["bypass"])

    # 2. Calcola la frammentazione: calculate_buckets(best_guess, raw_state.solutions, lookup)
    buckets = calculate_buckets(entropy_move, raw_state.solutions, lookup)

    # 3. Calcola le probabilità di ciascun bucket: p_i = len(bucket) / raw_state.n
    probabilities = [len(bucket) / raw_state.n for bucket in buckets.values()]

    # 4. Calcola l'Entropia di Shannon: H = -sum(p * log2(p) for p in probabilities if p > 0)
    h = -sum(p * math.log2(p) for p in probabilities if p > 0) 
    
    # 5. Calcola l'Incertezza Residua: residual_entropy = log2(raw_state.n) - H
    residual_entropy = math.log2(raw_state.n) - h

    # 6. Calcola il Caso Peggiore: n_max = max(len(s) for s in buckets.values())
    n_max = max(len(s) for s in buckets.values())

    # 7. Istanzia e ritorna FeatureRecord con tutti i campi calcolati
    record = FeatureRecord(
        raw_state.state_id,
        raw_state.n,
        residual_entropy,
        n_max,
        raw_state.best_cost,
        raw_state.source_tree
    )

    return record


def main():
    """
    Punto di ingresso principale della Fase 2:
    1. Verifica l'esistenza di data/raw_states.json e caricalo usando load_raw_states (dizionario di RawState).
    2. Carica i FeatureRecord già esistenti da data/features_dataset.json (se interrotto in passato).
    3. Crea un set di id già calcolati per saltare all'istante i record già pronti.
    4. Inizializza l'ambiente di gioco (permutazioni, lookup table forzata, EntropyStrategy).
    5. Avvia il ciclo tqdm solo sugli stati non ancora calcolati:
       - Chiama extract_features_for_state per ciascuno.
       - Aggiunge il FeatureRecord alla lista dataset.
       - Salva su disco ogni SAVE_INTERVAL iterazioni usando save_features_dataset.
    6. Salva definitivamente il dataset completo su data/features_dataset.json.
    7. Stampa un riepilogo con il numero totale di campioni arricchiti pronti per il training.
    """
    raw_states = load_raw_states(RAW_STATES_FILEPATH)

    features = load_features_dataset(FEATURES_DATASET_FILEPATH)

    cache_features = {record.state_id: record for record in features}

    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0]) # Lazy init
    entropy_strategy = EntropyStrategy(all_perms,lookup)
    print(f"Inizializzazione ambiente effettuata")

    for i, (state_id, state_data) in enumerate(tqdm(raw_states.items(), desc="Calcolo Feature")):
        if state_id in cache_features.keys():
            if state_data.best_cost < cache_features[state_id].target_cost:
                cache_features[state_id].target_cost = state_data.best_cost
                cache_features[state_id].source_tree = state_data.source_tree
        else:
            new_record = extract_features_for_state(state_data, entropy_strategy, lookup)
            cache_features[state_id] = new_record

        if (i + 1) % SAVE_INTERVAL == 0:
            save_features_dataset(list(cache_features.values()), FEATURES_DATASET_FILEPATH)


    save_features_dataset(list(cache_features.values()), FEATURES_DATASET_FILEPATH)
    print(f"\nDataset finale salvato con {len(cache_features)} campioni in {FEATURES_DATASET_FILEPATH}!")

if __name__ == "__main__":
    main() 