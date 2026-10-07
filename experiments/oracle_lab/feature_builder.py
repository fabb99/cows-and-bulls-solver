"""
Modulo: 2_feature_builder (Script)
Fase 2 della Pipeline:
Legge data/raw_states.json ed estrae le feature matematiche per ciascuno stato:
- N (dimensione)
- Entropia Residua (log2(N) - H)
- N_max (Caso peggiore della mossa migliore)
- target_m (Il costo reale precedentemente minato)
Salva il dataset finale per il training in data/features_dataset.json.
"""
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import math
from tqdm import tqdm

from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.core.engine import calculate_buckets
from src.strategies.entropy import EntropyStrategy

RAW_STATES_FILEPATH = Path("data/raw_states.json")
FEATURES_DATASET_FILEPATH = Path("data/features_dataset.json")


def load_json_file(filepath: Path) -> Any:
    """Carica e ritorna il contenuto di un file JSON (se esiste)."""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json_file(data: Any, filepath: Path) -> None:
    """Salva i dati in un file JSON formattato."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def compute_state_features(
    solutions: List[str], 
    entropy_strategy: EntropyStrategy, 
    lookup: FeedbackLookup
) -> Tuple[float, int]:
    """
    PRE: solutions è una lista non vuota di combinazioni.
    POST: Ritorna (residual_entropy, n_max) calcolati sulla mossa suggerita da EntropyStrategy.
    """
    num_solutions = len(solutions)
    
    # Bypassiamo le firme per trovare la mossa perfetta per questo specifico sotto-stato
    best_guess = entropy_strategy.choose_guess(solutions, history=["bypass"])
    buckets = calculate_buckets(best_guess, solutions, lookup)
    
    # 1. Calcolo Entropia di Shannon (H)
    probabilities = [len(s) / num_solutions for s in buckets.values()]
    h = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    # 2. Entropia Residua = Incertezza totale - informazione estratta
    residual_entropy = math.log2(num_solutions) - h
    
    # 3. Caso Peggiore (Dimensione del bucket più grande)
    n_max = max(len(s) for s in buckets.values())
    
    return residual_entropy, n_max


def main():
    if not RAW_STATES_FILEPATH.exists():
        print(f"Errore: {RAW_STATES_FILEPATH} non trovato. Esegui prima 1_data_miner.py!")
        return

    print("Caricamento dati grezzi...")
    raw_states: Dict[str, Any] = load_json_file(RAW_STATES_FILEPATH)
    print(f"Caricati {len(raw_states)} stati unici da data/raw_states.json.")

    # Checkpoint: controlliamo se esistono già feature calcolate in passato
    existing_dataset: List[Dict[str, Any]] = load_json_file(FEATURES_DATASET_FILEPATH) or []
    seen_ids = {item["id"] for item in existing_dataset}
    if seen_ids:
        print(f"Trovati {len(seen_ids)} record già elaborati in precedenza.")

    # Filtriamo solo gli stati che non abbiamo ancora calcolato
    states_to_process = [
        (state_id, state_data)
        for state_id, state_data in raw_states.items()
        if state_id not in seen_ids
    ]
    print(f"Nuovi stati da elaborare: {len(states_to_process)}.")

    if not states_to_process:
        print("Tutti gli stati sono già stati elaborati! Passa alla Fase 3.")
        return

    # Inizializzazione Ambiente
    print("\nInizializzazione ambiente e Lookup Table...")
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0]) # Lazy initialization forzata
    entropy_strategy = EntropyStrategy(all_perms, lookup)
    print("Ambiente pronto!\n")

    dataset = list(existing_dataset)

    # Ciclo di calcolo con barra di avanzamento
    for i, (state_id, state_data) in enumerate(tqdm(states_to_process, desc="Calcolo Feature")):
        solutions = state_data["solutions"]
        residual_entropy, n_max = compute_state_features(solutions, entropy_strategy, lookup)

        record = {
            "id": state_id,
            "n": state_data["n"],
            "residual_entropy": residual_entropy,
            "n_max": n_max,
            "target_m": state_data["best_cost"],
            "source_tree": state_data["best_strategy"]
        }
        dataset.append(record)

        # Salvataggio incrementale ogni 100 stati (sicurezza da crash/interruzioni)
        if (i + 1) % 100 == 0:
            save_json_file(dataset, FEATURES_DATASET_FILEPATH)

    # Salvataggio finale completo
    save_json_file(dataset, FEATURES_DATASET_FILEPATH)
    print(f"\nOperazione completata con successo!")
    print(f"Dataset salvato in {FEATURES_DATASET_FILEPATH} con {len(dataset)} campioni pronti!")


if __name__ == "__main__":
    main()