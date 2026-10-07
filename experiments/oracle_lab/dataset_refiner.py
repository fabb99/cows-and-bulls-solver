"""
Modulo: 2_dataset_refiner (Script)
Fase 1b della Pipeline (Dataset Bootstrapping / Policy Iteration):
Prende i dati estratti dagli alberi (data/raw_states.json) e perfeziona gli stati
dell'Endgame (N <= 20) calcolando il costo esatto con PureDeepEVMStrategy.
Se il Deep EVM trova una sequenza migliore rispetto all'albero originale,
il costo viene aggiornato al ribasso verso il limite teorico ottimo di Bellman.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from tqdm import tqdm

from experiments.oracle_lab.models import RawState
from experiments.oracle_lab.data_miner import (load_raw_states, save_raw_states)

from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.strategies.entropy import EntropyStrategy
from src.strategies.deep_evm import PureDeepEVMStrategy

# Percorsi e configurazioni del laboratorio
RAW_STATES_FILEPATH = Path("data/raw_states.json")
MAX_REFINE_N: int = 20        # Limite massimo di dimensione per il calcolo esatto
SAVE_INTERVAL: int = 25       # Salva su disco ogni N stati per prevenire perdite di dati


def load_json_data(filepath: Path) -> Dict[str, Any]:
    """
    Carica i dati dal file JSON specificato.
    
    PRE: filepath punta a un eventuale file JSON esistente.
    POST: Ritorna il dizionario deserializzato se il file esiste, altrimenti {}.
    """
    return load_raw_states(filepath)


def save_json_data(data: Dict[str, RawState], filepath: Path) -> None:
    """
    Salva il dizionario dei dati su file JSON in modo formattato.
    
    PRE: data è un dizionario valido serializzabile in JSON.
    POST: Sovrascrive il file specificato con indent=4.
    """
    save_raw_states(data, filepath)

def refine_states(
    raw_states: Dict[str, RawState], 
    exact_strategy: PureDeepEVMStrategy, 
    max_n: int = MAX_REFINE_N,
    save_interval: int = SAVE_INTERVAL
) -> int:
    """
    Perfeziona i costi degli stati piccoli applicando l'Equazione di Ottimalità di Bellman.
    
    PRE: 
        - raw_states è il dizionario estratto dallo script 1.
        - exact_strategy è un'istanza funzionante di PureDeepEVMStrategy.
        - max_n è la soglia massima di soluzioni (es. 20).
    POST: 
        - Aggiorna sul posto (in-place) i record di raw_states dove exact_cost < best_cost_m.
        - Ritorna il numero totale di stati che sono stati migliorati.
    """

    states_to_evaluate = [s for s in raw_states.values() if 2 < s.n <= max_n]    

    count_improvement = 0
    
    if len(states_to_evaluate) > 0:
        for i, state in enumerate(tqdm(states_to_evaluate, desc="Calcolo Deep EVM")):

            if state.source_tree != "exact_strategy":

                exact_cost = exact_strategy._compute_exact_cost(state.solutions)

                if exact_cost <= state.best_cost:
                    if exact_cost < state.best_cost:
                        count_improvement += 1
                    state.best_cost = exact_cost
                state.source_tree = "exact_strategy"

            if i % save_interval == 0:
                save_json_data(raw_states,RAW_STATES_FILEPATH)

    save_json_data(raw_states,RAW_STATES_FILEPATH)

    return count_improvement


def main():
    """
    Punto di ingresso principale della Fase 1b:
    1. Verifica l'esistenza di data/raw_states.json e caricalo.
    2. Inizializza l'ambiente di gioco (permutazioni, lookup table).
    3. Costruisci una tablebase preliminare con i costi noti per velocizzare il Deep EVM.
    4. Istanzia EntropyStrategy (per il warmup) e PureDeepEVMStrategy (con la tablebase).
    5. Esegui refine_states tracciando l'avanzamento.
    6. Salva definitivamente i dati aggiornati su data/raw_states.json.
    7. Stampa un riepilogo con gli stati analizzati e i miglioramenti ottenuti.
    """
    raw_states = load_raw_states(RAW_STATES_FILEPATH)

    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0]) # Lazy init
    print(f"Inizializzazione ambiente effettuata")

    entropy_strategy = EntropyStrategy(all_perms,lookup)

    already_perfected = { 
        s.state_id: s.best_cost 
        for s in raw_states.values() 
        if s.source_tree == "exact_strategy"
        }
    exact_strategy = PureDeepEVMStrategy(all_perms, lookup, entropy_strategy, already_perfected)

    num_improvement = refine_states(raw_states,exact_strategy)
    print(f"Migliorati {num_improvement} stati")


if __name__ == "__main__":
    main()