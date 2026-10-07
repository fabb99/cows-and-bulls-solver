

"""
Modulo: train_oracle (Script)
Pipeline completa di Data Science:
1. Mine di stati unici dagli alberi (N <= 40).
2. Caricamento/Aggiornamento persistente di data/oracle_dataset.json (zero ricalcoli).
3. Analisi della varianza e distribuzione delle feature.
4. Regressione Lineare Multipla per calcolare K1 e K2.
"""
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import math
import numpy as np
from tqdm import tqdm
from src.analysis.tree_io import (load_tree_from_json)
from src.core.engine import (calculate_buckets)
from src.analysis.tree_builder import TreeNode

from src.strategies.deep_evm import (PureDeepEVMStrategy, generate_id)
from src.strategies.entropy import EntropyStrategy

from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
    


DATASET_FILEPATH = Path("data/oracle_dataset.json")

def load_existing_dataset(filepath: Path) -> List[Dict[str, Any]]:
    """
    PRE: filepath punta a un eventuale file JSON.
    POST: Ritorna la lista dei record già calcolati se il file esiste, altrimenti una lista vuota.
    """
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as filepath:
            dataset = json.load(filepath)
        return dataset
    return []

def save_dataset(dataset: List[Dict[str, Any]], filepath: Path) -> None:
    """
    PRE: dataset lista di dizionari con i campi estratti.
    POST: Salva su disco formattato con indent=4.
    """
    with open(filepath, "w", encoding="utf-8") as filepath:
        json.dump(dataset, filepath, indent=4)

def print_dataset_statistics(dataset: List[Dict[str, Any]]) -> None:
    if not dataset:
        return
        
    res_ent = [d["residual_entropy"] for d in dataset]
    n_max = [d["n_max"] for d in dataset]
    cost = [d["exact_cost"] for d in dataset]

    print("\n" + "="*55)
    print(" STATISTICHE DATASET (EDA)")
    print("="*55)
    print(f"Campioni Totali Analizzati: {len(dataset)}")
    print(f"{'Feature':<20} | {'Media':<7} | {'Var':<7} | {'Min':<5} | {'Max':<5}")
    print("-" * 55)
    print(f"{'Entropia Residua':<20} | {np.mean(res_ent):.3f} | {np.var(res_ent):.3f} | {np.min(res_ent):.2f} | {np.max(res_ent):.2f}")
    print(f"{'N_max (Caso Peggiore)':<20} | {np.mean(n_max):.3f} | {np.var(n_max):.3f} | {np.min(n_max):.0f} | {np.max(n_max):.0f}")
    print(f"{'Costo Esatto (M)':<20} | {np.mean(cost):.3f} | {np.var(cost):.3f} | {np.min(cost):.2f} | {np.max(cost):.2f}")


def calculate_k_constants_from_dataset(dataset: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Estrae le matrici X e Y dal dataset e lancia np.linalg.lstsq
    """
    # 1. Prepara i vettori Y e X
    Y = []
    X = []

    for data in dataset:
        Y.append(data["exact_cost"]-1)
        X.append([data["residual_entropy"], data["n_max"]])


    np_Y = np.array(Y)
    np_X = np.array(X)

    coefficent, residual, ran, s = np.linalg.lstsq(np_X,np_Y, rcond=None)

    K1 = coefficent[0]
    K2 = coefficent[1]

    return float(K1), float(K2)
        
def mine_state_from_tree(
    tree: TreeNode, 
    min_bound: int = 3, 
    max_bound: int = 40,
    collected_states: List[List[str]] = None,
    seen_ids: set = None
    ) -> List[List[str]]:
    if collected_states is None:
        collected_states = []
    if seen_ids is None:
        seen_ids = set()

    solutions = tree["possible_solutions"]
    num_sol = len(solutions)

    if min_bound<= num_sol <= max_bound:
        state_id = generate_id(solutions)
        if state_id not in seen_ids:
            seen_ids.add(state_id)
            collected_states.append(solutions)

    for child in tree["children"].values():
        if child is not None:
            mine_state_from_tree(child,min_bound,max_bound,collected_states,seen_ids)

    return collected_states
            
def extract_metrics_for_state(
    possible_solutions: List[str], 
    entropy_strategy: EntropyStrategy, 
    exact_strategy: PureDeepEVMStrategy
) -> Dict[str, Any]:
    num_sol= len(possible_solutions)
    
    entropy_move = entropy_strategy.choose_guess(possible_solutions,["bypass_signaturne"])
    buckets =calculate_buckets(entropy_move,possible_solutions,entropy_strategy.lookup)
    
    n_max = len(max(buckets.values(), key= len))

    probabilities = [len(s)/num_sol for s in buckets.values()]
    h = -sum(p * math.log2(p) for p in probabilities if p != 0)
    residual_entropy = math.log2(num_sol)-h

    exact_cost = exact_strategy._compute_exact_cost(possible_solutions)

    return {
        "id" : generate_id(possible_solutions), 
        "n" : len(possible_solutions), 
        "residual_entropy": residual_entropy,
        "n_max": n_max,
        "exact_cost": exact_cost
        }

def main():
    print("Avvio Laboratorio EVM...")

    # ==========================================
    # 1. Caricamento Dataset e Setup Setaccio
    # ==========================================
    dataset = load_existing_dataset(DATASET_FILEPATH)
    # Estraiamo tutti gli id già salvati in passato per non ricalcolarli
    already_seen_ids = set([data["id"] for data in dataset])
    
    print(f"Dataset caricato: {len(dataset)} record già presenti su disco.")

    # ==========================================
    # 2. Mining degli Alberi
    # ==========================================
    print("Estrazione bivi dall'albero in corso...")
    tree = load_tree_from_json("data/trees/max_entropy.json") 
    
    # Passiamo il setaccio! Così non estrae stati che sono già nel dataset.
    data_to_evaluate = mine_state_from_tree(tree, min_bound=3, max_bound=20, seen_ids=already_seen_ids)
    
    print(f"Trovati {len(data_to_evaluate)} NUOVI stati da analizzare.")


    # ==========================================
    # 3. Inizializzazione Ambiente (Veloce)
    # ==========================================
    print(f"Inizializzazione ambiente ")
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0]) # Lazy init
    tablebase = {data["id"]: data["exact_cost"] for data in dataset}
    
    entropy_strategy = EntropyStrategy(all_perms, lookup)
    exact_strategy = PureDeepEVMStrategy(all_perms, lookup, entropy_strategy, tablebase)


    # ==========================================
    # 4. Calcolo Pesante (con salvataggio incrementale)
    # ==========================================
    if len(data_to_evaluate) > 0:
        # Usiamo tqdm per vedere la barra
        for i, state in enumerate(tqdm(data_to_evaluate, desc="Calcolo Deep EVM")):
            
            # Estraiamo i 4 valori magici
            record = extract_metrics_for_state(state, entropy_strategy, exact_strategy)
            dataset.append(record)
            
            # Salviamo su disco ogni 5 iterazioni per non perdere i dati se si blocca!
            if i % 5 == 0:
                save_dataset(dataset, DATASET_FILEPATH)
                
        # Salvataggio finale
        save_dataset(dataset, DATASET_FILEPATH)

    # ==========================================
    # 5. Stampa e Regressione Finale
    # ==========================================
    if len(dataset) < 2:
        print("Servono almeno 2 campioni nel dataset per fare la regressione.")
        return
        
    print_dataset_statistics(dataset)
    
    k1, k2 = calculate_k_constants_from_dataset(dataset)
    print("\n" + "="*50)
    print("ORACOLO ADDESTRATO CON SUCCESSO")
    print("="*50)
    print(f" Formula: Costo = 1.0 + ({k1:.4f} * Entropia_Residua) + ({k2:.4f} * N_max)")
    print("="*50)

if __name__ == "__main__":
    main()
