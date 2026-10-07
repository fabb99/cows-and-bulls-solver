"""
Modulo: 1_data_miner (Script)
Fase 1 della Pipeline:
Estrae tutti i bivi (da N=3 a 5040) da tutti gli alberi decisionali in data/trees/.
Per ciascun bivio memorizza il miglior costo reale (average_depth) trovato tra tutti gli alberi.
Salva il risultato grezzo in data/raw_states.json.
"""
from typing import Dict, Any
from pathlib import Path
import json
from tqdm import tqdm

from experiments.oracle_lab.models import RawState
from src.analysis.tree_io import load_tree_from_json
from src.analysis.tree_builder import TreeNode
from src.analysis.metrics import average_depth
from src.strategies.deep_evm import generate_id


OUTPUT_FILEPATH = Path("data/raw_states.json")
TREES_DIR = Path("data/trees")

def load_raw_states(filepath: Path) -> Dict[str, Any]:
    """
    PRE: filepath punta all'eventuale file JSON dei dati grezzi.c
    POST: Ritorna il dizionario degli stati già estratti, oppure un dizionario vuoto.
    """
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as file:
            dataset = json.load(file)
        return dataset
    return {}

def save_raw_states(data: Dict[str, Any], filepath: Path) -> None:
    """
    PRE: data è il dizionario degli stati unici.
    POST: Salva su disco con indent=4.
    """
    with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

def mine_tree_recursive(
    node: TreeNode, 
    raw_states: Dict[str, Any], 
    tree_name: str,
    min_n: int = 3, 
    max_n: int = 5040
) -> None:
    """
    Attraversa ricorsivamente i nodi dell'albero.
    Per ogni nodo con min_n <= len <= max_n:
    - Calcola average_depth(node).
    - Inserisce o aggiorna raw_states[state_id] tenendo il costo MINIMO (best_cost_m).
    """
    if raw_states is None:
        raw_states = {}
  
    solutions = node["possible_solutions"]
    num_sol = len(solutions)

    if min_n <= num_sol <= max_n:
        state_id = generate_id(solutions)
        cost = average_depth(node)
        
        # Se non c'è o se questo albero ha trovato un costo minore viene aggiornato
        if state_id not in raw_states or cost < raw_states[state_id]["best_cost"]:
            raw_states[state_id] = {
                "n": num_sol,
                "solutions": solutions,
                "best_cost": cost,
                "best_strategy": tree_name
            }

    for child in node["children"].values():
        if child is not None:
            mine_tree_recursive(child, raw_states, tree_name, min_n, max_n)

    return raw_states

def main():
    """
    1. Carica raw_states esistente (se c'è).
    2. Trova tutti i file .json in data/trees/.
    3. Per ogni albero, lancia mine_tree_recursive.
    4. Salva il dizionario completo su data/raw_states.json.
    5. Stampa a schermo quanti stati totali unici sono stati minati!
    """
    
    # 1. Carica il dizionario esistente (se esiste)
    raw_states = load_raw_states(OUTPUT_FILEPATH)
    print(f"Dataset caricato: {len(raw_states)} stati già presenti su disco.")

    print("Estrazione stati dagli alberi in corso...")
    
    # 2. Cicla su tutti gli alberi e accumula nello STESSO dizionario raw_states!
    for tree_json in TREES_DIR.glob("*.json"):
        tree_name = tree_json.stem
        print(f"- Mining dell'albero: {tree_name}...")
        tree_data = load_tree_from_json(str(tree_json))
        
        # Passiamo lo stesso dizionario raw_states che continua a riempirsi:
        mine_tree_recursive(tree_data, raw_states, tree_name, min_n=3, max_n=5040)

    # 3. Salvataggio UNICO finale su disco dopo aver scansionato TUTTI gli alberi
    save_raw_states(raw_states, OUTPUT_FILEPATH)
    print(f"\nOperazione completata. Totale stati unici estratti: {len(raw_states)}")
    

if __name__ == "__main__":
    main()