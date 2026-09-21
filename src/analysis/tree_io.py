"""
Modulo: tree_io
Gestisce il salvataggio (serializzazione) e il caricamento degli alberi decisionali in JSON.
Le tuple usate come chiavi per i feedback (es. (2, 2)) vengono convertite in stringhe per compatibilità JSON.
"""
import json
from typing import Dict, Any, Optional

TreeNode = Dict[str, Any]

def _tuple_keys_to_strings(node: Optional[TreeNode]) -> Optional[TreeNode]:
    """ Ricorsivamente trasforma le chiavi tupla (2, 2) in stringhe '2,2' per il JSON. """
    if node is None:
        return None
        
    safe_node = {
        "guess": node["guess"],
        "possible_solutions": node["possible_solutions"],
        "children": {}
    }
    
    for feedback_tuple, child_node in node["children"].items():
        str_key = f"{feedback_tuple[0]},{feedback_tuple[1]}"
        safe_node["children"][str_key] = _tuple_keys_to_strings(child_node)
        
    return safe_node

def _string_keys_to_tuples(node: Optional[TreeNode]) -> Optional[TreeNode]:
    """ Ricorsivamente trasforma le chiavi stringa '2,2' del JSON di nuovo in tuple (2, 2). """
    if node is None:
        return None
        
    restored_node = {
        "guess": node["guess"],
        "possible_solutions": node["possible_solutions"],
        "children": {}
    }
    
    for feedback_str, child_node in node["children"].items():
        bulls, cows = feedback_str.split(',')
        tuple_key = (int(bulls), int(cows))
        restored_node["children"][tuple_key] = _string_keys_to_tuples(child_node)
        
    return restored_node

def save_tree_to_json(tree: TreeNode, filepath: str) -> None:
    """
    PRE: tree è un albero decisionale valido. filepath è il percorso del file.
    POST: Salva l'albero convertendo le chiavi tupla in stringhe.
    """
    safe_tree = _tuple_keys_to_strings(tree)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(safe_tree, file, indent=4)

def load_tree_from_json(filepath: str) -> TreeNode:
    """
    PRE: filepath punta a un JSON salvato in precedenza.
    POST: Ritorna l'albero ricostruito in RAM, con i feedback convertiti in tuple vere.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        raw_tree = json.load(file)
    return _string_keys_to_tuples(raw_tree)

  