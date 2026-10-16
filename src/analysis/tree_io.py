"""
Modulo: tree_io
Serializzazione JSON degli alberi decisionali. Le chiavi tupla (bulls, cows) diventano
stringhe "b,c" nel file e tornano tuple al caricamento.
"""
import json

from src.analysis.tree_builder import TreeNode


def _convert_keys(node: TreeNode | None, to_json: bool) -> TreeNode | None:
    """
    PRE:  node è un nodo (o None).
    POST: copia ricorsiva con le chiavi dei figli convertite
          (tupla -> "b,c" se to_json, "b,c" -> tupla altrimenti).
    """
    if node is None:
        return None
    children = {}
    for key, child in node["children"].items():
        if to_json:
            new_key = f"{key[0]},{key[1]}"
        else:
            bulls, cows = key.split(",")
            new_key = (int(bulls), int(cows))
        children[new_key] = _convert_keys(child, to_json)
    return {
        "guess": node["guess"],
        "possible_solutions": node["possible_solutions"],
        "children": children,
    }


def save_tree_to_json(tree: TreeNode, filepath: str) -> None:
    """PRE: tree valido. POST: file JSON scritto (indent=4)."""
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(_convert_keys(tree, to_json=True), file, indent=4)


def load_tree_from_json(filepath: str) -> TreeNode:
    """PRE: filepath è un JSON prodotto da save_tree_to_json. POST: albero con chiavi tupla."""
    with open(filepath, "r", encoding="utf-8") as file:
        return _convert_keys(json.load(file), to_json=False)