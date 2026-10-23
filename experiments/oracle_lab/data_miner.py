"""
Modulo: data_miner (Script)
Fase 1: estrae tutti gli stati (bivi) da tutti gli alberi in TREES_DIR.
Per ogni stato conserva il miglior costo reale trovato tra gli alberi (un upper bound,
quindi is_exact=False). Salva in RAW_STATES_FILEPATH, accumulando con quanto già presente.

Uso: python -m experiments.oracle_lab.data_miner [--min-n 3] [--max-n 5040]
"""
import argparse

from experiments.oracle_lab.config import COST_TOL, RAW_STATES_FILEPATH, TREES_DIR
from experiments.oracle_lab.models import RawState
from experiments.oracle_lab.storage import load_raw_states, save_raw_states
from src.analysis.metrics import average_depth
from src.analysis.tree_builder import TreeNode
from src.analysis.tree_io import load_tree_from_json
from src.core.state import state_id as compute_state_id


def mine_tree(
    node: TreeNode,
    raw_states: dict[str, RawState],
    tree_name: str,
    min_n: int = 3,
    max_n: int = 5040,
) -> None:
    """
    PRE:  node è un nodo di albero valido; raw_states è il dizionario da aggiornare in-place.
    POST: per ogni nodo del sotto-albero con min_n <= N <= max_n, raw_states[state_id]
          contiene il costo più basso visto finora (strettamente migliore oltre COST_TOL;
          a parità resta la sorgente più vecchia). Un costo esatto non viene mai peggiorato.
    """
    solutions = node["possible_solutions"]
    if min_n <= len(solutions) <= max_n:
        cost = average_depth(node)
        sid = compute_state_id(solutions)
        known = raw_states.get(sid)
        if known is None or cost < known.best_cost - COST_TOL:
            raw_states[sid] = RawState(list(solutions), cost, tree_name)

    for child in node["children"].values():
        if child is not None:
            mine_tree(child, raw_states, tree_name, min_n, max_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estrae gli stati dagli alberi decisionali.")
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=5040)
    args = parser.parse_args()

    raw_states = load_raw_states(RAW_STATES_FILEPATH)
    print(f"Dataset caricato: {len(raw_states)} stati già presenti.")

    tree_paths = sorted(TREES_DIR.glob("*.json"))
    if not tree_paths:
        print(f"Nessun albero in {TREES_DIR}. Genera prima gli alberi.")
        return

    for path in tree_paths:
        print(f"- Mining di {path.stem}...")
        mine_tree(load_tree_from_json(str(path)), raw_states, path.stem, args.min_n, args.max_n)

    save_raw_states(raw_states, RAW_STATES_FILEPATH)
    print(f"Totale stati unici: {len(raw_states)} -> {RAW_STATES_FILEPATH}")


if __name__ == "__main__":
    main()