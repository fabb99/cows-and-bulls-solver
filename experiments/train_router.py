"""
Modulo: train_router (Script - experiments/oracle_lab/)
Il Torneo Frankenstein Doppio (Routing Dinamico Multi-Obiettivo):
1. Campione d'Attacco (Best Average): minimizza sempre la media.
2. Campione Difensivo (Pareto): minimizza prima il caso peggiore, poi la media.

In ogni bivio, decreta due vincitori e scende in entrambi i sotto-alberi vincenti 
(senza ricalcolarli se coincidono).
Salva la politica finale doppia in data/models/routing_policy.json.
"""
from typing import List, Dict, Tuple
from pathlib import Path
import json
import math
from tqdm import tqdm

from src.core.constants import DIM
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.analysis.tree_builder import build_decision_tree
from src.analysis.metrics import max_depth, calculate_total_moves_and_leaves
from src.strategies.base import BaseStrategy

from src.strategies.entropy import EntropyStrategy
from src.strategies.greedy_minimax import GreedyMinimaxStrategy
from src.strategies.estimated_evm import EstimatedEVMStrategy
from src.strategies.hybrid import HybridWeightedStrategy
from src.estimators.geometric import GeometricBucketEstimator
from src.estimators.piecewise import PiecewiseBucketEstimator

ROUTING_POLICY_FILEPATH = Path("data/models/routing_policy.json")

# Barra globale
pbar = tqdm(desc="Bivi Esplorati", unit=" nodi")


def simulate_tournament(
    solutions: List[str], 
    history_feedback: List[Tuple[int, int]], 
    competitors: Dict[str, BaseStrategy],
    policy_avg: Dict[str, str],
    policy_def: Dict[str, str],
    current_winning_strategy_name: str
) -> None:
    """Torneo ricorsivo a due teste (Attacco e Difesa)."""
    num_solutions = len(solutions)
    if num_solutions <= 2:
        return
        
    if not history_feedback:
        history_key = "START"
    else:
        history_key = " -> ".join([str(fb) for fb in history_feedback])
        
    pbar.set_postfix_str(f"Stato: {history_key[-20:]} | Sol: {num_solutions}")

    # Attacco (Minime mosse totali)
    best_avg_moves = math.inf
    avg_tree_max_depth = math.inf
    winner_avg = current_winning_strategy_name
    tree_avg = None

    # Difesa (Minimo caso peggiore, poi minime mosse totali)
    best_max_depth = math.inf
    def_tree_moves = math.inf
    winner_def = current_winning_strategy_name
    tree_def = None

    # IL TORNEO
    for name, strategy in competitors.items():
        tree = build_decision_tree(strategy, solutions, history=["bypass"])
        
        # Estraiamo gli INTERI PURI
        moves, _ = calculate_total_moves_and_leaves(tree)  # Numero intero di mosse
        m_depth = max_depth(tree)                          # Numero intero di livelli
        
        # ----------------------------------------------------
        # 1. AGGIORNAMENTO ATTACCO (Pura Efficienza Numerica)
        # ----------------------------------------------------
        if moves < best_avg_moves:
            best_avg_moves = moves
            avg_tree_max_depth = m_depth
            winner_avg = name
            tree_avg = tree
        elif moves == best_avg_moves:  # <--- UGUAGLIANZA ESATTA TRA INTERI!
            if m_depth < avg_tree_max_depth:
                avg_tree_max_depth = m_depth
                winner_avg = name
                tree_avg = tree

        # ----------------------------------------------------
        # 2. AGGIORNAMENTO DIFESA (Massima Sicurezza Pareto)
        # ----------------------------------------------------
        if m_depth < best_max_depth:
            best_max_depth = m_depth
            def_tree_moves = moves
            winner_def = name
            tree_def = tree
        elif m_depth == best_max_depth:  # <--- UGUAGLIANZA ESATTA TRA INTERI!
            if moves < def_tree_moves:
                def_tree_moves = moves
                winner_def = name
                tree_def = tree

    # REGISTRAZIONE POLICY
    policy_avg[history_key] = winner_avg
    policy_def[history_key] = winner_def
    pbar.update(1)
    
    # Controllo: Le due politiche hanno scelto la stessa mossa (guess)?
    same_guess_chosen = (tree_avg["guess"] == tree_def["guess"])
    
    # DISCESA BIANCONIGLIO ATTACCO
    if tree_avg is not None and "children" in tree_avg:
        for feedback, child_node in tree_avg["children"].items():
            if child_node is not None and feedback != (DIM, 0):
                nuovo_storico = history_feedback + [feedback]
                chiave_figlio = " -> ".join([str(fb) for fb in nuovo_storico])
                
                # Scendiamo solo se non ci siamo già passati
                if chiave_figlio not in policy_avg:
                    simulate_tournament(child_node["possible_solutions"], nuovo_storico, competitors, policy_avg, policy_def, winner_avg)

    # DISCESA BIANCONIGLIO DIFESA
    # Scendiamo nell'albero difensivo SOLO se la mossa scelta è DIVERSA da quella d'attacco.
    # Altrimenti, il ramo d'attacco ha già esplorato i figli e registrato le decisioni ottimali anche per la difesa!
    if not same_guess_chosen and tree_def is not None and "children" in tree_def:
        for feedback, child_node in tree_def["children"].items():
            if child_node is not None and feedback != (DIM, 0):
                nuovo_storico = history_feedback + [feedback]
                chiave_figlio = " -> ".join([str(fb) for fb in nuovo_storico])
                
                # Scendiamo solo se non ci siamo già passati
                if chiave_figlio not in policy_def:
                    simulate_tournament(child_node["possible_solutions"], nuovo_storico, competitors, policy_avg, policy_def, winner_def)


def main():
    print("Inizializzazione ambiente e Lookup Table...")
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])

    # Istanziazione delle strategie in gara
    max_entropy = EntropyStrategy(all_perms, lookup)
    greedy_minmax = GreedyMinimaxStrategy(all_perms, lookup)
    
    estimated_evm_piecewise = EstimatedEVMStrategy(
        all_perms, lookup, PiecewiseBucketEstimator(), max_entropy, fallback_threshold=0
    )
    
    estimated_evm_geometric = EstimatedEVMStrategy(
        all_perms, lookup, GeometricBucketEstimator(), max_entropy, fallback_threshold=0
    )
    
    hybrid_evm_minmax = HybridWeightedStrategy(
        estimated_evm_piecewise, greedy_minmax, ishigher_better_a=False, ishigher_better_b=False, weight_a=0.944, weight_b=0.056
    )
    
    hybrid_entropy_minmax = HybridWeightedStrategy(
        max_entropy, greedy_minmax, ishigher_better_a=True, ishigher_better_b=False, weight_a=0.94, weight_b=0.06
    )

    competitors = {
        "max_entropy": max_entropy,
        "estimated_evm_piecewise": estimated_evm_piecewise,
        "estimated_evm_geometric": estimated_evm_geometric,
        "hybrid_emv_minmax_9405": hybrid_evm_minmax,
        "hybrid_maxentropy_minmax_9406": hybrid_entropy_minmax
    }

    print(f"Torneo avviato con {len(competitors)} strategie in competizione...\n")
    
    policy_avg: Dict[str, str] = {}
    policy_def: Dict[str, str] = {}
    
    # AVVIO DEL TORNEO DALLA RADICE
    simulate_tournament(all_perms, [], competitors, policy_avg, policy_def, "max_entropy")
    
    pbar.close()

    # SALVATAGGIO
    output_data = {
        "best_average": policy_avg,
        "pareto_defensive": policy_def
    }
    
    ROUTING_POLICY_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ROUTING_POLICY_FILEPATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nTorneo concluso! Policy salvata in {ROUTING_POLICY_FILEPATH}.")
    print(f"- Nodi Mappati (Attacco) : {len(policy_avg)}")
    print(f"- Nodi Mappati (Difesa)  : {len(policy_def)}")


if __name__ == "__main__":
    main()