"""
Modulo: investigate_branch_01 (Script - experiments/)
Diagnostica Forense sul Ramo Critico (0, 1) - 1440 Soluzioni al Turno 1:
Mette a confronto diretto le scelte di Turno 2 tra MaxEntropy, Hybrid e le varianti EVM.
Rivela quale mossa è stata scelta, come ha partizionato le 1440 soluzioni, 
cosa ha previsto l'Oracolo e dove si sono originate le partite del Turno 7.
"""
from typing import List, Dict, Any
from pathlib import Path
import math
from collections import Counter

from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.core.engine import calculate_buckets
from src.analysis.tree_io import load_tree_from_json
from src.analysis.metrics import average_depth, get_depth_distribution
from src.cli.game_loop import fb_tuple_to_string
from src.estimators.piecewise import PiecewiseBucketEstimator

TREES_DIR = Path("data/trees")

# Gli alberi da mettere a confronto
STRATEGIES_TO_COMPARE = [
    "max_entropy",
    "hybrid_eg_9406",
    "estimated_evm_ul",
    "estimated_evm_ulg"
]


def analyze_guess_partition(guess: str, solutions: List[str], lookup: FeedbackLookup, estimator: PiecewiseBucketEstimator):
    """Calcola le metriche della partizione e la stima dell'Oracolo."""
    n_sol = len(solutions)
    buckets = calculate_buckets(guess, solutions, lookup)
    
    # 1. Entropia
    probs = [len(s) / n_sol for s in buckets.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    
    # 2. N_max e numero bucket
    n_max = max(len(s) for s in buckets.values())
    total_buckets = len(buckets)
    
    # 3. Cosa prevedeva l'Oracolo?
    predicted_cost = 1.0
    for fb, sol in buckets.items():
        if fb != (4, 0):
            p = len(sol) / n_sol
            predicted_cost += p * estimator.estimate_cost(sol)
            
    is_valid = guess in solutions
    return {
        "guess": guess,
        "is_valid": is_valid,
        "entropy": entropy,
        "n_max": n_max,
        "total_buckets": total_buckets,
        "predicted_cost": predicted_cost,
        "buckets": buckets
    }


def main():
    print("Inizializzazione ambiente e Lookup Table...")
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    estimator = PiecewiseBucketEstimator()
    print("Ambiente pronto!\n")

    # Identifichiamo il ramo (0, 1) dal feedback
    target_fb = (0, 2)
    results = {}

    print("=" * 85)
    print(f" AUTOPSIA FORENSE DEL RAMO CRITICO: (0, 1) [Feedback: -] - 1440 SOLUZIONI")
    print("=" * 85)

    solutions_1440 = None

    for name in STRATEGIES_TO_COMPARE:
        tree_path = TREES_DIR / f"{name}.json"
        if not tree_path.exists():
            continue

        full_tree = load_tree_from_json(str(tree_path))
        child_node = full_tree["children"].get(target_fb)
        
        if child_node is None:
            continue

        if solutions_1440 is None:
            solutions_1440 = child_node["possible_solutions"]

        guess = child_node["guess"]
        real_avg = average_depth(child_node)
        distribution = get_depth_distribution(child_node)
        
        part_info = analyze_guess_partition(guess, solutions_1440, lookup, estimator)
        
        results[name] = {
            "node": child_node,
            "real_avg": real_avg,
            "distribution": distribution,
            "partition": part_info
        }

    # 1. TABELLA DI CONFRONTO DELLE MOSSE AL TURNO 2
    print(f"{'Strategia':<20} | {'Mossa T2':<8} | {'Valida?':<8} | {'Entropia':<9} | {'N_max':<6} | {'Stima EVM':<10} | {'Media REALE':<11}")
    print("-" * 85)
    for name, data in results.items():
        p = data["partition"]
        valid_str = "Sì" if p["is_valid"] else "No"
        print(f"{name:<20} | {p['guess']:<8} | {valid_str:<8} | {p['entropy']:<9.4f} | {p['n_max']:<6} | {p['predicted_cost']:<10.4f} | {data['real_avg']:<11.5f}")
    print("=" * 85)

    # 2. DETTAGLIO DISTRIBUZIONE NEL RAMO
    print("\n" + "=" * 85)
    print(" DOVE VANNO A FINIRE LE 1440 PARTITE DEL RAMO (Turni relativi nel sotto-albero)")
    print("=" * 85)
    print(f"{'Strategia':<20} | {'+1 mossa':<8} | {'+2 mosse':<8} | {'+3 mosse':<8} | {'+4 mosse':<8} | {'+5 mosse':<8} | {'+6 mosse (T7!)':<14}")
    print("-" * 85)
    for name, data in results.items():
        dist = data["distribution"]
        print(f"{name:<20} | {dist.get(1, 0):<8} | {dist.get(2, 0):<8} | {dist.get(3, 0):<8} | {dist.get(4, 0):<8} | {dist.get(5, 0):<8} | {dist.get(6, 0):<14}")
    print("=" * 85)

    # 3. CONFRONTO DIRETTO DEI BUCKET TRA MAX_ENTROPY E ESTIMATED_EVM_PI
    if "max_entropy" in results and "estimated_evm_ul" in results:
        me_guess = results["max_entropy"]["partition"]["guess"]
        evm_guess = results["estimated_evm_ul"]["partition"]["guess"]
        
        print("\n" + "#" * 85)
        print(f" ⚔️ DUELLO BUCKET: MaxEntropy ({me_guess}) VS EstimatedEVM ({evm_guess})")
        print("#" * 85)
        
        me_b = results["max_entropy"]["partition"]["buckets"]
        evm_b = results["estimated_evm_ul"]["partition"]["buckets"]
        
        all_fbs = sorted(set(list(me_b.keys()) + list(evm_b.keys())))
        
        print(f"{'Feedback':<12} | {'Taglia MaxEntropy':<18} | {'Taglia EstimatedEVM':<18} | {'Differenza':<12}")
        print("-" * 65)
        for fb in all_fbs:
            len_me = len(me_b.get(fb, []))
            len_evm = len(evm_b.get(fb, []))
            diff = len_evm - len_me
            diff_str = f"{diff:+d}" if diff != 0 else "="
            print(f"{fb_tuple_to_string(fb):<12} | {len_me:<18} | {len_evm:<18} | {diff_str:<12}")
        print("#" * 65 + "\n")


if __name__ == "__main__":
    main()