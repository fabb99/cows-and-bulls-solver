"""
Modulo: tune_hybrid_weights (Script - experiments/)
Ottimizzatore Adattivo Generico per Strategie Ibride (HybridWeightedStrategy):
Trova simultaneamente due configurazioni ottimali tra due strategie arbitrarie:
1. Il Campione della Media Assoluta (minima media mosse complessiva).
2. Il Campione Difensivo di Pareto (minimo caso peggiore assoluto con la media più bassa).
Salva i risultati vincenti in data/models/hybrid_weights.json.
"""
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.strategies.base import BaseStrategy
from src.strategies.entropy import EntropyStrategy
from src.strategies.greedy_minimax import GreedyMinimaxStrategy
from src.strategies.hybrid import HybridWeightedStrategy
from src.strategies.estimated_evm import EstimatedEVMStrategy
from src.estimators.piecewise import PiecewiseBucketEstimator
from src.analysis.tree_builder import build_decision_tree
from src.analysis.metrics import max_depth, average_depth

OUTPUT_WEIGHTS_FILEPATH = Path("data/models/hybrid_weights.json")


def evaluate_single_weight(
    strategy_a: BaseStrategy,
    strategy_b: BaseStrategy,
    weight_a: float,
    all_perms: List[str],
    is_higher_better_a: bool = True,
    is_higher_better_b: bool = False,
) -> Tuple[int, float]:
    """
    Costruisce l'albero decisionale completo e misura caso peggiore e media.
    """
    w_a = round(weight_a, 4)
    w_b = round(1.0 - w_a, 4)

    hybrid = HybridWeightedStrategy(
        strategy_a, strategy_b,
        is_higher_better_a, is_higher_better_b,
        w_a, w_b,
    )
    tree = build_decision_tree(hybrid, all_perms)
    worst_case = max_depth(tree)
    avg_moves = average_depth(tree)
    return worst_case, avg_moves


def extract_champions(tested_weights: Dict[float, Tuple[int, float]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Estrae dal registro dei test sia il Campione della Media che il Campione Difensivo di Pareto.
    """
    # 1. Campione della Media Assoluta (minima media mosse)
    best_avg_w = min(tested_weights.keys(), key=lambda w: tested_weights[w][1])
    champion_avg = {
        "weight_a": best_avg_w,
        "weight_b": round(1.0 - best_avg_w, 4),
        "worst_case": tested_weights[best_avg_w][0],
        "avg_moves": round(tested_weights[best_avg_w][1], 5)
    }

    # 2. Campione Difensivo di Pareto (minimo caso peggiore assoluto, con spareggio sulla media)
    min_worst = min(res[0] for res in tested_weights.values())
    candidates_min_worst = [w for w, res in tested_weights.items() if res[0] == min_worst]
    best_defensive_w = min(candidates_min_worst, key=lambda w: tested_weights[w][1])
    champion_defensive = {
        "weight_a": best_defensive_w,
        "weight_b": round(1.0 - best_defensive_w, 4),
        "worst_case": tested_weights[best_defensive_w][0],
        "avg_moves": round(tested_weights[best_defensive_w][1], 5)
    }

    return champion_avg, champion_defensive

def run_adaptive_search(
    strategy_a: BaseStrategy,
    strategy_b: BaseStrategy,
    all_perms: List[str],
    is_higher_better_a: bool = True,
    is_higher_better_b: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Esegue la ricerca adattiva su 3 livelli di zoom (Coarse -> Zoom -> Micro Focus)
    tenendo traccia contemporaneamente di entrambi gli obiettivi.
    """
    tested_weights: Dict[float, Tuple[int, float]] = {}

    def test_weight(w: float, pbar=None):
        w_rounded = round(float(w), 4)
        if w_rounded in tested_weights:
            return
        worst, avg = evaluate_single_weight(
            strategy_a, strategy_b, w_rounded, all_perms,
            is_higher_better_a, is_higher_better_b
        )
        tested_weights[w_rounded] = (worst, avg)
        if pbar:
            pbar.set_postfix_str(f"w_a={w_rounded:.3f} | worst={worst} | avg={avg:.4f}")

    # --- FASE 1: Coarse Grid (Passo 0.05 da 0.50 a 1.00) ---
    coarse_grid = [round(float(x), 2) for x in np.arange(0.50, 1.01, 0.05)]
    with tqdm(coarse_grid, desc="Fase 1: Coarse Grid (0.05)") as pbar:
        for w in pbar:
            test_weight(w, pbar)

    # Identifica i due candidati attuali per zoomare attorno a entrambi
    cand_avg, cand_def = extract_champions(tested_weights)

    # --- FASE 2: Zoom Grid (Passo 0.01 attorno ai candidati) ---
    zoom_points = set()
    for w_target in [cand_avg["weight_a"], cand_def["weight_a"]]:
        low = max(0.50, w_target - 0.04)
        high = min(1.00, w_target + 0.04)
        for w in np.arange(low, high + 0.001, 0.01):
            zoom_points.add(round(float(w), 3))

    with tqdm(sorted(zoom_points), desc="Fase 2: Zoom Grid (0.01)") as pbar:
        for w in pbar:
            test_weight(w, pbar)

    # Ricalcola i candidati per il micro-focus
    cand_avg, cand_def = extract_champions(tested_weights)

    # --- FASE 3: Micro Focus (Passo 0.002 attorno ai vincitori) ---
    micro_points = set()
    for w_target in [cand_avg["weight_a"], cand_def["weight_a"]]:
        low = max(0.50, w_target - 0.008)
        high = min(1.00, w_target + 0.008)
        for w in np.arange(low, high + 0.0001, 0.002):
            micro_points.add(round(float(w), 4))

    with tqdm(sorted(micro_points), desc="Fase 3: Micro Focus (0.002)") as pbar:
        for w in pbar:
            test_weight(w, pbar)

    # Verdetto Finale definitivo
    return extract_champions(tested_weights)

def main():
    print("Inizializzazione ambiente e Lookup Table...")
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    print("Ambiente pronto!\n")

    # 1. Istanziazione Generica delle due strategie da ottimizzare
    strategy_c = EntropyStrategy(all_perms, lookup)
    strategy_a = EstimatedEVMStrategy(all_perms,lookup,PiecewiseBucketEstimator(),strategy_c,fallback_threshold=0)
    
    strategy_b = GreedyMinimaxStrategy(all_perms, lookup)
    #strategy_b = EntropyStrategy(all_perms, lookup)

    name_a = strategy_a.__class__.__name__
    name_b = strategy_b.__class__.__name__

    print("=" * 75)
    print(f" OTTIMIZZATORE IPERPARAMETRI ADATTIVO")
    print(f" Strategia A: {name_a}")
    print(f" Strategia B: {name_b}")
    print("=" * 75 + "\n")

    # 2. Esecuzione della Ricerca Adattiva
    champ_avg, champ_defensive = run_adaptive_search(
        strategy_a, strategy_b, all_perms,
        is_higher_better_a=False, is_higher_better_b=False
    )

    # 3. Stampa dei Risultati a Schermo
    print("\n" + "=" * 75)
    print(" RISULTATI OTTIMIZZAZIONE IBRIDA")
    print("=" * 75)
    print(f"1. CAMPIONE MEDIO:")
    print(f"   Peso {name_a} : {champ_avg['weight_a'] * 100:.2f}%")
    print(f"   Peso {name_b} : {champ_avg['weight_b'] * 100:.2f}%")
    print(f"   Media Mosse     : {champ_avg['avg_moves']:.5f}")
    print(f"   Caso Peggiore   : {champ_avg['worst_case']}")
    print("-" * 75)
    print(f"2. CAMPIONE DIFENSIVO (Miglior Caso Peggiore):")
    print(f"   Peso {name_a} : {champ_defensive['weight_a'] * 100:.2f}%")
    print(f"   Peso {name_b} : {champ_defensive['weight_b'] * 100:.2f}%")
    print(f"   Media Mosse     : {champ_defensive['avg_moves']:.5f}")
    print(f"   Caso Peggiore   : {champ_defensive['worst_case']}")
    print("=" * 75 + "\n")

    # 4. Salvataggio dell'Artefatto su Disco
    artifact = {
        "strategy_a": name_a,
        "strategy_b": name_b,
        "champion_best_average": champ_avg,
        "champion_pareto_defensive": champ_defensive
    }
    
    OUTPUT_WEIGHTS_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_WEIGHTS_FILEPATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=4)
        
    print(f"Configurazioni ottimali salvate con successo in {OUTPUT_WEIGHTS_FILEPATH}!")


if __name__ == "__main__":
    main()