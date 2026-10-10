"""
Modulo: generate_trees (Script)
Genera e salva su disco gli alberi decisionali per un set di strategie configurate.
"""
import time
from tqdm import tqdm
from src.core.lookup_table import FeedbackLookup
from src.core.generator import generate_all_permutations
from src.analysis.tree_builder import build_decision_tree
from src.analysis.tree_io import save_tree_to_json

from src.estimators.geometric import GeometricBucketEstimator
from src.estimators.piecewise import PiecewiseBucketEstimator

from src.strategies.base import BaseStrategy
from src.strategies.basic import NaiveStrategy,MaxFrequencyStrategy
from src.strategies.greedy_minimax import GreedyMinimaxStrategy
from src.strategies.entropy import EntropyStrategy
from src.strategies.hybrid import HybridWeightedStrategy
from src.strategies.deep_evm import PureDeepEVMStrategy
from src.strategies.estimated_evm import EstimatedEVMStrategy

from experiments.train_oracle import (load_existing_dataset)

def setup_strategies(all_perms, lookup) -> dict[str, BaseStrategy]:
    """
    POST: Inizializza e restituisce un dizionario con tutte le strategie che vogliamo tracciare.
          Es: {"naive": NaiveStrategy(...), "entropy": EntropyStrategy(...), ...}
    """
    strategies = {
        # "naive": NaiveStrategy(all_perms, lookup),
        # "max_frequency": MaxFrequencyStrategy(all_perms, lookup),
        # "greedy_minimax": GreedyMinimaxStrategy(all_perms,lookup),
        # "max_entropy": EntropyStrategy(all_perms,lookup), 
        # "hybrid_eg_9406": HybridWeightedStrategy(EntropyStrategy(all_perms,lookup),GreedyMinimaxStrategy(all_perms,lookup),True,False,94,6,lookup),
        # "pure_deep_emv": PureDeepEVMStrategy(all_perms,lookup, EntropyStrategy(all_perms,lookup)),
        #"estimated_evm_t2": EstimatedEVMStrategy(all_perms,lookup,GeometricBucketEstimator(),PureDeepEVMStrategy(all_perms,lookup, EntropyStrategy(all_perms,lookup)),fallback_threshold=0),
        "estimated_evm_pi": EstimatedEVMStrategy(all_perms,lookup,PiecewiseBucketEstimator(),PureDeepEVMStrategy(all_perms,lookup, EntropyStrategy(all_perms,lookup)),fallback_threshold=0),
    }
    return strategies

def main():
    """
    Logica principale:
    1. Genera all_perms e lookup table (come nel main.py).
    2. Ottiene il dizionario delle strategie da setup_strategies.
    3. Per ogni (nome, strategia):
       - build_decision_tree(...)
       - save_tree_to_json(..., filepath="data/trees/{nome}.json")
       - Stampa "Albero {nome} generato con successo!"
    """
    print("Generazione permutazioni e Lookup Table in corso (Lazy)...")
    start_time = time.time()
    
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    
    lookup.get_feedback(all_perms[0], all_perms[0]) 
    
    print(f"Sistema pronto! ({time.time() - start_time:.2f} sec)\n")

    for name, strategy in setup_strategies(all_perms,lookup).items():
        print(f"\nCalcolo dell'albero per: {name}")
        with tqdm(total=len(all_perms), desc=f"Generando {name}", unit="partite") as progress_bar:
            # Passiamo il metodo update della barra come callback!
            tree = build_decision_tree(
                strategy, 
                all_perms, 
                history=[], 
                on_victory_callback=progress_bar.update
            )
          
        save_tree_to_json(tree, f"data/trees/{name}.json")

        print(f"Albero {name} salvato con successo!")

if __name__ == "__main__": 
    main()
    