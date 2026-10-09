"""
Modulo: estimated_evm
Contiene la strategia EVM Stimata.
Usa la formula di regressione dell'Oracolo (Entropia Residua) per i grandi numeri 
e delega a PureDeepEVM per il finale di partita (Endgame Tablebase).
"""
from math import inf, log2
from typing import List, Dict, Optional
from src.strategies.base import BaseStrategy
from src.core.constants import DIM
from src.core.engine import calculate_buckets
from src.core.signatures import get_representatives
from src.estimators.base import BaseBucketEstimator
class EstimatedEVMStrategy(BaseStrategy):

    def __init__(
        self, 
        all_permutations: List[str], 
        lookup, 
        estimator: BaseBucketEstimator,
        exact_fallback_strategy: Optional[BaseStrategy] = None,
        fallback_threshold: int = 0,


    ):
        super().__init__(all_permutations, lookup)
        self.estimator = estimator
        self.exact_fallback_strategy = exact_fallback_strategy
        self.fallback_threshold = fallback_threshold
   
    def evaluate_candidates(self, candidates_to_test: List[str], possible_solutions: List[str]) -> Dict[str, float]:
        """
        PRE: candidates_to_test e possible_solutions validi.
        POST: Ritorna {candidato: costo_stimato}.
        """
        scores_of_candidates: Dict[str, float] = {}
        num_solutions = len(possible_solutions)
            
        for candidate in candidates_to_test:
            buckets = calculate_buckets(candidate, possible_solutions, self.lookup)
            
            candidate_cost = 1.0
            for fb, sol in buckets.items():
                if fb != (DIM, 0):
                    p = len(sol) / num_solutions
                    candidate_cost += p * self.estimator.estimate_cost(sol)

            # Bonus colpo diretto se la mossa è una soluzione valida
            if candidate in possible_solutions:
                candidate_cost -= (1.0 / num_solutions)

            scores_of_candidates[candidate] = candidate_cost
            
            
        return scores_of_candidates

    def choose_guess(self, possible_solutions: List[str], history: List[str] = None) -> str:
        """
        PRE: possible_solutions lista non vuota.
        POST: Ritorna la mossa a costo minimo stimato (o delega a exact_fallback_strategy).
        """
        num_solutions = len(possible_solutions)
        if num_solutions == 1:
            return possible_solutions[0]
        elif num_solutions == 2:
            return possible_solutions[0]
        elif num_solutions < self.fallback_threshold:
            return self.exact_fallback_strategy.choose_guess(possible_solutions, history)
    
        if history is None:
            history = []
        if len(history) < 1:
            candidates_to_test = get_representatives(self.all_permutations, history, possible_solutions) 
        else:
            candidates_to_test = self.all_permutations
            
        scores = self.evaluate_candidates(candidates_to_test, possible_solutions)
        min_cost = min(scores.values())
        best_candidates = [c for c, cost in scores.items() if cost == min_cost]

        # Spareggio a favore delle soluzioni ancora compatibili
        for candidate in best_candidates:
            if candidate in possible_solutions:
                return candidate

        return best_candidates[0]