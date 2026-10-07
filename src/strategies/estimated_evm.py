"""
Modulo: estimated_evm
Contiene la strategia EVM Stimata.
Usa la formula di regressione dell'Oracolo (Entropia Residua) per i grandi numeri 
e delega a PureDeepEVM per il finale di partita (Endgame Tablebase).
"""
from math import inf, log2
from typing import List, Dict
from src.strategies.base import BaseStrategy
from src.core.constants import DIM
from src.core.engine import calculate_buckets
from src.core.signatures import get_representatives

class EstimatedEVMStrategy(BaseStrategy):


    K0_BASE: float = 1.8769
    K1_FACTOR: float = 0.4659



    def __init__(
        self, 
        all_permutations: List[str], 
        lookup, 
        exact_fallback_strategy: BaseStrategy,
        fallback_threshold: int = 0,

        k0_base: float = K0_BASE,
        k1_factor: float = K1_FACTOR
    ):
        super().__init__(all_permutations, lookup)
        self.exact_fallback_strategy = exact_fallback_strategy
        self.fallback_threshold = fallback_threshold
        self.k0_base = k0_base
        self.k1_factor = k1_factor


    def _estimate_cost(self, entropy: float, num_solutions: int, candidate_is_valid: bool) -> float:
        """
        Calcola l'aspettativa dell'intera mossa tramite l'entropia residua e la penalità del caso peggiore.
        """
        if num_solutions == 1:
            return 1.0
        elif num_solutions == 2:
            return 1.5
        residual_entropy = log2(num_solutions) - entropy
        
        cost = self.k0_base + (self.k1_factor * residual_entropy)
        if candidate_is_valid: 
            return cost - (1.0 / num_solutions)
        return cost

    def evaluate_candidates(self, candidates_to_test: List[str], possible_solutions: List[str]) -> Dict[str, float]:
        """
        PRE: candidates_to_test e possible_solutions validi.
        POST: Ritorna {candidato: costo_stimato}.
        """
        scores_of_candidates: Dict[str, float] = {}
        num_solutions = len(possible_solutions)
            
        for candidate in candidates_to_test:
            buckets = calculate_buckets(candidate, possible_solutions, self.lookup)
            
            probabilities = [len(s) / num_solutions for s in buckets.values()]
            entropy = -sum(p * log2(p) for p in probabilities if p > 0)
            
            
            is_valid = candidate in possible_solutions
            scores_of_candidates[candidate] = self._estimate_cost(entropy, num_solutions, is_valid)
                        
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