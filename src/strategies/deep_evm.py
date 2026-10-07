"""
Modulo: deep_evm
Contiene la strategia EVM Ricorsiva Pura.
Garantisce la ricerca del minimo matematico assoluto senza approssimazioni.
Ottimizzata esclusivamente tramite Memoization (Cache) e Branch & Bound rigoroso.
"""
from typing import List, Dict, FrozenSet, Optional
from math import inf
from src.strategies.base import BaseStrategy
from src.core.constants import DIM
from src.core.engine import calculate_buckets
from src.core.signatures import get_representatives


class PureDeepEVMStrategy(BaseStrategy):

    def __init__(self, all_permutations: List[str], lookup, warmup_strategy: BaseStrategy, precomputed_tablebase: Optional[Dict[str, float]] = None):
        """
        Inizializza la strategia EVM pura.
        PRE: warmup_strategy è usata SOLO per trovare un primo limite superiore (Upper Bound)
             reale, senza alterare la completezza della ricerca.
        """
        super().__init__(all_permutations, lookup)
        self.warmup_strategy = warmup_strategy
        self.precomputed_tablebase = precomputed_tablebase or {}
        self.cost_cache: Dict[FrozenSet[str], float] = {}

    def _compute_exact_cost(self, solutions: List[str], best_known_bound: float = float('inf')) -> float:
        """
        CUORE RICORSIVO PURO: Esplora l'intero spazio per trovare il costo atteso minimo esatto.
        
        PRE: solutions è la lista delle soluzioni rimaste.
        POST: Ritorna il Valore Atteso (Float) matematicamente perfetto.
        
        REGOLE LOGICHE:
        - Non taglia candidati a priori.
        - Testa la warmup_guess per prima per abbassare il best_known_bound.
        - Applica il Branch & Bound solo quando partial_cost >= best_known_bound.
        """
        num_solutions = len(solutions)  
        if num_solutions == 1:
            return 1
        elif num_solutions == 2:
            return 1.5

        id_state = generate_id(solutions)
        if id_state in self.precomputed_tablebase:
            return self.precomputed_tablebase[id_state]

        key = frozenset(solutions)
        if key in self.cost_cache:
            return self.cost_cache[key]

        warmup_guess = self.warmup_strategy.choose_guess(solutions)
        other_candidates = [c for c in self.all_permutations if c != warmup_guess]
        candidates_to_test = [warmup_guess] + other_candidates
        
        for candidate in candidates_to_test:
            buckets = calculate_buckets(candidate, solutions, self.lookup)

            sorted_buckets = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse = True)

            if len(buckets)==1 and (DIM,0) not in buckets:
                continue

            candidate_score = 1
            unwrapped = False

            for fb, sol in sorted_buckets:

                if (DIM,0) == fb:
                    continue

                if candidate_score >= best_known_bound:
                    unwrapped = True
                    break

                probability = len(sol)/num_solutions
                candidate_score += probability*self._compute_exact_cost(sol)

            if not unwrapped and candidate_score < best_known_bound:
                best_known_bound = candidate_score

        self.cost_cache[key] = best_known_bound

        return best_known_bound
                
    def evaluate_candidates(self, candidates_to_test: List[str], possible_solutions: List[str]) -> Dict[str, float]:
        """
        Valuta i candidati passati calcolando il loro vero valore atteso per la vittoria.
        
        PRE: candidates_to_test sono i candidati da valutare (spesso i rappresentanti delle firme al Turno 1).
        POST: Ritorna {candidato: costo_esatto}.
        """
        scores_of_candidates= {}
        num_solutions = len(possible_solutions)
        
        for candidate in candidates_to_test:
            buckets = calculate_buckets(candidate, possible_solutions, self.lookup)
            candidate_cost = 1.0
            for fb, sol in buckets.items():
                if fb != (DIM, 0):
                    candidate_cost += (len(sol) / num_solutions) * self._compute_exact_cost(sol)
            scores_of_candidates[candidate] = candidate_cost
                    
        return scores_of_candidates

    def choose_guess(self, possible_solutions: List[str], history: List[str] = None) -> str:
            """
            PRE: possible_solutions lista non vuota.
            POST: Ritorna il candidato globale che garantisce il minor numero medio di mosse.
            """
            num_solutions = len(possible_solutions)
            if num_solutions == 1:
                return possible_solutions[0]
            elif num_solutions == 2:
                return possible_solutions[0]

            if history is None:
                history = []
            if len(history) < 1:
                candidates_to_test = get_representatives(self.all_permutations, history, possible_solutions) 
            else:
                candidates_to_test = self.all_permutations
        
            min_cost = inf
            guess = None
            
            for candidate in candidates_to_test:
                buckets = calculate_buckets(candidate, possible_solutions, self.lookup)
                candidate_cost = 1.0

                for fb, sol in buckets.items():
                    if fb != (DIM, 0):
                        candidate_cost += (len(sol) / num_solutions) * self._compute_exact_cost(sol)
                    
                if candidate_cost < min_cost:
                    guess = candidate
                    min_cost = candidate_cost 
                elif candidate_cost == min_cost:
                    if candidate in possible_solutions and (guess is None or guess not in possible_solutions):
                        guess = candidate

            return guess


def generate_id(state:List[str]):
        return ",".join(sorted(state))
