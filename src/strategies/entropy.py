"""
Modulo: entropy
Contiene la strategia che massimizza l'Entropia di Shannon.
"""
import math
from src.strategies.base import BaseStrategy
from src.core.signatures import get_representatives
from src.core.engine import calculate_buckets

class EntropyStrategy(BaseStrategy):

    def evaluate_candidates(self, candidates_to_test: list[str], possible_solutions: list[str]) -> dict[str, float]:
        """
        Valuta una lista di candidati calcolando la loro entropia.
        
        PRE: candidates_to_test è la lista dei rappresentanti. possible_solutions sono le soluzioni rimaste.
        POST: Ritorna un dizionario { candidato: entropia_calcolata }.
              NOTA: Per l'Entropia, un punteggio PIÙ ALTO è MIGLIORE.
        """
        scores_of_candidates= {}

        num_soluzioni = len(possible_solutions)
        for candidate in candidates_to_test:
            bucket = [len(x)/num_soluzioni for x in calculate_buckets(candidate, possible_solutions, self.lookup).values()]
            scores_of_candidates[candidate] = sum(x*math.log2(x) for x in bucket if x != 0)*(-1)
            
        return scores_of_candidates

    def choose_guess(self, possible_solutions: list[str], history: list[str] = None) -> str:
        """
        PRE: possible_solutions lista non vuota.
        POST: Ritorna il candidato che massimizza il punteggio (preferendo soluzioni valide in caso di parità).
        """
        if history is None:
            history = []
            
        if len(history)<3:
            candidates_to_test = get_representatives(self.all_permutations, history) 
        else:
            candidates_to_test = self.all_permutations
    
        scores = self.evaluate_candidates(candidates_to_test, possible_solutions)
        best_score = max(scores.values())
        candidates = [candidate for candidate, punteggio in scores.items() if punteggio==best_score]
        for x in candidates:
            if x in possible_solutions:
                return x
            
        return candidates[0]