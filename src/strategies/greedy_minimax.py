"""
Modulo: greedy_minimax
Contiene la strategia che minimizza la dimensione del bucket al turno successivo (Profondità 1).
"""
from src.strategies.base import BaseStrategy
from src.core.signatures import get_representatives
from src.core.engine import calculate_buckets

class GreedyMinimaxStrategy(BaseStrategy):
    
    def evaluate_candidates(self, candidates_to_test: list[str], possible_solutions: list[str]) -> dict[str, int]:
        """
        Valuta una lista di candidati calcolando il loro caso peggiore.
        
        PRE: candidates_to_test è la lista dei rappresentanti. possible_solutions sono le soluzioni rimaste.
        POST: Ritorna un dizionario { candidato: dimensione_massima_del_bucket }.
              NOTA: Per il Minimax, un punteggio PIÙ BASSO è MIGLIORE.
        """
        scores_of_candidates= {}

        for candidate in candidates_to_test:
            bucket =  calculate_buckets(candidate, possible_solutions, self.lookup)
            scores_of_candidates[candidate] = max(len(x) for x in bucket.values())
            
        return scores_of_candidates

    def choose_guess(self, possible_solutions: list[str], history: list[str] = None) -> str:
        """
        PRE: possible_solutions lista non vuota.
        POST: Ritorna il candidato che minimizza il punteggio (preferendo soluzioni valide in caso di parità).
        """
        if history is None:
            history = []

        if len(history):    
            candidates_to_test = get_representatives(self.all_permutations, history)
        else:
            candidates_to_test = self.all_permutations
        
        scores = self.evaluate_candidates(candidates_to_test, possible_solutions)
        
        best_score = min(scores.values())

        candidates = [candidate for candidate, punteggio in scores.items() if punteggio==best_score]

        for x in candidates:
            if x in possible_solutions:
                return x
            
        return candidates[0]