"""
Modulo: hybrid
Contiene le strategie Ensemble/Ibride che combinano diverse euristiche.
"""
from src.core.lookup_table import FeedbackLookup
from src.strategies.base import BaseStrategy
from src.core.signatures import get_representatives

class HybridWeightedStrategy(BaseStrategy):
    """
    Strategia Ibrida Pesata: normalizza i punteggi di due strategie 
    (es. Entropy e Minimax) e li somma usando dei pesi specifici.
    """
    
    def __init__(self, strategy_a: BaseStrategy, strategy_b: BaseStrategy,ishigher_better_a: bool, ishigher_better_b: bool, weight_a: float, weight_b: float, lookup: FeedbackLookup):
        """
        PRE: strategy_a e strategy_b sono istanze già inizializzate. I pesi sono float (es. 0.94 e 0.06).
        POST: Inizializza l'oggetto ibrido.
        """
        self.all_permutations = strategy_a.all_permutations
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.ishigher_better_a = ishigher_better_a
        self.ishigher_better_b = ishigher_better_b
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.lookup = lookup

    def _normalize_scores(self, scores: dict[str, float], is_higher_better: bool) -> dict[str, float]:
        """
        Normalizza i punteggi in un range [0, 1] per poterli sommare.
        
        PRE: scores è il dizionario {candidato: punteggio}. 
        POST: Ritorna il dizionario con i punteggi normalizzati.
              Se is_higher_better è True (come in Entropy), il max prende 1.0.
              Se is_higher_better è False (come in Minimax), il min prende 1.0 (tramite formula: min_score / punteggio).
        """
        if is_higher_better:
            best_score= max(scores.values())
            return {c: s/best_score for c, s in scores.items()}
        else:
            best_score = min(scores.values())
            return {c: best_score/s for c, s in scores.items()}
       
    def evaluate_candidates(self, candidates_to_test: list[str], possible_solutions: list[str]) -> dict[str, float]:
        scores_strategy_a = self._normalize_scores(
            self.strategy_a.evaluate_candidates(candidates_to_test, possible_solutions), self.ishigher_better_a)

        scores_strategy_b = self._normalize_scores(
            self.strategy_b.evaluate_candidates(candidates_to_test, possible_solutions), self.ishigher_better_b)
        
        hybrid_scores = {candidate: scores_strategy_a[candidate]*self.weight_a +
                         scores_strategy_b[candidate]*self.weight_b for candidate in candidates_to_test}

        return hybrid_scores

    def choose_guess(self, possible_solutions: list[str], history: list[str] = None) -> str:
        """
        PRE: possible_solutions lista non vuota.
        POST: Valuta le due strategie, normalizza i risultati, applica i pesi e ritorna il migliore.
        """
        if history is None:
                history = []
                
        if len(history)<1:
            candidates_to_test = get_representatives(self.all_permutations, history, possible_solutions) 
        else:
            candidates_to_test = self.all_permutations

        scores = self.evaluate_candidates(candidates_to_test, possible_solutions)
        best_score = max(scores.values())  
        candidates = [candidate for candidate, punteggio in scores.items() if punteggio==best_score]
        
        for x in candidates:
            if x in possible_solutions:
                return x
                    
        return candidates[0]



    