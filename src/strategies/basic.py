"""
Modulo: basic
Contiene le strategie di base (Naive, Max Frequenza).
"""
from src.strategies.base import BaseStrategy
from src.core.costants import DIM, BASE


class NaiveStrategy(BaseStrategy):
    """
    Strategia Naive: restituisce semplicemente il primo elemento
    della lista delle soluzioni ancora compatibili.
    """
    
    def choose_guess(self, possible_solutions: list[str]) -> str:
        return possible_solutions[0]


class MaxFrequencyStrategy(BaseStrategy):
    """
    Strategia Max Frequenza: sceglie l'ipotesi (tra le possibili_solutions)
    che massimizza la somma delle frequenze posizionali.
    """
    def _calculate_positional_frequencies(self, solutions: list[str]) -> list[dict[str, int]]:
        """
        Calcola la frequenza di ogni cifra (0-9) per ogni posizione (0-3).
        
        PRE: solutions è la lista delle soluzioni ancora valide.
        POST: Ritorna una lista di lunghezza DIM (es. 4). Ogni elemento è un dizionario
              dove la chiave è la cifra ('0', '1', ...) e il valore è il numero di 
              volte che compare in quella posizione in `solutions`.
        """
        frequencies = []
        matrix = [[solution[pos] for solution in solutions] for pos in range(0,DIM)]
        for p in range(0, DIM):
            p_frequencies={}
            for n in range(0,BASE):
                num = str(n)
                p_frequencies[num] = matrix[p].count(num)
            frequencies.append(p_frequencies)
        return frequencies


    def _evaluate_candidate_score(self, candidate: str, frequencies: list[dict[str, int]]) -> int:
        """
        Calcola il punteggio di una singola combinazione basandosi sulle frequenze.
        
        PRE: candidate è una stringa valida (es. "0123"). frequencies è l'output del metodo sopra.
        POST: Ritorna un intero che rappresenta la somma delle frequenze per quella combinazione.
        """
        score = 0
        for p in range(0,DIM):
            score += frequencies[p][candidate[p]]
        return score

    def choose_guess(self, possible_solutions: list[str]) -> str:
        """
        PRE: possible_solutions è una lista non vuota.
        POST: Ritorna il candidato (appartenente a possible_solutions) con il punteggio più alto.
        """
        frequencies = self._calculate_positional_frequencies(possible_solutions)
        guess = None
        best_score = -1
        for candidate in possible_solutions:
            candidate_score = self._evaluate_candidate_score(candidate, frequencies)
            if candidate_score > best_score:
                best_score = candidate_score
                guess = candidate
        return guess