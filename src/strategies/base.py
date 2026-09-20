"""
Modulo: base
Definisce l'interfaccia standard (Classe Astratta) per tutte le strategie.
"""
from src.core.lookup_table import FeedbackLookup

class BaseStrategy:
    """
    Classe base astratta per i risolutori di Bulls and Cows.
    """
    
    def __init__(self, all_permutations: list[str], lookup: FeedbackLookup):
        """
        Inizializza la strategia.
        PRE: all_permutations contiene tutte le 5040 combinazioni. lookup è inizializzato.
        """
        self.all_permutations = all_permutations
        self.lookup = lookup

    def choose_guess(self, possible_solutions: list[str], history: list[str] = None) -> str:
        """
        PRE: possible_solutions è una lista non vuota di combinazioni ancora valide.
        POST: Restituisce la stringa (es. "0123") della mossa scelta.
        """
        raise NotImplementedError("Ogni strategia deve implementare questo metodo.")