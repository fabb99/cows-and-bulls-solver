"""
Modulo: base (src/estimators)
Definisce l'interfaccia astratta (BaseBucketEstimator) per qualsiasi modello 
o oracolo deputato alla stima del costo residuo di un singolo bucket di soluzioni.
"""
from abc import ABC, abstractmethod
from typing import List

class BaseBucketEstimator(ABC):
    """
    Classe base astratta per gli stimatori di costo dei bucket.
    """

    @abstractmethod
    def estimate_cost(self, solutions: List[str]) -> float:
        """
        Stima il numero medio di mosse necessarie per risolvere il gruppo di soluzioni fornito.
        
        PRE: solutions è una lista non vuota di combinazioni a 4 cifre distinte.
        POST: Ritorna il costo atteso minimo (float >= 1.0).
        """
        raise NotImplementedError("Ogni oracolo deve implementare questo metodo")

    @staticmethod
    def generate_id(state:List[str]):
        return ",".join(sorted(state))
