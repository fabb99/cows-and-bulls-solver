"""
Modulo: tablebase (src/estimators)
Applica il Pattern Decorator: controlla se lo stato è già presente nella memoria
pre-calcolata (Tablebase). Se presente, ritorna il valore esatto in O(1);
altrimenti delega il calcolo a un altro BaseBucketEstimator.
"""
from typing import List, Dict
from src.estimators.base import BaseBucketEstimator

class TablebaseBucketEstimator(BaseBucketEstimator):

    def __init__(self, tablebase: Dict[str, float], fallback_estimator: BaseBucketEstimator):
        """
        PRE:
            - tablebase: Dizionario {state_id: costo_esatto} caricato in memoria.
            - fallback_estimator: Istanza di BaseBucketEstimator da usare se lo stato non è in tablebase.
        """
        self.tablebase = tablebase
        self.fallback_estimator = fallback_estimator

    def estimate_cost(self, solutions: List[str]) -> float:
        """
        LOGICA INTERNA:
        1. Genera lo state_id canonico ordinando le soluzioni con la virgola.
        2. Se lo state_id è presente in self.tablebase:
           - Ritorna immediatamente il costo memorizzato (O(1)).
        3. Se non è presente:
           - Ritorna self.fallback_estimator.estimate_cost(solutions).
        """
        state_id = self.generate_id(solutions)
        if state_id in self.tablebase:
            return self.tablebase[state_id]
        return self.fallback_estimator.estimate_cost(solutions)