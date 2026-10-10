"""
Modulo: geometric (src/estimators)
Contiene GeometricBucketEstimator: stima la difficoltà di un bucket basandosi 
sulla dimensione logaritmica log2(N), la concentrazione delle cifre (Unique Digits Ratio)
e la mobilità posizionale (Entropia delle 4 colonne).
"""
from typing import List
import math
from src.core.constants import BASE, DIM
from src.estimators.base import BaseBucketEstimator

class GeometricBucketEstimator(BaseBucketEstimator):
    """
    Stimatore geometrico addestrato sui dati della Data Science (Modello 3).
    Formula: Costo = K0 + K1*log2(N) + K2*digits_ratio + K3*col_entropy
    """

    # Coefficienti ottimali addestrati su 4.121 stati (R² = 84.89%)
    DEFAULT_K0: float = 1.3732
    DEFAULT_K1_LOG_N: float = 0.3466
    DEFAULT_K2_DIGITS: float = -0.0173
    DEFAULT_K3_COLUMNS: float = -0.1854

    def __init__(
        self,
        k0: float = DEFAULT_K0,
        k1_log_n: float = DEFAULT_K1_LOG_N,
        k2_digits: float = DEFAULT_K2_DIGITS,
        k3_columns: float = DEFAULT_K3_COLUMNS
    ):
        """
        Inizializza lo stimatore geometrico con i pesi di regressione.
        """
        self.k0 = k0
        self.k1_log_n = k1_log_n
        self.k2_digits = k2_digits
        self.k3_columns = k3_columns

    def _compute_unique_digits_ratio(self, solutions: List[str]) -> float:
        """
        Funzione ausiliaria: calcola il rapporto di cifre distinte (tra 0.4 e 1.0).
        PRE: solutions lista non vuota.
        POST: Ritorna len(set_cifre) / BASE.
        """
        set_sol = {c for s in solutions for c in s}
        return len(set_sol) / BASE

    def _compute_column_entropy(self, solutions: List[str]) -> float:
        """
        Funzione ausiliaria: calcola la media dell'entropia posizionale delle 4 colonne.
        PRE: solutions lista non vuota.
        POST: Ritorna la media aritmetica dell'entropia di Shannon calcolata per ogni colonna.
        """
        acc_entropy = 0
        n_sol = len(solutions)
        for col in range(0,DIM):
            val_col = [s[col] for s in solutions]
            frequencies = self._calculate__frequencies(val_col)
            probabilities = [frequency/n_sol for frequency in frequencies.values()]
            h_col = -sum(p * math.log2(p) for p in probabilities if p > 0)
            acc_entropy += h_col
        return acc_entropy / DIM

    def _calculate__frequencies(self ,col: list[str]) -> dict[str, int]:
        """
        Calcola le frequenza delle cifre in una colonna.

        """
        frequencies = {}
        for n in range(0,BASE):
            c = str(n)
            frequencies[c] =col.count(c)
        return frequencies

    def estimate_cost(self, solutions: List[str]) -> float:
        """
        Calcola la stima della difficoltà del bucket.
        
        LOGICA INTERNA:
        1. Casi base deterministici:
           - len == 1 -> 1.0
           - len == 2 -> 1.5
        2. Calcola log2(len(solutions)).
        3. Calcola il rapporto cifre con _compute_unique_digits_ratio.
        4. Calcola l'entropia delle colonne con _compute_column_entropy.
        5. Calcola e ritorna l'equazione lineare:
           costo = k0 + (k1 * log_n) + (k2 * digits_ratio) + (k3 * col_entropy)
        """
        n = len(solutions)
        if n == 1:
            return 1.0
        elif n == 2:
            return 1.5

        log_n = math.log2(n)
        digits_ratio = self._compute_unique_digits_ratio(solutions)
        col_entropy = self._compute_column_entropy(solutions)

        return self.k0 + (self.k1_log_n * log_n) + (self.k2_digits * digits_ratio) + (self.k3_columns * col_entropy)



