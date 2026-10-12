"""
Modulo: piecewise (src/estimators)
Contiene PiecewiseBucketEstimator:
Stimatore a 3 Scaglioni (Endgame, Midgame, Opening) basato sui tagli CART:
- Scaglione 1 (Endgame): N <= 13
- Scaglione 2 (Midgame): 13 < N <= 40
- Scaglione 3 (Opening): N > 40
Applica coefficienti geometrici specializzati per ciascuna fase di gioco.
"""
from typing import List
import math
import numpy as np
from collections import Counter
from src.core.constants import BASE, DIM
from src.estimators.base import BaseBucketEstimator


class PiecewiseBucketEstimator(BaseBucketEstimator):
    """
    Oracolo a 3 Scaglioni:
    Seleziona dinamicamente il modello lineare ottimale in base alla dimensione del bucket.
    """

    # 1. Le soglie matematiche scoperte da CART
    DEFAULT_SPLIT_ENDGAME: int = 13
    DEFAULT_SPLIT_MIDGAME: int = 40

    # 2. I Coefficienti dello Scaglione 1: ENDGAME (N <= 13)
    # MAE: +-0.113
    # 1. Scaglione ENDGAME (N <= 13) - MAE: ±0.148
    K_ENDGAME = {
        "k0": 1.6106,
        "k1_log_n": 0.1830,
        "k2_col_entropy": -0.1549,
        "k3_digit_std": 0.0961
    }

    # 2. Scaglione MIDGAME (13 < N <= 40) - MAE: ±0.100
    K_MIDGAME = {
        "k0": 1.4711,
        "k1_log_n": 0.3356,
        "k2_col_entropy": -0.2216,
        "k3_digit_std": 0.0149
    }

    # 3. Scaglione OPENING (N > 40) - MAE: ±0.093
    K_OPENING = {
        "k0": 1.4210,
        "k1_log_n": 0.3552,
        "k2_col_entropy": -0.2036,
        "k3_digit_std": 0.0010
    }

    def __init__(
        self,
        split_endgame: int = DEFAULT_SPLIT_ENDGAME,
        split_midgame: int = DEFAULT_SPLIT_MIDGAME,
        k_endgame: dict = None,
        k_midgame: dict = None,
        k_opening: dict = None
    ):
        """
        Inizializza lo stimatore a scaglioni memorizzando le soglie e i 3 set di coefficienti.
        """
        self.split_endgame = split_endgame
        self.split_midgame = split_midgame
        self.k_endgame = k_endgame or self.K_ENDGAME
        self.k_midgame = k_midgame or self.K_MIDGAME
        self.k_opening = k_opening or self.K_OPENING



    @staticmethod
    def _compute_digit_std(solutions: List[str]) -> float:
        all_chars = "".join(solutions)
        counts = Counter(all_chars)
        frequencies = [counts.get(str(i), 0) for i in range(10)]
        
        # Matematica pura veloce:
        mean_val = sum(frequencies) / 10.0
        variance = sum((x - mean_val) ** 2 for x in frequencies) / 10.0
        return math.sqrt(variance)

    @staticmethod
    def _compute_column_entropy(solutions: List[str]) -> float:
        """
        Calcola l'entropia media delle 4 colonne del gruppo.
        PRE: solutions lista non vuota.
        POST: Ritorna float (media delle 4 colonne).
        """
        acc_entropy = 0
        n_sol = len(solutions)
        for col in range(0,DIM):
            val_col = [s[col] for s in solutions]
            frequencies = {}
            for n in range(0,BASE):
                c = str(n)
                frequencies[c] =val_col.count(c)
            probabilities = [frequency/n_sol for frequency in frequencies.values()]
            h_col = -sum(p * math.log2(p) for p in probabilities if p > 0)
            acc_entropy += h_col
        return acc_entropy / DIM

    
    def estimate_cost(self, solutions: List[str]) -> float:
        """
        Calcola il costo del bucket instradandolo nello scaglione corretto.
        
        LOGICA INTERNA:
        1. Casi base deterministici inviolabili:
           - len == 1 -> 1.0
           - len == 2 -> 1.5
           
        2. Calcola le feature geometriche del bucket:
           - n = len(solutions)
           - log_n = math.log2(n)
           - digits_ratio = self._compute_unique_digits_ratio(solutions)
           - col_entropy = self._compute_column_entropy(solutions)
           
        3. Selezione dello Scaglione (Bivio Condizionale):
           - Se n <= self.split_endgame:
             Usa i coefficienti K_ENDGAME
           - Altrimenti se n <= self.split_midgame:
             Usa i coefficienti K_MIDGAME
           - Altrimenti (n > self.split_midgame):
             Usa i coefficienti K_OPENING
             
        4. Calcola l'equazione con i coefficienti selezionati:
           costo = k0 + (k1 * log_n) + (k2 * digits_ratio) + (k3 * col_entropy)
           
        5. Ritorna il costo (float).
        """
        n = len(solutions)
        if n == 1:
            return 1.0
        elif n == 2:
            return 1.5

        log_n = math.log2(n)
        digits_std = self._compute_digit_std(solutions)
        col_entropy = self._compute_column_entropy(solutions)

        if n <= self.split_endgame:
            k0, k1_log_n, k2_columns, k3_digits, = self.k_endgame.values()
        elif self.split_endgame<n<=self.split_midgame:
            k0, k1_log_n, k2_columns, k3_digits = self.k_midgame.values()
        else:
            k0, k1_log_n, k2_columns, k3_digits = self.k_opening.values()

        return k0 + (k1_log_n * log_n) + (k2_columns * col_entropy) + (k3_digits * digits_std)
        
        
        
        