"""
Modulo: geometric (src/estimators)
Contiene GeometricBucketEstimator:
Carica i coefficienti addestrati da data/models/geometric_model.json (se esistente)
oppure usa i valori di default della classe come fallback.
"""
from typing import List, Optional
from pathlib import Path
import json
import math
from collections import Counter

from src.core.constants import BASE, DIM
from src.estimators.base import BaseBucketEstimator

DEFAULT_WEIGHTS_FILEPATH = Path("data/models/geometric_model.json")

class GeometricBucketEstimator(BaseBucketEstimator):
    """
    Stimatore geometrico globale a formula unica.
    Costo = K0 + K1*log2(N) + K2*digits + K3*columns
    """

    # Valori di fallback nel caso in cui il file JSON non esista
    DEFAULT_K0: float = 1.3732
    DEFAULT_K1_LOG_N: float = 0.3466
    DEFAULT_K2_DIGITS: float = -0.0173
    DEFAULT_K3_COLUMNS: float = -0.1854

    def __init__(
        self,
        weights_path: Optional[Path] = DEFAULT_WEIGHTS_FILEPATH,
        k0: Optional[float] = None,
        k1_log_n: Optional[float] = None,
        k2_digits: Optional[float] = None,
        k3_columns: Optional[float] = None
    ):
        # 1. Caricamento da file JSON (se presente)
        data = {}
        if weights_path is not None and weights_path.exists():
            try:
                with open(weights_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception:
                data = {}

        # 2. Assegnazione: Priorità parametro esplicito > JSON da file > Default della classe
        self.k0 = k0 if k0 is not None else data.get("k0", self.DEFAULT_K0)
        self.k1_log_n = k1_log_n if k1_log_n is not None else data.get("k1_log_n", self.DEFAULT_K1_LOG_N)
        self.k2_digits = k2_digits if k2_digits is not None else data.get("k2_digits", self.DEFAULT_K2_DIGITS)
        self.k3_columns = k3_columns if k3_columns is not None else data.get("k3_columns", self.DEFAULT_K3_COLUMNS)

    @staticmethod
    def _compute_unique_digits_ratio(solutions: List[str]) -> float:
        set_sol = {c for s in solutions for c in s}
        return len(set_sol) / BASE

    @staticmethod
    def _compute_column_entropy(solutions: List[str]) -> float:
        acc_entropy = 0.0
        n_sol = len(solutions)
        for col in range(DIM):
            val_col = [s[col] for s in solutions]
            counts = Counter(val_col)
            h_col = -sum((count / n_sol) * math.log2(count / n_sol) for count in counts.values())
            acc_entropy += h_col
        return acc_entropy / DIM

    def estimate_cost(self, solutions: List[str]) -> float:
        n = len(solutions)
        if n == 1:
            return 1.0
        elif n == 2:
            return 1.5

        log_n = math.log2(n)
        digits_ratio = self._compute_unique_digits_ratio(solutions)
        col_entropy = self._compute_column_entropy(solutions)

        return (
            self.k0 
            + (self.k1_log_n * log_n) 
            + (self.k2_digits * digits_ratio) 
            + (self.k3_columns * col_entropy)
        )