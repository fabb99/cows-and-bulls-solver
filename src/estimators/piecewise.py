"""
Modulo: piecewise (src/estimators)
Contiene PiecewiseBucketEstimator:
Carica automaticamente le soglie e i 3 set di coefficienti da data/models/piecewise_model.json
oppure usa i valori di default della classe come fallback.
"""
from typing import List, Optional
from pathlib import Path
import json
import math
from collections import Counter

from src.core.constants import BASE, DIM
from src.estimators.base import BaseBucketEstimator

DEFAULT_WEIGHTS_FILEPATH = Path("data/models/piecewise_model.json")


class PiecewiseBucketEstimator(BaseBucketEstimator):
    """
    Oracolo a 3 Scaglioni (Endgame, Midgame, Opening).
    """

    DEFAULT_SPLIT_ENDGAME: int = 13
    DEFAULT_SPLIT_MIDGAME: int = 40

    K_ENDGAME = {
        "k0": 1.6106,
        "k1_log_n": 0.1830,
        "k2_columns": -0.1549,
        "k3_digit_std": 0.0961
    }

    K_MIDGAME = {
        "k0": 1.4711,
        "k1_log_n": 0.3356,
        "k2_columns": -0.2216,
        "k3_digit_std": 0.0149
    }

    K_OPENING = {
        "k0": 1.4210,
        "k1_log_n": 0.3552,
        "k2_columns": -0.2036,
        "k3_digit_std": 0.0010
    }

    def __init__(
        self,
        weights_path: Optional[Path] = DEFAULT_WEIGHTS_FILEPATH,
        split_endgame: Optional[int] = None,
        split_midgame: Optional[int] = None,
        k_endgame: Optional[dict] = None,
        k_midgame: Optional[dict] = None,
        k_opening: Optional[dict] = None
    ):
        # 1. Caricamento da file JSON (se presente)
        data = {}
        if weights_path is not None and weights_path.exists():
            try:
                with open(weights_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception:
                data = {}

        # 2. Assegnazione parametri con fallback
        self.split_endgame = split_endgame if split_endgame is not None else data.get("split_endgame", self.DEFAULT_SPLIT_ENDGAME)
        self.split_midgame = split_midgame if split_midgame is not None else data.get("split_midgame", self.DEFAULT_SPLIT_MIDGAME)

        self.k_endgame = k_endgame if k_endgame is not None else data.get("k_endgame", self.K_ENDGAME)
        self.k_midgame = k_midgame if k_midgame is not None else data.get("k_midgame", self.K_MIDGAME)
        self.k_opening = k_opening if k_opening is not None else data.get("k_opening", self.K_OPENING)

    @staticmethod
    def _compute_digit_std(solutions: List[str]) -> float:
        """Calcolo veloce in C-Python della deviazione standard delle frequenze delle cifre."""
        all_chars = "".join(solutions)
        counts = Counter(all_chars)
        frequencies = [counts.get(str(i), 0) for i in range(10)]
        mean_val = sum(frequencies) / 10.0
        variance = sum((x - mean_val) ** 2 for x in frequencies) / 10.0
        return math.sqrt(variance)

    @staticmethod
    def _compute_column_entropy(solutions: List[str]) -> float:
        """Calcolo veloce dell'entropia posizionale delle colonne."""
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
        digit_std = self._compute_digit_std(solutions)
        col_entropy = self._compute_column_entropy(solutions)

        # Selezione dinamica dei pesi dello scaglione
        if n <= self.split_endgame:
            k = self.k_endgame
        elif n <= self.split_midgame:
            k = self.k_midgame
        else:
            k = self.k_opening

        # Accesso sicuro per chiave esplicita
        return (
            k["k0"] 
            + (k["k1_log_n"] * log_n) 
            + (k["k2_columns"] * col_entropy) 
            + (k["k3_digit_std"] * digit_std)
        )