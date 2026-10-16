"""
Modulo: segmented
Stimatore lineare a segmenti, con modello serializzato auto-descrittivo.

    costo(N) = intercept_k + sum_i coef_k[i] * feature_i(bucket),   k = segmento di N

Un modello "globale" (geometrico) è il caso con zero soglie e un solo segmento.
Le soglie e i nomi delle feature vivono SOLO nel JSON.
"""
import json
from bisect import bisect_left
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.estimators.base import BaseBucketEstimator
from src.features.bucket_features import FEATURES, compute_features

SCHEMA_VERSION = 1


class SegmentedLinearEstimator(BaseBucketEstimator):
    """
    Il segmento k vale per splits[k-1] < N <= splits[k]
    (primo segmento: N <= splits[0]; ultimo: N > splits[-1]).
    """

    def __init__(
        self,
        features: Sequence[str],
        splits: Sequence[int],
        segments: Sequence[dict[str, Any]],
        alpha: float = 0.5,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        PRE:  features ⊂ chiavi di FEATURES; splits strettamente crescenti;
              len(segments) == len(splits) + 1; ogni segmento è
              {"intercept": float, "coef": [float, ...]} con len(coef) == len(features).
        POST: estimator pronto. Solleva ValueError se il modello è incoerente.
        """
        self.features = list(features)
        self.splits = [int(s) for s in splits]
        self.alpha = float(alpha)
        self.meta = dict(meta or {})

        unknown = [f for f in self.features if f not in FEATURES]
        if unknown:
            raise ValueError(f"Feature sconosciute: {unknown}")
        if any(a >= b for a, b in zip(self.splits, self.splits[1:])):
            raise ValueError("Le soglie devono essere strettamente crescenti")
        if len(segments) != len(self.splits) + 1:
            raise ValueError("Servono len(splits) + 1 segmenti")

        self._segments: list[tuple[float, tuple[float, ...]]] = []
        for seg in segments:
            coef = tuple(float(c) for c in seg["coef"])
            if len(coef) != len(self.features):
                raise ValueError("Numero di coefficienti diverso dal numero di feature")
            self._segments.append((float(seg["intercept"]), coef))

    # ------------------------------------------------------------ inferenza
    def estimate_cost(self, solutions: list[str]) -> float:
        """
        PRE:  solutions non vuota.
        POST: N=1 -> 1.0, N=2 -> 1.5 (valori esatti); altrimenti la stima lineare del
              segmento di N, con pavimento a 1.0.
        """
        n = len(solutions)
        if n < 1:
            raise ValueError("solutions deve essere non vuota")
        if n == 1:
            return 1.0
        if n == 2:
            return 1.5

        intercept, coef = self._segments[bisect_left(self.splits, n)]
        values = compute_features(solutions, self.features)
        cost = intercept + sum(c * v for c, v in zip(coef, values))
        return max(1.0, cost)

    # -------------------------------------------------------- serializzazione
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "features": self.features,
            "splits": self.splits,
            "segments": [
                {"intercept": intercept, "coef": list(coef)}
                for intercept, coef in self._segments
            ],
            "alpha": self.alpha,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentedLinearEstimator":
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"schema_version non supportata: {version}")
        return cls(
            features=data["features"],
            splits=data["splits"],
            segments=data["segments"],
            alpha=data.get("alpha", 0.5),
            meta=data.get("meta"),
        )

    def save(self, path: Path) -> None:
        """POST: scrive il modello su `path` (crea le cartelle mancanti)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def load(cls, path: Path) -> "SegmentedLinearEstimator":
        """PRE: path esiste. POST: modello caricato e validato."""
        with open(path, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    # ------------------------------------------- modelli legacy (solo bootstrap)
    @classmethod
    def legacy_geometric(cls) -> "SegmentedLinearEstimator":
        """Coefficienti di default del vecchio GeometricBucketEstimator (v1)."""
        return cls(
            features=["log_n", "unique_digits_ratio", "column_entropy"],
            splits=[],
            segments=[{"intercept": 1.3732, "coef": [0.3466, -0.0173, -0.1854]}],
            meta={"origin": "legacy_v1 geometric defaults"},
        )

    @classmethod
    def legacy_piecewise(cls) -> "SegmentedLinearEstimator":
        """
        Coefficienti di default del vecchio PiecewiseBucketEstimator (v1).
        ATTENZIONE: il vecchio default di classe usava la soglia 40, il training 43;
        qui uso 43. Serve solo per far girare le strategie prima del riaddestramento.
        """
        return cls(
            features=["log_n", "column_entropy", "digit_std"],
            splits=[13, 43],
            segments=[
                {"intercept": 1.6106, "coef": [0.1830, -0.1549, 0.0961]},
                {"intercept": 1.4711, "coef": [0.3356, -0.2216, 0.0149]},
                {"intercept": 1.4210, "coef": [0.3552, -0.2036, 0.0010]},
            ],
            meta={"origin": "legacy_v1 piecewise defaults"},
        )