"""
Modulo: segmented
Stimatore lineare a segmenti, con modello serializzato auto-descrittivo (schema v2).

    costo(N) = intercept_k + sum_i coef_k[i] * feature_{k,i}(bucket),   k = segmento di N

Ogni segmento ha le PROPRIE feature (anche nessuna: solo intercetta).
Un modello globale (geometrico) è il caso con zero soglie e un solo segmento.
Soglie e nomi delle feature vivono SOLO nel JSON: training e inferenza non possono divergere.
"""
import json
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.estimators.base import BaseBucketEstimator
from src.features.bucket_features import FEATURES, compute_features

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Segment:
    """Modello lineare di un segmento: intercept + sum coef[i] * feature[i]."""

    features: tuple[str, ...]
    intercept: float
    coef: tuple[float, ...]

    def __post_init__(self) -> None:
        """POST: ValueError se le feature sono sconosciute o duplicate, o se len(coef) != len(features)."""
        unknown = [f for f in self.features if f not in FEATURES]
        if unknown:
            raise ValueError(f"Feature sconosciute: {unknown}")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Feature duplicate nello stesso segmento")
        if len(self.coef) != len(self.features):
            raise ValueError("Numero di coefficienti diverso dal numero di feature")

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "intercept": self.intercept,
            "coef": list(self.coef),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        """PRE: data ha features, intercept, coef."""
        return cls(
            features=tuple(data["features"]),
            intercept=float(data["intercept"]),
            coef=tuple(float(c) for c in data["coef"]),
        )


class SegmentedLinearEstimator(BaseBucketEstimator):
    """
    Il segmento k vale per splits[k-1] < N <= splits[k]
    (primo segmento: N <= splits[0]; ultimo: N > splits[-1]).
    """

    def __init__(
        self,
        splits: Sequence[int],
        segments: Sequence[Segment | dict[str, Any]],
        alpha: float = 0.5,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        PRE:  splits strettamente crescenti; len(segments) == len(splits) + 1;
              ogni segmento è un Segment oppure un dict {features, intercept, coef}.
        POST: estimator pronto. Solleva ValueError se il modello è incoerente.
        """
        self.splits = [int(s) for s in splits]
        self.alpha = float(alpha)
        self.meta = dict(meta or {})

        if any(a >= b for a, b in zip(self.splits, self.splits[1:], strict=False)):
            raise ValueError("Le soglie devono essere strettamente crescenti")
        if len(segments) != len(self.splits) + 1:
            raise ValueError("Servono len(splits) + 1 segmenti")
        self.segments: list[Segment] = [
            s if isinstance(s, Segment) else Segment.from_dict(s) for s in segments
        ]

    @classmethod
    def from_common_features(
        cls,
        features: Sequence[str],
        splits: Sequence[int],
        segments: Sequence[dict[str, Any]],
        alpha: float = 0.5,
        meta: dict[str, Any] | None = None,
    ) -> "SegmentedLinearEstimator":
        """
        PRE:  segments = [{"intercept": float, "coef": [...]}, ...] con len(coef) == len(features).
        POST: estimator in cui tutti i segmenti usano le stesse feature.
        """
        full = [{"features": list(features), **seg} for seg in segments]
        return cls(splits, full, alpha, meta)

    # ------------------------------------------------------------ inferenza
    def segment_for(self, n: int) -> Segment:
        """PRE: n >= 1. POST: il segmento che copre N = n."""
        return self.segments[bisect_left(self.splits, n)]

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

        seg = self.segment_for(n)
        values = compute_features(solutions, list(seg.features))
        cost = seg.intercept + sum(c * v for c, v in zip(seg.coef, values, strict=True))
        return max(1.0, cost)

    # -------------------------------------------------------- serializzazione
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "splits": self.splits,
            "segments": [s.to_dict() for s in self.segments],
            "alpha": self.alpha,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentedLinearEstimator":
        """
        PRE:  data è un modello schema v1 (feature a livello modello) o v2.
        POST: estimator equivalente. Lo schema v1 viene convertito replicando le feature
              comuni su ogni segmento.
        """
        version = data.get("schema_version")
        if version == 1:
            return cls.from_common_features(
                features=data["features"],
                splits=data["splits"],
                segments=data["segments"],
                alpha=data.get("alpha", 0.5),
                meta=data.get("meta"),
            )
        if version != SCHEMA_VERSION:
            raise ValueError(f"schema_version non supportata: {version}")
        return cls(
            splits=data["splits"],
            segments=data["segments"],
            alpha=data.get("alpha", 0.5),
            meta=data.get("meta"),
        )

    def save(self, path: Path) -> None:
        """POST: scrive il modello (schema v2) su `path`, creando le cartelle mancanti."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def load(cls, path: Path) -> "SegmentedLinearEstimator":
        """PRE: path esiste. POST: modello caricato e validato (v1 o v2)."""
        with open(path, encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    # ------------------------------------------- modelli legacy (solo bootstrap)
    @classmethod
    def legacy_geometric(cls) -> "SegmentedLinearEstimator":
        """Coefficienti di default del vecchio GeometricBucketEstimator (v1)."""
        return cls.from_common_features(
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
        return cls.from_common_features(
            features=["log_n", "column_entropy", "digit_std"],
            splits=[13, 43],
            segments=[
                {"intercept": 1.6106, "coef": [0.1830, -0.1549, 0.0961]},
                {"intercept": 1.4711, "coef": [0.3356, -0.2216, 0.0149]},
                {"intercept": 1.4210, "coef": [0.3552, -0.2036, 0.0010]},
            ],
            meta={"origin": "legacy_v1 piecewise defaults"},
        )