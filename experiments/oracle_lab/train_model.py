"""
Modulo: train_model (Script)
Fase 4: addestra uno SegmentedLinearEstimator e lo salva come JSON auto-descrittivo.
Un solo script per tutti i modelli:
- geometrico (globale):  --splits               (nessuna soglia, un segmento)
- piecewise (default):   --splits 13 43
- asimmetrico:           --alpha 0.8            (expectile: sottostime pesate di più)

Per ogni segmento stampa le metriche IN-SAMPLE e di CROSS-VALIDATION e le salva nel
campo "meta" del modello, insieme all'hash del dataset di origine.

Esempi:
  python -m experiments.oracle_lab.train_model --splits --out data/runs/iter_00/models/geometric.json
  python -m experiments.oracle_lab.train_model --features log_n column_entropy digit_std --alpha 0.5
"""
import argparse
import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.oracle_lab.config import (
    DEFAULT_SPLITS,
    FEATURES_DATASET_FILEPATH,
    MODELS_DIR,
)
from experiments.oracle_lab.models import FeatureRecord
from experiments.oracle_lab.regression import (
    compute_regression_metrics,
    cross_validate,
    fit_linear,
    predict,
)
from experiments.oracle_lab.storage import load_features
from src.estimators.segmented import SegmentedLinearEstimator
from src.features.bucket_features import FEATURES

DEFAULT_FEATURES = ["log_n", "column_entropy", "digit_std"]


def train_segmented_model(
    records: list[FeatureRecord],
    features: Sequence[str],
    splits: Sequence[int],
    alpha: float = 0.5,
    k_folds: int = 5,
    seed: int = 0,
    dataset_sha256: str | None = None,
) -> SegmentedLinearEstimator:
    """
    PRE:  records con le feature richieste; splits crescenti.
          Ogni segmento (N <= s0, s0 < N <= s1, ..., N > s_last) ha almeno len(features) + 2 campioni.
    POST: estimator addestrato; meta contiene, per segmento, numero di campioni, di stati
          esatti, metriche in-sample e di cross-validation.
    """
    splits = sorted(int(s) for s in splits)
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        raise ValueError(f"Feature sconosciute: {unknown}")

    groups: list[list[FeatureRecord]] = [[] for _ in range(len(splits) + 1)]
    for r in records:
        groups[sum(s < r.n for s in splits)].append(r)  # stessa regola di SegmentedLinearEstimator

    segments, report = [], []
    for k, group in enumerate(groups):
        if len(group) < len(features) + 2:
            raise ValueError(f"Segmento {k}: solo {len(group)} campioni, troppo pochi")
        X = np.array([[r.features[f] for f in features] for r in group])
        y = np.array([r.target_cost for r in group])
        coeffs = fit_linear(X, y, alpha)
        segments.append(
            {
                "features": list(features),
                "intercept": float(coeffs[0]),
                "coef": [float(c) for c in coeffs[1:]],
            }
        )
        report.append(
            {
                "features": list(features),
                "n_samples": len(group),
                "n_exact": sum(r.is_exact for r in group),
                "in_sample": compute_regression_metrics(y, predict(X, coeffs)).to_dict(),
                "cv": cross_validate(X, y, alpha, k_folds, seed).to_dict(),
            }
        )

    return SegmentedLinearEstimator(
        splits=splits,
        segments=segments,
        alpha=alpha,
        meta={
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset_sha256": dataset_sha256,
            "k_folds": k_folds,
            "seed": seed,
            "segments": report,
        },
    )


def print_report(model: SegmentedLinearEstimator) -> None:
    """POST: stampa la tabella delle metriche per segmento e la formula di ciascuno."""
    print(f"\nSoglie: {model.splits} | alpha = {model.alpha}")
    print("=" * 84)
    print(f"{'Segmento':<9} | {'Campioni':>8} | {'Esatti':>6} | {'MAE in':>7} | {'MAE cv':>7} | {'R² in':>7} | {'R² cv':>7}")
    print("-" * 84)
    for k, rep in enumerate(model.meta["segments"]):
        i, c = rep["in_sample"], rep["cv"]
        print(f"{k:<9} | {rep['n_samples']:>8} | {rep['n_exact']:>6} | {i['mae']:>7.3f} | {c['mae']:>7.3f} | {i['r2']:>7.3f} | {c['r2']:>7.3f}")
    print("=" * 84)
    for k, seg in enumerate(model.segments):
        terms = " ".join(f"{c:+.4f}*{f}" for c, f in zip(seg.coef, seg.features, strict=True))
        print(f"Segmento {k}: costo = {seg.intercept:.4f} {terms}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Addestra uno SegmentedLinearEstimator.")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, choices=sorted(FEATURES))
    parser.add_argument("--splits", nargs="*", type=int, default=None,
                        help="soglie degli scaglioni; senza valori = modello globale")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=FEATURES_DATASET_FILEPATH)
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "model.json")
    args = parser.parse_args()

    records = load_features(args.dataset)
    if not records:
        print(f"{args.dataset} assente o vuoto: esegui prima feature_builder.")
        return
    splits = DEFAULT_SPLITS if args.splits is None else args.splits
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()

    model = train_segmented_model(
        records, args.features, splits, args.alpha, args.folds, args.seed, digest
    )
    print_report(model)
    model.save(args.out)
    print(f"\nModello salvato in {args.out}")


if __name__ == "__main__":
    main()