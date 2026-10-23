"""
Modulo: select_model (Script)
Selezione del modello a fasi: trova insieme
  - la partizione dell'asse N in K fasi (soglie), e
  - il sottoinsieme di feature di OGNI fase,
che minimizza l'errore fuori fold (cross-validation a fold raggruppati per ramo radice).

Algoritmo: programmazione dinamica sulle soglie. Il costo di una fase (a, b] è l'SSE
fuori fold del miglior sottoinsieme di feature (il più semplice entro `tol` dal migliore).
L'SSE è additivo tra fasi, quindi la partizione ottima per ogni K è esatta (sui tagli
candidati). K è il più piccolo entro `tol_k` dal migliore.

Uso: python -m experiments.oracle_lab.select_model [--max-phases 4] [--min-samples 50]
"""
import argparse
import hashlib
import itertools
import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiments.oracle_lab.config import FEATURES_DATASET_FILEPATH, MODELS_DIR
from experiments.oracle_lab.models import FeatureRecord
from experiments.oracle_lab.regression import fit_linear
from experiments.oracle_lab.storage import load_features
from src.core.feedback import calculate_feedback
from src.estimators.segmented import Segment, SegmentedLinearEstimator
from src.features.bucket_features import FEATURES

FEATURE_NAMES: list[str] = list(FEATURES)
FIRST_GUESS = "0123"
MIXED_GROUP = (-1, -1)
RIDGE = 1e-8


# ---------------------------------------------------------------- dati e fold
@dataclass(frozen=True)
class Dataset:
    """Stati ordinati per N crescente. F ha le colonne nell'ordine di FEATURE_NAMES."""

    n: np.ndarray
    F: np.ndarray
    y: np.ndarray
    is_exact: np.ndarray
    groups: list


def root_branch_group(record: FeatureRecord) -> Any:
    """
    POST: feedback del primo tentativo ('0123') comune a tutte le soluzioni dello stato
          (il ramo radice da cui discende), oppure MIXED_GROUP se non è comune.
    """
    feedbacks = {calculate_feedback(FIRST_GUESS, s) for s in record.state_id.split(",")}
    return next(iter(feedbacks)) if len(feedbacks) == 1 else MIXED_GROUP


def prepare(
    records: list[FeatureRecord], group_fn: Callable[[FeatureRecord], Any] = root_branch_group
) -> Dataset:
    """
    PRE:  records non vuota, con tutte le feature del registry.
    POST: Dataset ordinato per (N, state_id): ordine deterministico.
    """
    if not records:
        raise ValueError("records è vuota")
    rs = sorted(records, key=lambda r: (r.n, r.state_id))
    return Dataset(
        n=np.array([r.n for r in rs], dtype=int),
        F=np.array([[r.features[f] for f in FEATURE_NAMES] for r in rs], dtype=float),
        y=np.array([r.target_cost for r in rs], dtype=float),
        is_exact=np.array([r.is_exact for r in rs], dtype=bool),
        groups=[group_fn(r) for r in rs],
    )


def make_folds(groups: Sequence, k: int = 5, seed: int = 0) -> np.ndarray:
    """
    PRE:  almeno 2 gruppi distinti.
    POST: id di fold (0..k'-1, con k' = min(k, n_gruppi)) per ogni stato. Un gruppo sta
          sempre in un solo fold; i gruppi grandi vanno per primi nel fold meno carico.
          Deterministico dato il seed.
    """
    sizes = Counter(groups)
    labels = sorted(sizes, key=repr)
    if len(labels) < 2:
        raise ValueError("Servono almeno 2 gruppi per la cross-validation")
    k = min(k, len(labels))
    perm = np.random.default_rng(seed).permutation(len(labels))
    order = sorted((labels[i] for i in perm), key=lambda g: -sizes[g])  # sort stabile
    load = [0] * k
    assign = {}
    for g in order:
        fold = load.index(min(load))
        assign[g] = fold
        load[fold] += sizes[g]
    return np.array([assign[g] for g in groups], dtype=int)


# ------------------------------------------------------- SSE fuori fold esatto
def feature_subsets(n_features: int, max_features: int) -> list[tuple[int, ...]]:
    """POST: tutti i sottoinsiemi di indici (compreso il vuoto) con al più max_features elementi."""
    top = min(max_features, n_features)
    return [c for r in range(top + 1) for c in itertools.combinations(range(n_features), r)]


class CVTables:
    """
    Somme cumulate per fold di ZᵀZ, Zᵀy e yᵀy (Z = [1, feature]) sugli stati ordinati per N.
    Permette l'SSE fuori fold di qualsiasi (intervallo, sottoinsieme) senza ripassare i dati.
    """

    def __init__(self, ds: Dataset, folds: np.ndarray) -> None:
        m = len(ds.y)
        self.k = int(folds.max()) + 1
        Z = np.column_stack([np.ones(m), ds.F])
        p1 = Z.shape[1]
        outer = np.einsum("ni,nj->nij", Z, Z)
        zy = Z * ds.y[:, None]
        yy = ds.y**2
        self.G = np.zeros((self.k, m + 1, p1, p1))
        self.c = np.zeros((self.k, m + 1, p1))
        self.s = np.zeros((self.k, m + 1))
        for f in range(self.k):
            mask = folds == f
            self.G[f, 1:] = np.cumsum(outer * mask[:, None, None], axis=0)
            self.c[f, 1:] = np.cumsum(zy * mask[:, None], axis=0)
            self.s[f, 1:] = np.cumsum(yy * mask)

    def sst(self, lo: int, hi: int) -> float:
        """POST: somma dei quadrati totale di y rispetto alla media, sugli stati [lo, hi)."""
        n = (self.G[:, hi, 0, 0] - self.G[:, lo, 0, 0]).sum()
        sy = (self.c[:, hi, 0] - self.c[:, lo, 0]).sum()
        syy = (self.s[:, hi] - self.s[:, lo]).sum()
        return float(syy - sy * sy / n)

    def subset_sse(self, lo: int, hi: int, subsets: list[tuple[int, ...]]) -> np.ndarray:
        """
        PRE:  0 <= lo < hi <= numero di stati.
        POST: per ogni sottoinsieme (indici di feature) l'SSE fuori fold sugli stati [lo, hi),
              con intercetta. Fold con train troppo piccolo: predice la media del train
              (o del segmento se il train è vuoto).
        """
        Gf = self.G[:, hi] - self.G[:, lo]
        cf = self.c[:, hi] - self.c[:, lo]
        sf = self.s[:, hi] - self.s[:, lo]
        Gt, ct = Gf.sum(axis=0), cf.sum(axis=0)
        out = np.empty(len(subsets))
        for t, subset in enumerate(subsets):
            cols = [0] + [1 + j for j in subset]
            q = len(cols)
            Gs = Gf[:, cols][:, :, cols]
            cs = cf[:, cols]
            Gtr = Gt[np.ix_(cols, cols)] - Gs
            ctr = ct[cols] - cs
            n_train = Gtr[:, 0, 0]
            ridge = RIDGE * np.maximum(np.einsum("kii->k", Gtr) / q, 1.0)
            beta = np.linalg.solve(Gtr + ridge[:, None, None] * np.eye(q), ctr[..., None])[..., 0]
            weak = n_train < q + 1
            if weak.any():
                fallback = np.where(
                    n_train > 0, ctr[:, 0] / np.maximum(n_train, 1), ct[0] / Gt[0, 0]
                )
                beta[weak] = 0.0
                beta[weak, 0] = fallback[weak]
            sse = sf - 2 * np.einsum("ki,ki->k", beta, cs) + np.einsum("ki,kij,kj->k", beta, Gs, beta)
            out[t] = max(float(sse.sum()), 0.0)
        return out


# ----------------------------------------------------------------- DP sulle fasi
@dataclass(frozen=True)
class SegmentFit:
    """Fase (lo, hi] di N con il suo sottoinsieme di feature e l'errore fuori fold."""

    lo: int
    hi: int
    features: tuple[str, ...]
    cv_sse: float
    sst: float
    n: int
    n_exact: int

    @property
    def cv_rmse(self) -> float:
        return math.sqrt(self.cv_sse / self.n)

    @property
    def cv_r2(self) -> float:
        return 1.0 - self.cv_sse / self.sst if self.sst > 0 else 0.0


class PartitionSearch:
    def __init__(
        self,
        ds: Dataset,
        folds: np.ndarray,
        min_samples: int = 50,
        n_candidates: int = 50,
        max_features: int = len(FEATURE_NAMES),
        tol: float = 0.005,
    ) -> None:
        """
        PRE:  ds ordinato per N; folds da make_folds; min_samples >= 1.
        POST: pronto a cercare. Le fasi candidate sono (P[i], P[j]] con P = [0, tagli..., max N].
        """
        self.ds = ds
        self.tables = CVTables(ds, folds)
        self.subsets = feature_subsets(len(FEATURE_NAMES), max_features)
        self.min_samples = min_samples
        self.tol = tol
        self.points = self._candidate_points(n_candidates)
        self.idx = np.searchsorted(ds.n, self.points, side="right")
        self._cache: dict[tuple[int, int], SegmentFit | None] = {}

    def _candidate_points(self, n_candidates: int) -> list[int]:
        ns = self.ds.n
        distinct = np.unique(ns)
        interior = distinct[(distinct > distinct[0]) & (distinct < distinct[-1])]
        if len(interior) > n_candidates:
            qs = np.quantile(ns, np.linspace(0, 1, n_candidates + 2)[1:-1], method="lower")
            interior = np.unique(qs[(qs > distinct[0]) & (qs < distinct[-1])])
        return [0, *(int(v) for v in interior), int(distinct[-1])]

    def segment_fit(self, i: int, j: int) -> SegmentFit | None:
        """
        PRE:  0 <= i < j < len(points).
        POST: miglior fit della fase (P[i], P[j]] oppure None se ha meno di min_samples stati.
              Sottoinsieme: il più piccolo con SSE <= migliore + tol * SST (cioè R² cv entro tol dal migliore).
        """
        key = (i, j)
        if key in self._cache:
            return self._cache[key]
        lo, hi = int(self.idx[i]), int(self.idx[j])
        fit = None
        if hi - lo >= self.min_samples:
            sse = self.tables.subset_sse(lo, hi, self.subsets)
            sst = self.tables.sst(lo, hi)
            # una feature entra solo se aumenta l'R² fuori fold di almeno `tol`
            threshold = sse.min() + self.tol * sst + 1e-12
            chosen = min(
                (t for t in range(len(self.subsets)) if sse[t] <= threshold),
                key=lambda t: (len(self.subsets[t]), sse[t]),
            )
            fit = SegmentFit(
                lo=self.points[i],
                hi=self.points[j],
                features=tuple(FEATURE_NAMES[f] for f in self.subsets[chosen]),
                cv_sse=float(sse[chosen]),
                sst=sst,
                n=hi - lo,
                n_exact=int(self.ds.is_exact[lo:hi].sum()),
            )
        self._cache[key] = fit
        return fit

    def best_partitions(self, max_phases: int) -> dict[int, list[SegmentFit]]:
        """
        PRE:  max_phases >= 1.
        POST: {K: partizione di errore minimo in K fasi}, per i soli K fattibili
              (servono almeno K * min_samples stati e abbastanza tagli candidati).
        """
        last = len(self.points) - 1
        best = np.full((max_phases + 1, last + 1), np.inf)
        back = np.zeros((max_phases + 1, last + 1), dtype=int)
        best[0, 0] = 0.0
        for k in range(1, max_phases + 1):
            for j in range(1, last + 1):
                for i in range(j):
                    if not np.isfinite(best[k - 1, i]):
                        continue
                    fit = self.segment_fit(i, j)
                    if fit is None:
                        continue
                    value = best[k - 1, i] + fit.cv_sse
                    if value < best[k, j]:
                        best[k, j], back[k, j] = value, i

        partitions: dict[int, list[SegmentFit]] = {}
        for k in range(1, max_phases + 1):
            if not np.isfinite(best[k, last]):
                continue
            fits, j = [], last
            for level in range(k, 0, -1):
                i = int(back[level, j])
                fits.append(self.segment_fit(i, j))
                j = i
            partitions[k] = fits[::-1]
        return partitions


def total_sse(partition: list[SegmentFit]) -> float:
    return sum(f.cv_sse for f in partition)


def choose_k(partitions: dict[int, list[SegmentFit]], tol_k: float = 0.02) -> int:
    """
    PRE:  partitions non vuoto.
    POST: il K più piccolo il cui SSE totale è <= migliore * (1 + tol_k).
    """
    totals = {k: total_sse(p) for k, p in partitions.items()}
    threshold = min(totals.values()) * (1 + tol_k) + 1e-12
    return min(k for k, t in totals.items() if t <= threshold)


def build_estimator(
    ds: Dataset,
    partition: list[SegmentFit],
    alpha: float = 0.5,
    meta: dict[str, Any] | None = None,
) -> SegmentedLinearEstimator:
    """
    PRE:  partition da best_partitions.
    POST: estimator con coefficienti rifittati su tutti i dati di ogni fase
          (OLS se alpha = 0.5, altrimenti expectile). meta riporta le metriche CV per fase.
    """
    segments, report = [], []
    for fit in partition:
        lo = int(np.searchsorted(ds.n, fit.lo, side="right"))
        hi = int(np.searchsorted(ds.n, fit.hi, side="right"))
        cols = [FEATURE_NAMES.index(f) for f in fit.features]
        coeffs = fit_linear(ds.F[lo:hi][:, cols], ds.y[lo:hi], alpha)
        segments.append(Segment(fit.features, float(coeffs[0]), tuple(float(c) for c in coeffs[1:])))
        report.append(
            {
                "lo": fit.lo,
                "hi": fit.hi,
                "features": list(fit.features),
                "n_samples": fit.n,
                "n_exact": fit.n_exact,
                "cv_rmse": fit.cv_rmse,
                "cv_r2": fit.cv_r2,
            }
        )
    return SegmentedLinearEstimator(
        splits=[f.hi for f in partition[:-1]],
        segments=segments,
        alpha=alpha,
        meta={**(meta or {}), "segments": report},
    )


# ------------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Selezione di soglie e feature per fase (DP + CV).")
    parser.add_argument("--dataset", type=Path, default=FEATURES_DATASET_FILEPATH)
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "selected.json")
    parser.add_argument("--max-phases", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=50, help="numero di tagli candidati")
    parser.add_argument("--max-features", type=int, default=len(FEATURE_NAMES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=0.005, help="una feature entra solo se aumenta l'R² cv di almeno questo valore")
    parser.add_argument("--tol-k", type=float, default=0.02, help="tolleranza sul numero di fasi")
    parser.add_argument("--alpha", type=float, default=0.5, help="solo per il rifit finale")
    args = parser.parse_args()

    records = load_features(args.dataset)
    if not records:
        raise SystemExit(f"{args.dataset} assente o vuoto: esegui prima feature_builder.")
    ds = prepare(records)
    folds = make_folds(ds.groups, args.folds, args.seed)
    print(f"Stati: {len(ds.y)} ({int(ds.is_exact.sum())} esatti) | gruppi: {len(set(ds.groups))} "
          f"| fold: {np.bincount(folds).tolist()}")

    start = time.time()
    search = PartitionSearch(ds, folds, args.min_samples, args.candidates, args.max_features, args.tol)
    partitions = search.best_partitions(args.max_phases)
    print(f"Ricerca completata in {time.time() - start:.1f} s "
          f"({len(search.points) - 1} punti di taglio candidati)")
    if not partitions:
        raise SystemExit("Nessuna partizione fattibile: abbassa --min-samples.")

    print("\n  K | SSE cv   | RMSE cv | soglie")
    for k, part in sorted(partitions.items()):
        sse = total_sse(part)
        print(f"{k:>3} | {sse:>8.3f} | {math.sqrt(sse / len(ds.y)):>7.4f} | {[f.hi for f in part[:-1]]}")

    k = choose_k(partitions, args.tol_k)
    chosen = partitions[k]
    print(f"\nK scelto = {k}")
    print(f"{'Fase':<14} | {'Campioni':>8} | {'Esatti %':>8} | {'RMSE cv':>7} | {'R² cv':>6} | Feature")
    for fit in chosen:
        share = 100 * fit.n_exact / fit.n
        flag = "  <- target prevalentemente upper bound" if share < 50 else ""
        print(f"({fit.lo:>4},{fit.hi:>5}] | {fit.n:>8} | {share:>7.1f}% | {fit.cv_rmse:>7.4f} | "
              f"{fit.cv_r2:>6.3f} | {', '.join(fit.features) or '(solo intercetta)'}{flag}")

    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    model = build_estimator(
        ds, chosen, args.alpha,
        meta={
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset_sha256": digest,
            "selection": {
                "max_phases": args.max_phases, "min_samples": args.min_samples,
                "candidates": args.candidates, "max_features": args.max_features,
                "folds": args.folds, "seed": args.seed, "tol": args.tol, "tol_k": args.tol_k,
                "cv_total_sse": total_sse(chosen),
            },
        },
    )
    model.save(args.out)
    print(f"\nModello salvato in {args.out}")


if __name__ == "__main__":
    main()