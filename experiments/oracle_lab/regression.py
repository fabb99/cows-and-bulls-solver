"""
Modulo: regression (experiments/oracle_lab)
Regressione lineare simmetrica (OLS) o asimmetrica (expectile / Asymmetric Least Squares,
risolta con IRLS) e cross-validation K-fold.
Unica implementazione, usata da train_model.py.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """
    PRE:  y_true e y_pred sono array 1D della stessa lunghezza (non vuoti).
    POST: MAE, RMSE e R². Se la varianza di y_true è nulla, R² = 0.0.
    """
    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return RegressionMetrics(
        mae=float(np.mean(np.abs(residuals))),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
    )


def _design(X: np.ndarray) -> np.ndarray:
    """POST: X con una colonna di 1 davanti (intercetta)."""
    return np.column_stack([np.ones(len(X)), X])


def fit_linear(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    PRE:  X (n, p), y (n,), 0 < alpha < 1.
    POST: coefficienti [intercetta, c1, ..., cp].
          alpha = 0.5: OLS.
          alpha > 0.5: expectile regression: i residui positivi (sottostime: la realtà è
          peggiore della stima) pesano alpha, quelli negativi 1 - alpha. Con alpha = 0.8
          una sottostima costa 4 volte una sovrastima. Risolta con IRLS.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha deve essere in (0, 1)")
    A = _design(X)
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    if alpha == 0.5:
        return coeffs
    for _ in range(max_iter):
        weights = np.where(y - A @ coeffs > 0, alpha, 1.0 - alpha)
        sqrt_w = np.sqrt(weights)
        new = np.linalg.lstsq(A * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
        converged = float(np.max(np.abs(new - coeffs))) < tol
        coeffs = new
        if converged:
            break
    return coeffs


def predict(X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """POST: predizioni per le righe di X."""
    return _design(X) @ coeffs


def cross_validate(
    X: np.ndarray, y: np.ndarray, alpha: float = 0.5, k: int = 5, seed: int = 0
) -> RegressionMetrics:
    """
    PRE:  almeno 2 campioni.
    POST: metriche calcolate sulle predizioni FUORI CAMPIONE di un K-fold casuale
          (con seed fisso). Limite noto: gli stati annidati sono correlati, quindi
          le metriche sono ottimistiche rispetto a stati davvero nuovi.
    """
    n = len(y)
    k = min(k, n)
    if k < 2:
        raise ValueError("Servono almeno 2 campioni per la cross-validation")
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    predictions = np.empty(n)
    for i, test in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        coeffs = fit_linear(X[train], y[train], alpha)
        predictions[test] = predict(X[test], coeffs)
    return compute_regression_metrics(y, predictions)