"""Scoring functions and statistical tests for forecast evaluation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import statsmodels.api as sm


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((actual - predicted) ** 2))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(actual, predicted)))


METRIC_REGISTRY: dict[str, Callable] = {
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
}


def diebold_mariano_hac(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    maxlags: int | None = None,
) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test with HAC standard errors.

    Tests H0: E[loss_a - loss_b] = 0.

    Returns:
        ``(t_stat, p_value)`` tuple.
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    t = len(d)
    if t < 2:
        return float("nan"), float("nan")
    if maxlags is None:
        maxlags = max(1, int(np.floor(4 * (t / 100) ** (2 / 9))))
    res = sm.OLS(d, np.ones(t)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.tvalues[0]), float(res.pvalues[0])
