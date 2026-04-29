"""AutoARIMAForecaster -- data-driven ARIMA order selection via BIC.

Workflow per fit() call:
  1. Choose d in {0, 1, ..., d_max} using sequential ADF tests.
     Stop at the smallest d for which the differenced series is stationary
     (ADF p-value < adf_alpha).  This avoids over-differencing.
  2. Grid-search p in {0..p_max} x q in {0..q_max} with fixed d.
     Select the order minimising statsmodels' BIC.
  3. Keep the winning fit result for predict / predict_quantiles.

The forecaster name is the fixed string "AutoARIMA" (used as a dict key in
BenchmarkResults).  The chosen order for the last fit is available via
self.best_order_ after fit() has been called.
"""

from __future__ import annotations

import math
import warnings
from itertools import product

import numpy as np

from benchmark.forecasters.base import Forecaster

_AUTO_ARIMA_QUANTILE_LEVELS = np.linspace(0.1, 0.9, 9, dtype=np.float64)


def _std_normal_ppf(levels: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import norm
        return np.asarray(norm.ppf(np.clip(levels, 1e-12, 1 - 1e-12)), dtype=np.float64)
    except Exception:
        out = np.empty(len(levels), dtype=np.float64)
        for i, p in enumerate(levels):
            p = float(np.clip(p, 1e-12, 1 - 1e-12))
            out[i] = math.sqrt(2.0) * math.erfinv(2.0 * p - 1.0)
        return out


def _choose_d(y: np.ndarray, d_max: int, adf_alpha: float) -> int:
    """Return the smallest d in {0..d_max} such that diff(y, d) is stationary."""
    from statsmodels.tsa.stattools import adfuller

    for d in range(d_max + 1):
        y_test = np.diff(y, n=d) if d > 0 else y
        if len(y_test) < 10:
            return d
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p_val, *_ = adfuller(y_test, autolag="AIC")
            if p_val < adf_alpha:
                return d
        except Exception:
            return d
    return d_max


class AutoARIMAForecaster(Forecaster):
    """Automated ARIMA(p,d,q) selection using ADF for d and BIC for (p,q).

    Parameters
    ----------
    p_max: Maximum AR order to search.
    q_max: Maximum MA order to search.
    d_max: Maximum integration order (ADF tests stop earlier if possible).
    adf_alpha: Significance level for ADF stationarity test.
    """

    name: str = "AutoARIMA"

    def __init__(
        self,
        p_max: int = 4,
        q_max: int = 2,
        d_max: int = 2,
        adf_alpha: float = 0.05,
    ) -> None:
        self.p_max = p_max
        self.q_max = q_max
        self.d_max = d_max
        self.adf_alpha = adf_alpha
        self.best_order_: tuple[int, int, int] | None = None
        self._best_bic: float = float("inf")
        self._fit_result = None

    # ------------------------------------------------------------------

    def fit(self, history: np.ndarray) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        y = np.asarray(history, dtype=np.float64)

        # Step 1: choose d
        d = _choose_d(y, self.d_max, self.adf_alpha)

        # Step 2: BIC grid search over (p, q) with fixed d
        best_bic = float("inf")
        best_order: tuple[int, int, int] = (1, d, 0)
        best_result = None

        for p, q in product(range(self.p_max + 1), range(self.q_max + 1)):
            if p == 0 and q == 0:
                continue  # no dynamics — skip
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(y, order=(p, d, q)).fit()
                if res.bic < best_bic:
                    best_bic = res.bic
                    best_order = (p, d, q)
                    best_result = res
            except Exception:
                continue

        # Fallback: if all fits failed, use ARIMA(1, d, 0)
        if best_result is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_result = ARIMA(y, order=(1, d, 0)).fit()
            best_order = (1, d, 0)

        self.best_order_ = best_order
        self._best_bic = best_bic
        self._fit_result = best_result

    def predict(self, horizon: int) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("Must call fit() before predict()")
        fc = self._fit_result.forecast(steps=horizon)
        return np.asarray(fc, dtype=np.float64).ravel()

    def predict_quantiles(
        self, horizon: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self._fit_result is None:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fr = self._fit_result.get_forecast(steps=horizon)
            mu = np.asarray(fr.predicted_mean, dtype=np.float64).ravel()
            var = np.asarray(fr.var_pred_mean, dtype=np.float64).ravel()
        except Exception:
            return None
        if mu.size != horizon or var.size != horizon:
            return None
        std = np.sqrt(np.maximum(var, 1e-20))
        z = _std_normal_ppf(_AUTO_ARIMA_QUANTILE_LEVELS)
        qvals = z[:, np.newaxis] * std[np.newaxis, :] + mu[np.newaxis, :]
        return _AUTO_ARIMA_QUANTILE_LEVELS.copy(), np.asarray(qvals, dtype=np.float64)
