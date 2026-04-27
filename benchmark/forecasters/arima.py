"""ARIMAForecaster -- wraps statsmodels ARIMA."""

from __future__ import annotations

import math
import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from benchmark.forecasters.base import Forecaster

_ARIMA_QUANTILE_LEVELS = np.linspace(0.1, 0.9, 9, dtype=np.float64)


def _std_normal_ppf(levels: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import norm
        return np.asarray(norm.ppf(np.clip(levels, 1e-12, 1.0 - 1e-12)), dtype=np.float64)
    except Exception:
        out = np.empty(levels.shape, dtype=np.float64)
        for i, p in enumerate(np.asarray(levels, dtype=np.float64).ravel()):
            p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
            out.ravel()[i] = math.sqrt(2.0) * math.erfinv(2.0 * p - 1.0)
        return out


class ARIMAForecaster(Forecaster):
    def __init__(self, order: tuple[int, int, int] = (2, 0, 0)) -> None:
        self.order = order
        self.name = f"ARIMA{order}"
        self._fit_result = None

    def fit(self, history: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit_result = ARIMA(
                history.astype(np.float64), order=self.order
            ).fit()

    def predict(self, horizon: int) -> np.ndarray:
        if self._fit_result is None:
            raise RuntimeError("Must call fit() before predict()")
        fc = self._fit_result.forecast(steps=horizon)
        return np.asarray(fc, dtype=np.float64).ravel()

    def predict_quantiles(self, horizon: int) -> tuple[np.ndarray, np.ndarray] | None:
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
        z = _std_normal_ppf(_ARIMA_QUANTILE_LEVELS)
        qvals = z[:, np.newaxis] * std[np.newaxis, :] + mu[np.newaxis, :]
        return _ARIMA_QUANTILE_LEVELS.copy(), np.asarray(qvals, dtype=np.float64)
