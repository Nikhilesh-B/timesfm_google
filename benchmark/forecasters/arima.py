"""ARIMAForecaster -- wraps statsmodels ARIMA."""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from benchmark.forecasters.base import Forecaster


class ARIMAForecaster(Forecaster):
    """Fit an ARIMA(p, d, q) model on the history and forecast.

    Parameters:
        order: ``(p, d, q)`` tuple passed to :class:`statsmodels.tsa.arima.model.ARIMA`.
    """

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
