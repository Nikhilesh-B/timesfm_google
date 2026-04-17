"""MeanForecaster -- the dumb benchmark (rolling mean)."""

from __future__ import annotations

import numpy as np

from benchmark.forecasters.base import Forecaster


class MeanForecaster(Forecaster):
    """Predicts the mean of the last *window* observations for every horizon step.

    Parameters:
        window: Number of trailing observations to average.
                ``None`` means use the entire history.
    """

    name: str = "Mean"

    def __init__(self, window: int | None = None) -> None:
        self.window = window
        self._mean: float = 0.0

    def fit(self, history: np.ndarray) -> None:
        h = history if self.window is None else history[-self.window :]
        self._mean = float(np.mean(h))

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._mean, dtype=np.float64)
