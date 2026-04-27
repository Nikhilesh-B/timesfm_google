"""Abstract base class for all forecasters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """A forecaster that can be fitted on a history window and produce predictions.

    Every concrete subclass must implement :meth:`fit` and :meth:`predict`.
    Config / hyper-parameters live as instance attributes on the subclass so
    each model is fully self-contained.
    """

    name: str

    @abstractmethod
    def fit(self, history: np.ndarray) -> None:
        """Fit (or re-fit) the model on *history* (1-D float64 array)."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return a ``(horizon,)`` array of point forecasts.

        Must be called after :meth:`fit`.
        """

    def fit_predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Convenience: fit then predict in one call."""
        self.fit(history)
        return self.predict(horizon)
