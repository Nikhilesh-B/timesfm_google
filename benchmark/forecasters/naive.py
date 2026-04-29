"""NaiveBenchmarkForecaster -- rule-based naive baseline.

Rule: fit a preliminary AR(1) on history.
  |rho| > rho_threshold  =>  persistence (last observation for all horizons)
  |rho| <= rho_threshold =>  rolling mean

Quantile intervals use a Gaussian approximation:
  - Persistence: y_T ± z_p * sigma_innov * sqrt(h)  (random-walk error grows with h)
  - Mean:        mu  ± z_p * sigma_proc              (bounded stationary variance)

where sigma_innov is the std of AR(1) one-step residuals and
sigma_proc = sigma_innov / sqrt(max(1 - rho^2, 1e-6)).
"""

from __future__ import annotations

import math

import numpy as np

from benchmark.forecasters.base import Forecaster

_NAIVE_QUANTILE_LEVELS = np.linspace(0.1, 0.9, 9, dtype=np.float64)


def _normal_ppf(levels: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import norm
        return np.asarray(norm.ppf(np.clip(levels, 1e-12, 1 - 1e-12)), dtype=np.float64)
    except Exception:
        out = np.empty(len(levels), dtype=np.float64)
        for i, p in enumerate(levels):
            p = float(np.clip(p, 1e-12, 1 - 1e-12))
            out[i] = math.sqrt(2.0) * math.erfinv(2.0 * p - 1.0)
        return out


class NaiveBenchmarkForecaster(Forecaster):
    """Rule-based naive benchmark.

    Parameters
    ----------
    rho_threshold:
        If |rho_AR1| > this, use persistence; otherwise use rolling mean.
    mean_window:
        Trailing observations to average in mean mode (None = all history).
    """

    name: str = "Naive"

    def __init__(
        self,
        rho_threshold: float = 0.7,
        mean_window: int | None = None,
    ) -> None:
        self.rho_threshold = rho_threshold
        self.mean_window = mean_window
        self._mode: str = "mean"   # 'persist' or 'mean'
        self._rho: float = 0.0
        self._forecast_val: float = 0.0
        self._sigma_innov: float = 1.0
        self._sigma_proc: float = 1.0

    # ------------------------------------------------------------------

    def fit(self, history: np.ndarray) -> None:
        y = np.asarray(history, dtype=np.float64)

        # ── AR(1) OLS estimate ──────────────────────────────────────────
        if len(y) >= 3:
            mu = y.mean()
            y_dm = y - mu
            num = float(np.dot(y_dm[1:], y_dm[:-1]))
            den = float(np.dot(y_dm[:-1], y_dm[:-1]))
            rho = num / den if den > 1e-12 else 0.0
            rho = float(np.clip(rho, -0.9999, 0.9999))
            # Innovation residuals: e_t = y_t - rho*y_{t-1} - c, c = mu*(1-rho)
            c = mu * (1.0 - rho)
            resid = y[1:] - rho * y[:-1] - c
            sigma_innov = float(np.std(resid, ddof=1)) if len(resid) > 1 else float(np.std(y))
        else:
            rho = 0.0
            sigma_innov = float(np.std(y)) if len(y) > 1 else 1.0

        sigma_innov = max(sigma_innov, 1e-12)
        sigma_proc = sigma_innov / math.sqrt(max(1.0 - rho ** 2, 1e-6))

        self._rho = rho
        self._sigma_innov = sigma_innov
        self._sigma_proc = sigma_proc

        # ── Choose mode ─────────────────────────────────────────────────
        self._mode = "persist" if abs(rho) > self.rho_threshold else "mean"

        if self._mode == "persist":
            self._forecast_val = float(y[-1])
        else:
            window = y if self.mean_window is None else y[-self.mean_window:]
            self._forecast_val = float(np.mean(window))

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._forecast_val, dtype=np.float64)

    def predict_quantiles(
        self, horizon: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        z = _normal_ppf(_NAIVE_QUANTILE_LEVELS)   # shape (Q,)
        if self._mode == "persist":
            # Variance grows linearly: sigma^2 * h  (random walk)
            h_vals = np.sqrt(np.arange(1, horizon + 1, dtype=np.float64))
            qvals = (
                z[:, np.newaxis] * self._sigma_innov * h_vals[np.newaxis, :]
                + self._forecast_val
            )
        else:
            # Stationary process: constant marginal variance for all horizons
            qvals = (
                z[:, np.newaxis] * self._sigma_proc
                + self._forecast_val
            ) * np.ones((1, horizon), dtype=np.float64)
        return _NAIVE_QUANTILE_LEVELS.copy(), qvals.astype(np.float64)

    # convenience for inspection
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def rho(self) -> float:
        return self._rho
