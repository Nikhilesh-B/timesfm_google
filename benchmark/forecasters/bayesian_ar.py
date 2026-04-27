"""BayesianARForecaster -- Gaussian prior on AR coefficients (ridge or Minnesota)."""

from __future__ import annotations

import math

import numpy as np

from benchmark.forecasters.base import Forecaster

_BAYES_QUANTILE_LEVELS = np.linspace(0.1, 0.9, 9, dtype=np.float64)


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


def _build_lag_design(history: np.ndarray, p: int, include_intercept: bool) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(history, dtype=np.float64).ravel()
    n = len(y)
    if n <= p:
        raise ValueError("history too short for AR order")
    m = n - p
    X = np.empty((m, p + int(include_intercept)), dtype=np.float64)
    if include_intercept:
        X[:, 0] = 1.0
        off = 1
    else:
        off = 0
    for j in range(p):
        X[:, off + j] = y[p - 1 + np.arange(m) - j]
    y_tgt = y[p:].copy()
    return X, y_tgt


def _prior_precision_matrix(
    p: int,
    include_intercept: bool,
    prior_mode: str,
    prior_precision: float,
    minnesota_lag_decay_exponent: float,
) -> np.ndarray:
    d = p + int(include_intercept)
    if prior_mode == "ridge":
        return prior_precision * np.eye(d, dtype=np.float64)
    if minnesota_lag_decay_exponent < 0:
        raise ValueError("minnesota_lag_decay_exponent must be non-negative")
    diag = np.zeros(d, dtype=np.float64)
    if include_intercept:
        diag[0] = prior_precision
        lag_slice = slice(1, None)
    else:
        lag_slice = slice(0, None)
    j = np.arange(1, p + 1, dtype=np.float64)
    diag[lag_slice] = prior_precision * (j ** float(minnesota_lag_decay_exponent))
    return np.diag(diag)


def _prior_mean_vector(
    p: int,
    include_intercept: bool,
    prior_mode: str,
    minnesota_center_rw: bool,
) -> np.ndarray:
    d = p + int(include_intercept)
    mu0 = np.zeros(d, dtype=np.float64)
    if prior_mode == "minnesota" and minnesota_center_rw:
        if include_intercept:
            mu0[1] = 1.0
        else:
            mu0[0] = 1.0
    return mu0


class BayesianARForecaster(Forecaster):
    def __init__(
        self,
        p: int = 2,
        prior_precision: float = 1.0,
        include_intercept: bool = False,
        **kwargs: object,
    ) -> None:
        prior_mode = kwargs.pop("prior_mode", "ridge")
        minnesota_lag_decay_exponent = float(
            kwargs.pop("minnesota_lag_decay_exponent", 2.0)
        )
        minnesota_center_rw = bool(kwargs.pop("minnesota_center_rw", True))
        if kwargs:
            bad = ", ".join(sorted(kwargs))
            raise TypeError(
                f"BayesianARForecaster got unexpected keyword arguments: {bad}"
            )
        if p < 1:
            raise ValueError("p must be at least 1")
        if prior_precision < 0:
            raise ValueError("prior_precision must be non-negative")
        if prior_mode not in ("ridge", "minnesota"):
            raise ValueError('prior_mode must be "ridge" or "minnesota"')
        self.p = p
        self.prior_precision = float(prior_precision)
        self.prior_mode = str(prior_mode)
        self.minnesota_lag_decay_exponent = float(minnesota_lag_decay_exponent)
        self.minnesota_center_rw = bool(minnesota_center_rw)
        self.include_intercept = bool(include_intercept)
        self.name = self._make_name()
        self._mu: np.ndarray | None = None
        self._last_y: np.ndarray | None = None
        self._sigma2_hat: float | None = None
        self._Sigma_beta: np.ndarray | None = None

    def _make_name(self) -> str:
        lam = self.prior_precision
        if self.prior_mode == "ridge":
            return f"BayesAR(p={self.p},ridge,λ={lam})"
        rw = "RW" if self.minnesota_center_rw else "0"
        dec = self.minnesota_lag_decay_exponent
        return f"BayesAR(p={self.p},MN,λ={lam},dec={dec},μ={rw})"

    def fit(self, history: np.ndarray) -> None:
        y = np.asarray(history, dtype=np.float64).ravel()
        self._last_y = y.copy()
        self._mu = None
        self._sigma2_hat = None
        self._Sigma_beta = None
        d = self.p + int(self.include_intercept)
        if len(y) <= self.p:
            return
        X, y_tgt = _build_lag_design(y, self.p, self.include_intercept)
        xtx = X.T @ X
        sigma_inv = _prior_precision_matrix(
            self.p,
            self.include_intercept,
            self.prior_mode,
            self.prior_precision,
            self.minnesota_lag_decay_exponent,
        )
        mu0 = _prior_mean_vector(
            self.p,
            self.include_intercept,
            self.prior_mode,
            self.minnesota_center_rw,
        )
        rhs = X.T @ y_tgt + sigma_inv @ mu0
        prec = xtx + sigma_inv
        try:
            self._mu = np.linalg.solve(prec, rhs)
        except np.linalg.LinAlgError:
            return
        resid = y_tgt - X @ self._mu
        denom = max(len(y_tgt) - d, 1)
        self._sigma2_hat = float(max(np.sum(resid**2) / denom, 1e-20))
        try:
            self._Sigma_beta = self._sigma2_hat * np.linalg.inv(prec)
        except np.linalg.LinAlgError:
            self._Sigma_beta = None

    def predict(self, horizon: int) -> np.ndarray:
        if self._mu is None or self._last_y is None:
            return np.full(horizon, np.nan, dtype=np.float64)
        mu = self._mu
        p = self.p
        ext = np.concatenate([self._last_y, np.zeros(horizon, dtype=np.float64)])
        n_hist = len(self._last_y)
        out = np.empty(horizon, dtype=np.float64)
        for h in range(horizon):
            t = n_hist + h
            if self.include_intercept:
                x = np.empty(p + 1, dtype=np.float64)
                x[0] = 1.0
                x[1:] = ext[t - 1 : t - p - 1 : -1][:p]
            else:
                x = ext[t - 1 : t - p - 1 : -1][:p].copy()
            pred = float(x @ mu)
            out[h] = pred
            ext[t] = pred
        return out

    def predict_quantiles(self, horizon: int) -> tuple[np.ndarray, np.ndarray] | None:
        if (
            self._mu is None
            or self._last_y is None
            or self._Sigma_beta is None
            or self._sigma2_hat is None
        ):
            return None
        mu = self._mu
        p = self.p
        s2 = self._sigma2_hat
        Sig = self._Sigma_beta
        z = _std_normal_ppf(_BAYES_QUANTILE_LEVELS)
        ext = np.concatenate([self._last_y, np.zeros(horizon, dtype=np.float64)])
        n_hist = len(self._last_y)
        qvals = np.empty((len(_BAYES_QUANTILE_LEVELS), horizon), dtype=np.float64)
        for h in range(horizon):
            t = n_hist + h
            if self.include_intercept:
                x = np.empty(p + 1, dtype=np.float64)
                x[0] = 1.0
                x[1:] = ext[t - 1 : t - p - 1 : -1][:p]
            else:
                x = ext[t - 1 : t - p - 1 : -1][:p].copy()
            m = float(x @ mu)
            v = s2 + float(x @ Sig @ x)
            std = float(np.sqrt(max(v, 1e-20)))
            qvals[:, h] = m + z * std
            ext[t] = float(x @ mu)
        return _BAYES_QUANTILE_LEVELS.copy(), qvals
