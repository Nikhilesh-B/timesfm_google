"""MLBayesARForecaster -- Bayesian AR with marginal-likelihood-optimal prior tightness.

The prior precision scale λ is chosen by maximising the profile marginal
likelihood (σ² integrated out analytically).

Model
-----
    y | β, σ² ~ N(Xβ, σ²Iₙ)
    β | σ², λ ~ N(μ₀, (σ²/λ) M⁻¹)

where M is the normalised prior structure matrix:
  Ridge     : M = I_d
  Minnesota : M = diag(1^θ, 2^θ, ..., p^θ)   (intercept row = 1 if present)

Integrating out β yields the profile log marginal likelihood (further
optimised over σ²):

    log p(y|λ) = ½ log|λM| - ½ log|P| - (n/2) log Q  + const

where
    P  = λM + XᵀX          (d×d posterior precision)
    μₙ = P⁻¹(Xᵀy + λM μ₀) (posterior mean)
    Q  = yᵀy + μ₀ᵀ(λM)μ₀ - μₙᵀ P μₙ  (≈ n × σ̂²_MLE)

Optimisation is done in log λ space via scipy.optimize.minimize_scalar
(Brent's method, ~15–25 evaluations per fit()).

After finding λ*, the posterior mean and predictive variance are computed
using the same formulae as BayesianARForecaster so that predict() and
predict_quantiles() are identical.

References
----------
Bishop (2006) §3.5; Kadiyala & Karlsson (1997) for Minnesota prior.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from benchmark.forecasters.base import Forecaster
from benchmark.forecasters.bayesian_ar import (
    _build_lag_design,
    _prior_mean_vector,
    _prior_precision_matrix,
    _std_normal_ppf,
)

_ML_QUANTILE_LEVELS = np.linspace(0.1, 0.9, 9, dtype=np.float64)


# ── Marginal likelihood helpers ────────────────────────────────────────────

def _structure_matrix(
    p: int,
    include_intercept: bool,
    prior_mode: str,
    decay: float,
) -> np.ndarray:
    """Unit-tightness prior precision matrix (λ=1).

    Ridge     → I_d
    Minnesota → diag(j^decay, j=1..p) [+ 1 for intercept row]
    """
    return _prior_precision_matrix(
        p, include_intercept, prior_mode, 1.0, decay
    )


def _profile_log_marglik(
    log_lambda: float,
    X: np.ndarray,
    y: np.ndarray,
    M: np.ndarray,
    mu0: np.ndarray,
) -> float:
    """Profile log marginal likelihood at log(λ).

    Returns -∞ on any numerical failure (treated as infeasible by optimiser).
    """
    lam = math.exp(log_lambda)
    n = X.shape[0]
    lam_M = lam * M
    P = lam_M + X.T @ X

    try:
        L = np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        return -math.inf

    # Posterior mean: P μₙ = Xᵀy + λM μ₀
    rhs = X.T @ y + lam_M @ mu0
    # Solve via Cholesky: P μₙ = rhs  →  L Lᵀ μₙ = rhs
    mu_n = np.linalg.solve(P, rhs)

    # Q = yᵀy + μ₀ᵀ (λM) μ₀ - μₙᵀ P μₙ
    Q = (float(np.dot(y, y))
         + float(mu0 @ lam_M @ mu0)
         - float(mu_n @ P @ mu_n))
    if Q <= 0.0:
        Q = 1e-20

    # log|λM| = Σ log(λM_ii)  [M is diagonal]
    log_det_lam_M = float(np.sum(np.log(np.maximum(np.diag(lam_M), 1e-300))))

    # log|P| = 2 Σ log diag(L)
    log_det_P = 2.0 * float(np.sum(np.log(np.diag(L))))

    return 0.5 * log_det_lam_M - 0.5 * log_det_P - 0.5 * n * math.log(Q)


def _optimise_lambda(
    X: np.ndarray,
    y: np.ndarray,
    M: np.ndarray,
    mu0: np.ndarray,
    lambda_min: float,
    lambda_max: float,
) -> float:
    """Return λ* maximising the profile marginal likelihood."""
    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(
            fun=lambda ll: -_profile_log_marglik(ll, X, y, M, mu0),
            bounds=(math.log(lambda_min), math.log(lambda_max)),
            method="bounded",
        )
        return math.exp(result.x)
    except Exception:
        # Fallback: 20-point log-grid search
        log_grid = np.linspace(math.log(lambda_min), math.log(lambda_max), 20)
        scores = [_profile_log_marglik(ll, X, y, M, mu0) for ll in log_grid]
        return math.exp(log_grid[int(np.argmax(scores))])


# ── Forecaster ─────────────────────────────────────────────────────────────

class MLBayesARForecaster(Forecaster):
    """Bayesian AR where λ (prior tightness) is chosen by marginal likelihood.

    Parameters
    ----------
    p:
        AR lag order.  Default 24 (one year of monthly lags) as recommended
        for macroeconomic series; set equal to the forecast horizon.
    prior_mode:
        ``'ridge'``       -- isotropic prior  N(0, (σ²/λ)I)
        ``'minnesota'``   -- lag-decay prior  N(μ₀, (σ²/λ)M⁻¹)
    minnesota_lag_decay_exponent:
        θ ≥ 0; prior precision on lag j is j^θ.  Larger → more shrinkage on
        longer lags.  Recommended: 1.0 (linear decay).
    minnesota_center_rw:
        If True, centre the prior on the unit-root (φ₁=1, φⱼ=0 for j>1).
        Appropriate for near-integrated series (interest rates, unemployment).
    lambda_min / lambda_max:
        Search bounds for λ (in natural scale).
    include_intercept:
        Whether to include an intercept term.
    """

    name: str = "MLBayesAR"  # overwritten in __init__

    def __init__(
        self,
        p: int = 24,
        prior_mode: str = "minnesota",
        minnesota_lag_decay_exponent: float = 1.0,
        minnesota_center_rw: bool = True,
        lambda_min: float = 1e-4,
        lambda_max: float = 1e4,
        include_intercept: bool = False,
    ) -> None:
        if p < 1:
            raise ValueError("p must be >= 1")
        if prior_mode not in ("ridge", "minnesota"):
            raise ValueError('prior_mode must be "ridge" or "minnesota"')
        self.p = p
        self.prior_mode = prior_mode
        self.minnesota_lag_decay_exponent = float(minnesota_lag_decay_exponent)
        self.minnesota_center_rw = bool(minnesota_center_rw)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.include_intercept = bool(include_intercept)
        self.name = self._make_name()

        self.best_lambda_: float | None = None
        self._mu: np.ndarray | None = None
        self._last_y: np.ndarray | None = None
        self._sigma2_hat: float | None = None
        self._Sigma_beta: np.ndarray | None = None

    def _make_name(self) -> str:
        if self.prior_mode == "ridge":
            return f"MLBayesAR(p={self.p},ridge)"
        rw = "RW" if self.minnesota_center_rw else "0"
        dec = self.minnesota_lag_decay_exponent
        return f"MLBayesAR(p={self.p},MN,dec={dec},μ={rw})"

    # ------------------------------------------------------------------

    def fit(self, history: np.ndarray) -> None:
        y = np.asarray(history, dtype=np.float64).ravel()
        self._last_y = y.copy()
        self._mu = None
        self._sigma2_hat = None
        self._Sigma_beta = None
        self.best_lambda_ = None

        if len(y) <= self.p:
            return

        X, y_tgt = _build_lag_design(y, self.p, self.include_intercept)
        d = X.shape[1]

        # Unit-tightness structure matrix M (λ factored out)
        M = _structure_matrix(
            self.p, self.include_intercept, self.prior_mode,
            self.minnesota_lag_decay_exponent,
        )
        mu0 = _prior_mean_vector(
            self.p, self.include_intercept, self.prior_mode,
            self.minnesota_center_rw,
        )

        # ── Step 1: optimise λ via profile marginal likelihood ────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lam_opt = _optimise_lambda(X, y_tgt, M, mu0,
                                       self.lambda_min, self.lambda_max)
        self.best_lambda_ = lam_opt

        # ── Step 2: compute posterior with optimal λ ───────────────────────
        sigma_inv = lam_opt * M           # = _prior_precision_matrix(…, lam_opt, …)
        rhs = X.T @ y_tgt + sigma_inv @ mu0
        prec = X.T @ X + sigma_inv
        try:
            self._mu = np.linalg.solve(prec, rhs)
        except np.linalg.LinAlgError:
            return

        resid = y_tgt - X @ self._mu
        denom = max(len(y_tgt) - d, 1)
        self._sigma2_hat = float(max(np.sum(resid ** 2) / denom, 1e-20))
        try:
            self._Sigma_beta = self._sigma2_hat * np.linalg.inv(prec)
        except np.linalg.LinAlgError:
            self._Sigma_beta = None

    # ------------------------------------------------------------------
    # predict / predict_quantiles — identical logic to BayesianARForecaster
    # ------------------------------------------------------------------

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

    def predict_quantiles(
        self, horizon: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
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
        z = _std_normal_ppf(_ML_QUANTILE_LEVELS)
        ext = np.concatenate([self._last_y, np.zeros(horizon, dtype=np.float64)])
        n_hist = len(self._last_y)
        qvals = np.empty((len(_ML_QUANTILE_LEVELS), horizon), dtype=np.float64)
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
            ext[t] = m
        return _ML_QUANTILE_LEVELS.copy(), qvals
