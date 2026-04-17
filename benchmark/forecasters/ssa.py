"""SSAForecaster -- full Singular Spectrum Analysis with closed-form forecasting.

Implements the 7-step algorithm:
  1. Hankel matrix (M = N/2)
  2. SVD
  3. Automatic rank via gap ratio
  4. Truncated signal Hankel
  5. Diagonal averaging → recovered signal f_hat
  6. Signal recursion → characteristic polynomial → closed-form f(t)
  7. AR(p) on residuals (Yule-Walker + BIC)

Forecasting evaluates the closed-form signal function directly at future t
rather than iterating the recurrence, avoiding error accumulation.
"""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.regression.linear_model import yule_walker

from benchmark.forecasters.base import Forecaster


class SSAForecaster(Forecaster):
    """Full SSA forecaster with closed-form signal extrapolation.

    The signal is decomposed into a sum of exponential/oscillatory
    components by solving the characteristic polynomial of the learned
    recurrence.  Forecasts evaluate ``f(t) = Σ c_j λ_j^t`` directly,
    so horizon-h errors do not compound through iterated one-step
    predictions.

    Parameters:
        max_rank: Upper bound on the signal rank ``d`` selected by the
            gap-ratio heuristic.  ``None`` lets the gap ratio decide freely.
        ar_max_order: Maximum AR order considered for the residual noise model.
    """

    name: str = "SSA"

    def __init__(
        self,
        max_rank: int | None = None,
        ar_max_order: int = 15,
    ) -> None:
        self.max_rank = max_rank
        self.ar_max_order = ar_max_order

        self._signal: np.ndarray | None = None
        self._roots: np.ndarray | None = None
        self._coeffs: np.ndarray | None = None
        self._rank: int = 0
        self._history_len: int = 0
        self._ar_phi: np.ndarray | None = None
        self._ar_order: int = 0
        self._residual_tail: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, history: np.ndarray) -> None:
        y = np.asarray(history, dtype=np.float64)
        N = len(y)
        self._history_len = N

        if N < 6:
            self._signal = y.copy()
            self._roots = None
            self._coeffs = None
            self._ar_phi = None
            return

        # ── Step 1: Hankel matrix with M = N // 2 ────────────────────
        M = N // 2
        K = N - M + 1
        H = np.empty((M, K), dtype=np.float64)
        for i in range(M):
            H[i] = y[i : i + K]

        # ── Step 2: SVD ──────────────────────────────────────────────
        U, s, Vt = np.linalg.svd(H, full_matrices=False)

        # ── Step 3: Gap ratio → rank d ───────────────────────────────
        d = self._select_rank(s)
        self._rank = d

        # ── Step 4: Truncated reconstruction ─────────────────────────
        recon = U[:, :d] @ np.diag(s[:d]) @ Vt[:d, :]

        # ── Step 5: Diagonal averaging ───────────────────────────────
        f_hat = np.zeros(N, dtype=np.float64)
        counts = np.zeros(N, dtype=np.float64)
        for i in range(M):
            for j in range(K):
                f_hat[i + j] += recon[i, j]
                counts[i + j] += 1.0
        f_hat /= counts
        self._signal = f_hat

        # ── Step 6: Recursion coefficients via least squares ─────────
        #   f_t = a_1·f_{t-1} + … + a_d·f_{t-d}
        X = np.empty((N - d, d), dtype=np.float64)
        for j in range(d):
            X[:, j] = f_hat[d - j - 1 : N - j - 1]
        a_coef, _, _, _ = np.linalg.lstsq(X, f_hat[d:], rcond=None)

        # ── Solve the recurrence: characteristic polynomial ──────────
        #   z^d − a_1·z^{d−1} − … − a_d = 0
        poly = np.empty(d + 1, dtype=np.float64)
        poly[0] = 1.0
        poly[1:] = -a_coef
        roots = np.roots(poly)

        # Project unstable roots onto the unit circle
        magnitudes = np.abs(roots)
        unstable = magnitudes > 1.0
        if np.any(unstable):
            roots[unstable] = roots[unstable] / magnitudes[unstable]

        self._roots = roots

        # ── Fit constants c_j via overdetermined Vandermonde ─────────
        #   f_hat_t ≈ Σ_j c_j · λ_j^t   for t = 1, …, N
        t_idx = np.arange(1, N + 1)
        V = np.column_stack([
            roots[j] ** t_idx for j in range(d)
        ]).astype(np.complex128)
        c, _, _, _ = np.linalg.lstsq(
            V, f_hat.astype(np.complex128), rcond=None
        )
        self._coeffs = c

        # ── Step 7: AR(p) on residuals via Yule-Walker + BIC ────────
        residuals = y - f_hat
        self._fit_ar_residuals(residuals)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, horizon: int) -> np.ndarray:
        if self._signal is None:
            raise RuntimeError("Must call fit() before predict()")

        N = self._history_len

        # Signal forecast via closed form: f(t) = Σ c_j · λ_j^t
        signal_fc = np.zeros(horizon, dtype=np.float64)
        if self._roots is not None and self._coeffs is not None:
            for h in range(horizon):
                t = N + h + 1
                val = np.sum(self._coeffs * self._roots ** t)
                signal_fc[h] = val.real

        # Noise forecast via AR model
        noise_fc = self._forecast_ar(horizon)

        return signal_fc + noise_fc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_rank(self, s: np.ndarray) -> int:
        """Gap-ratio rank selection: d = argmax_k  σ_k / σ_{k+1}."""
        max_k = len(s) - 1
        if self.max_rank is not None:
            max_k = min(max_k, self.max_rank)

        best_idx = 0
        best_ratio = 0.0
        for k in range(max_k):
            if s[k + 1] < 1e-12:
                best_idx = k
                break
            ratio = s[k] / s[k + 1]
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = k
        return best_idx + 1  # rank = gap position + 1 (1-indexed)

    def _fit_ar_residuals(self, residuals: np.ndarray) -> None:
        """Select AR order by BIC (Yule-Walker) and store coefficients."""
        n = len(residuals)
        max_p = min(self.ar_max_order, n // 4)
        if max_p < 1:
            self._ar_phi = None
            self._ar_order = 0
            self._residual_tail = residuals[-1:]
            return

        best_bic = np.inf
        best_p = 1
        best_phi: np.ndarray | None = None

        for p in range(1, max_p + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    phi, sigma = yule_walker(residuals, order=p, method="mle")
                sig2 = float(sigma) ** 2 if float(sigma) > 0 else 1e-12
                bic = n * np.log(sig2) + p * np.log(n)
                if bic < best_bic:
                    best_bic = bic
                    best_p = p
                    best_phi = np.asarray(phi, dtype=np.float64)
            except Exception:
                continue

        self._ar_phi = best_phi
        self._ar_order = best_p
        self._residual_tail = residuals[-best_p:].copy() if best_phi is not None else residuals[-1:]

    def _forecast_ar(self, horizon: int) -> np.ndarray:
        """Iterate the fitted AR(p) forward for *horizon* steps."""
        if self._ar_phi is None or self._residual_tail is None:
            return np.zeros(horizon, dtype=np.float64)

        p = self._ar_order
        phi = self._ar_phi
        buf = list(self._residual_tail)
        fc = np.empty(horizon, dtype=np.float64)
        for h in range(horizon):
            recent = np.array(buf[-p:], dtype=np.float64)[::-1]
            val = float(np.dot(phi, recent))
            fc[h] = val
            buf.append(val)
        return fc
