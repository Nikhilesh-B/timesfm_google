"""Probabilistic forecast metrics from quantile grids.

Phase 1: coverage, CRPS, PIT, pinball, calibration from piecewise-linear Q(τ).

Phase 2 (optional SciPy): approximate predictive density by KDE on samples drawn
from the implied inverse-CDF, then mean log score ``mean log f̂(y)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid rule; NumPy 2.0+ uses ``trapezoid``, older versions ``trapz``."""
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def quantile_function_scalar(tau: float, levels: np.ndarray, q: np.ndarray) -> float:
    """Interpolate Q(tau) for scalar tau; linear extrapolation outside ``levels``."""
    levels = np.asarray(levels, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if levels.size == 0 or q.size == 0:
        return float("nan")
    if levels.size != q.size:
        raise ValueError("levels and q must have same length")
    if tau <= levels[0]:
        if levels.size < 2:
            return float(q[0])
        s = (q[1] - q[0]) / (levels[1] - levels[0] + 1e-30)
        return float(q[0] + s * (tau - levels[0]))
    if tau >= levels[-1]:
        if levels.size < 2:
            return float(q[-1])
        s = (q[-1] - q[-2]) / (levels[-1] - levels[-2] + 1e-30)
        return float(q[-1] + s * (tau - levels[-1]))
    return float(np.interp(tau, levels, q))


def pit_from_quantiles(y: float, levels: np.ndarray, q: np.ndarray) -> float:
    """PIT = F_hat(y) with piecewise-linear CDF from (levels, q)."""
    y = float(y)
    q = np.asarray(q, dtype=np.float64).ravel()
    levels = np.asarray(levels, dtype=np.float64).ravel()
    if y <= q[0]:
        if levels.size < 2:
            return float(levels[0])
        s = (levels[1] - levels[0]) / (q[1] - q[0] + 1e-30)
        return float(np.clip(levels[0] + s * (y - q[0]), 0.0, 1.0))
    if y >= q[-1]:
        if levels.size < 2:
            return float(levels[-1])
        s = (levels[-1] - levels[-2]) / (q[-1] - q[-2] + 1e-30)
        return float(np.clip(levels[-1] + s * (y - q[-1]), 0.0, 1.0))
    idx = int(np.searchsorted(q, y, side="right") - 1)
    idx = np.clip(idx, 0, len(q) - 2)
    w = (y - q[idx]) / (q[idx + 1] - q[idx] + 1e-30)
    return float(np.clip(levels[idx] + w * (levels[idx + 1] - levels[idx]), 0.0, 1.0))


def central_interval_hits(
    actual: np.ndarray,
    levels: np.ndarray,
    quantiles: np.ndarray,
    nominal: float,
) -> np.ndarray:
    """Boolean mask same shape as *actual*: y in [Q(α/2), Q(1-α/2)]."""
    alpha = 1.0 - nominal
    lo_tau = alpha / 2.0
    hi_tau = 1.0 - alpha / 2.0
    actual = np.asarray(actual, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim == 2:
        quantiles = quantiles[np.newaxis, :, :]
    n_o, _, h = quantiles.shape
    if actual.shape != (n_o, h):
        raise ValueError(f"actual shape {actual.shape} vs quantiles {quantiles.shape}")
    hits = np.zeros_like(actual, dtype=bool)
    for i in range(n_o):
        for t in range(h):
            qcol = quantiles[i, :, t]
            if np.any(~np.isfinite(qcol)):
                hits[i, t] = False
                continue
            lo = quantile_function_scalar(lo_tau, levels, qcol)
            hi = quantile_function_scalar(hi_tau, levels, qcol)
            if lo > hi:
                lo, hi = hi, lo
            yv = actual[i, t]
            hits[i, t] = np.isfinite(yv) and (lo <= yv <= hi)
    return hits


def empirical_coverage(
    actual: np.ndarray,
    levels: np.ndarray,
    quantiles: np.ndarray,
    nominal: float,
) -> tuple[float, int]:
    """Pooled empirical coverage and count of finite actual cells."""
    hits = central_interval_hits(actual, levels, quantiles, nominal)
    actual = np.asarray(actual, dtype=np.float64)
    valid = np.isfinite(actual)
    hits = hits & valid
    n = int(np.sum(valid))
    if n == 0:
        return float("nan"), 0
    return float(np.sum(hits) / n), n


def crps_from_quantiles_grid(
    y: np.ndarray,
    levels: np.ndarray,
    quantiles: np.ndarray,
    n_grid: int = 99,
) -> float:
    """CRPS ≈ 2 * ∫_0^1 ρ_τ(y - Q(τ)) dτ via trapezoid on a uniform τ grid."""
    y = np.asarray(y, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim != 3:
        raise ValueError("expected quantiles (n_origins, Q, horizon)")
    n_o, q_sz, h = quantiles.shape
    ys = y.reshape(n_o, h) if y.size == n_o * h else y
    total = 0.0
    count = 0
    for i in range(n_o):
        for t in range(h):
            yi = ys[i, t]
            if not np.isfinite(yi):
                continue
            qcol = quantiles[i, :, t]
            if np.any(~np.isfinite(qcol)):
                continue
            taus = np.linspace(0.005, 0.995, n_grid)
            q_interp = np.array(
                [quantile_function_scalar(float(t), levels, qcol) for t in taus]
            )
            rho = np.maximum(taus * (yi - q_interp), (taus - 1.0) * (yi - q_interp))
            total += 2.0 * float(_trapezoid_integral(rho, taus))
            count += 1
    return float(total / max(count, 1))


def mean_interval_width(
    actual: np.ndarray,
    levels: np.ndarray,
    quantiles: np.ndarray,
    nominal: float,
) -> float:
    """Mean width Q(1-α/2) - Q(α/2) over valid cells."""
    alpha = 1.0 - nominal
    lo_tau = alpha / 2.0
    hi_tau = 1.0 - alpha / 2.0
    actual = np.asarray(actual, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim == 2:
        quantiles = quantiles[np.newaxis, :, :]
    n_o, _, h = quantiles.shape
    widths: list[float] = []
    for i in range(n_o):
        for t in range(h):
            qcol = quantiles[i, :, t]
            yv = actual[i, t]
            if not np.isfinite(yv) or np.any(~np.isfinite(qcol)):
                continue
            lo = quantile_function_scalar(lo_tau, levels, qcol)
            hi = quantile_function_scalar(hi_tau, levels, qcol)
            widths.append(abs(hi - lo))
    return float(np.mean(widths)) if widths else float("nan")


def calibration_curve_points(
    actual: np.ndarray,
    levels: np.ndarray,
    quantiles: np.ndarray,
    nominal_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each nominal p, empirical fraction of y with y <= Q(p)."""
    if nominal_grid is None:
        nominal_grid = np.linspace(0.05, 0.95, 19)
    actual = np.asarray(actual, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim == 2:
        quantiles = quantiles[np.newaxis, :, :]
    n_o, _, h = quantiles.shape
    empirical: list[float] = []
    for p in nominal_grid:
        hits: list[float] = []
        for i in range(n_o):
            for t in range(h):
                yv = actual[i, t]
                qcol = quantiles[i, :, t]
                if not np.isfinite(yv) or np.any(~np.isfinite(qcol)):
                    continue
                q_p = quantile_function_scalar(float(p), levels, qcol)
                hits.append(1.0 if yv <= q_p else 0.0)
        empirical.append(float(np.mean(hits)) if hits else float("nan"))
    return nominal_grid, np.array(empirical, dtype=np.float64)


def pinball_table(
    actual: np.ndarray,
    levels: np.ndarray,
    quantiles: np.ndarray,
) -> pd.DataFrame:
    """Mean pinball loss per quantile level."""
    actual = np.asarray(actual, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim == 2:
        quantiles = quantiles[np.newaxis, :, :]
    n_o, q_sz, h = quantiles.shape
    ys = actual.reshape(n_o, h)
    rows = []
    for j, tau in enumerate(levels):
        losses = []
        for i in range(n_o):
            for t in range(h):
                yv = ys[i, t]
                qh = quantiles[i, j, t]
                if not np.isfinite(yv) or not np.isfinite(qh):
                    continue
                e = yv - qh
                losses.append(max(float(tau) * e, (float(tau) - 1.0) * e))
        rows.append({"tau": float(tau), "pinball_mean": float(np.mean(losses)) if losses else float("nan")})
    return pd.DataFrame(rows)


def build_coverage_table(
    forecaster_names: list[str],
    actuals: np.ndarray,
    quantile_predictions: dict[str, np.ndarray] | None,
    quantile_levels: dict[str, np.ndarray] | None,
    nominals: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> pd.DataFrame:
    """Long-format coverage table: one row per (forecaster, nominal interval)."""
    rows: list[dict[str, object]] = []
    if not quantile_predictions or not quantile_levels:
        for name in forecaster_names:
            for nom in nominals:
                rows.append({
                    "Forecaster": name,
                    "Interval": f"{int(round(nom * 100))}% central",
                    "Nominal": nom,
                    "Empirical_coverage": np.nan,
                    "N_valid": 0,
                    "Note": "no quantiles",
                })
        return pd.DataFrame(rows)
    for name in forecaster_names:
        if name not in quantile_predictions or name not in quantile_levels:
            for nom in nominals:
                rows.append({
                    "Forecaster": name,
                    "Interval": f"{int(round(nom * 100))}% central",
                    "Nominal": nom,
                    "Empirical_coverage": np.nan,
                    "N_valid": 0,
                    "Note": "no quantiles",
                })
            continue
        qcube = quantile_predictions[name]
        lv = quantile_levels[name]
        for nom in nominals:
            cov, n_v = empirical_coverage(actuals, lv, qcube, nom)
            rows.append({
                "Forecaster": name,
                "Interval": f"{int(round(nom * 100))}% central",
                "Nominal": nom,
                "Empirical_coverage": cov,
                "N_valid": n_v,
                "Note": "",
            })
    return pd.DataFrame(rows)


def sample_from_quantile_fan(
    levels: np.ndarray,
    qcol: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_samples`` from the distribution with piecewise-linear quantile Q(τ).

    For each uniform ``u ~ U(0,1)``, returns ``Q(u)`` via :func:`quantile_function_scalar`.
    """
    levels = np.asarray(levels, dtype=np.float64).ravel()
    qcol = np.asarray(qcol, dtype=np.float64).ravel()
    if levels.size == 0 or qcol.size == 0 or n_samples <= 0:
        return np.array([], dtype=np.float64)
    u = rng.random(int(n_samples))
    return np.array(
        [quantile_function_scalar(float(ui), levels, qcol) for ui in u],
        dtype=np.float64,
    )


def kde_mean_log_score(
    actual: np.ndarray,
    levels: np.ndarray,
    quantiles: np.ndarray,
    *,
    n_samples: int = 512,
    random_state: int | np.random.Generator | None = 42,
    kde_min_density: float = 1e-30,
    bw_method: str | float = "silverman",
) -> tuple[float, int]:
    """Mean ``log f̂(y)`` where ``f̂`` is a Gaussian KDE on inverse-CDF samples.

    For each evaluation cell ``(origin, horizon)``, draws ``n_samples`` values
    from the quantile fan (piecewise-linear Q), fits :class:`scipy.stats.gaussian_kde`,
    evaluates ``log max(pdf(y), kde_min_density)``, and averages over valid cells.

    Requires **SciPy**. If SciPy is missing or every cell fails, returns
    ``(nan, 0)``.

    Args:
        actual: Shape ``(n_origins, horizon)``.
        levels: Quantile levels ``(Q,)`` in ``(0, 1)``.
        quantiles: ``(n_origins, Q, horizon)`` or ``(Q, horizon)`` (single origin).
        n_samples: Monte Carlo sample size per cell for KDE training.
        random_state: Seed or ``Generator`` for reproducible sampling.
        kde_min_density: Floor on KDE pdf before ``log`` (numerical stability).
        bw_method: Passed to ``gaussian_kde`` (e.g. ``"silverman"``, ``"scott"``).
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return float("nan"), 0

    actual = np.asarray(actual, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64).ravel()
    if quantiles.ndim == 2:
        quantiles = quantiles[np.newaxis, :, :]
    n_o, _, h = quantiles.shape
    ys = actual.reshape(n_o, h) if actual.size == n_o * h else actual
    if ys.shape != (n_o, h):
        raise ValueError(f"actual shape {actual.shape} vs quantiles {quantiles.shape}")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    log_scores: list[float] = []
    for i in range(n_o):
        for t in range(h):
            yi = float(ys[i, t])
            if not np.isfinite(yi):
                continue
            qcol = quantiles[i, :, t]
            if np.any(~np.isfinite(qcol)):
                continue
            samples = sample_from_quantile_fan(levels, qcol, n_samples, rng)
            if samples.size == 0:
                continue
            std = float(np.std(samples))
            if std < 1e-14 * (1.0 + abs(float(np.mean(samples)))):
                samples = samples + rng.normal(0.0, 1e-8, size=samples.shape)
            try:
                kde = gaussian_kde(samples, bw_method=bw_method)
                dens_arr = kde.pdf(yi)
                dens = float(np.asarray(dens_arr).ravel()[0])
            except (np.linalg.LinAlgError, ValueError):
                continue
            if not np.isfinite(dens) or dens < 0.0:
                continue
            log_scores.append(float(np.log(max(dens, kde_min_density))))

    if not log_scores:
        return float("nan"), 0
    return float(np.mean(log_scores)), len(log_scores)


def build_sharpness_calibration_table(
    forecaster_names: list[str],
    actuals: np.ndarray,
    quantile_predictions: dict[str, np.ndarray] | None,
    quantile_levels: dict[str, np.ndarray] | None,
    nominal: float = 0.9,
) -> pd.DataFrame:
    """Per-forecaster sharpness (mean PI width) vs calibration (empirical coverage).

    Uses the same central interval definition as :func:`empirical_coverage`.
    """
    rows: list[dict[str, object]] = []
    if not quantile_predictions or not quantile_levels:
        for name in forecaster_names:
            rows.append({
                "Forecaster": name,
                "Nominal": nominal,
                "Empirical_coverage": np.nan,
                "Mean_PI_width": np.nan,
                "Coverage_error": np.nan,
                "N_valid": 0,
            })
        return pd.DataFrame(rows)

    for name in forecaster_names:
        if name not in quantile_predictions or name not in quantile_levels:
            rows.append({
                "Forecaster": name,
                "Nominal": nominal,
                "Empirical_coverage": np.nan,
                "Mean_PI_width": np.nan,
                "Coverage_error": np.nan,
                "N_valid": 0,
            })
            continue
        qcube = quantile_predictions[name]
        lv = quantile_levels[name]
        cov, n_v = empirical_coverage(actuals, lv, qcube, nominal)
        w = mean_interval_width(actuals, lv, qcube, nominal)
        rows.append({
            "Forecaster": name,
            "Nominal": nominal,
            "Empirical_coverage": cov,
            "Mean_PI_width": w,
            "Coverage_error": float(cov - nominal) if np.isfinite(cov) else np.nan,
            "N_valid": n_v,
        })
    return pd.DataFrame(rows)
    for name in forecaster_names:
        if name not in quantile_predictions or name not in quantile_levels:
            for nom in nominals:
                rows.append({
                    "Forecaster": name,
                    "Interval": f"{int(round(nom * 100))}% central",
                    "Nominal": nom,
                    "Empirical_coverage": np.nan,
                    "N_valid": 0,
                    "Note": "no quantiles",
                })
            continue
        qcube = quantile_predictions[name]
        lv = quantile_levels[name]
        for nom in nominals:
            cov, n_v = empirical_coverage(actuals, lv, qcube, nom)
            rows.append({
                "Forecaster": name,
                "Interval": f"{int(round(nom * 100))}% central",
                "Nominal": nom,
                "Empirical_coverage": cov,
                "N_valid": n_v,
                "Note": "",
            })
    return pd.DataFrame(rows)
