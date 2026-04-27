"""BenchmarkResults -- stores predictions/actuals and computes comparisons."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.metrics import METRIC_REGISTRY, diebold_mariano_hac
from . import probabilistic_metrics as pm


class BenchmarkResults:
    """Container for the output of a benchmark run.

    Attributes:
        series_name: Name of the :class:`TimeSeries` that was benchmarked.
        horizon: Forecast horizon used.
        k_first: Initial training-window size.
        forecaster_names: Ordered list of forecaster names.
        predictions: ``{forecaster_name: np.ndarray}`` of shape ``(n_origins, horizon)``.
        actuals: ``np.ndarray`` of shape ``(n_origins, horizon)``.
        quantile_predictions: Optional per-model tensor ``(n_origins, Q, horizon)``.
        quantile_levels: Optional per-model ``(Q,)`` probability levels in (0,1).
    """

    def __init__(
        self,
        series_name: str,
        horizon: int,
        k_first: int,
        forecaster_names: list[str],
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
        quantile_predictions: dict[str, np.ndarray] | None = None,
        quantile_levels: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.series_name = series_name
        self.horizon = horizon
        self.k_first = k_first
        self.forecaster_names = forecaster_names
        self.predictions = predictions
        self.actuals = actuals
        self.n_origins = actuals.shape[0]
        self.quantile_predictions = quantile_predictions
        self.quantile_levels = quantile_levels

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def metrics(self, metric: str = "mse") -> pd.DataFrame:
        """Compute a single aggregate metric per forecaster.

        Returns a DataFrame with columns ``["Forecaster", "<metric>"]``.
        """
        fn = METRIC_REGISTRY.get(metric)
        if fn is None:
            raise ValueError(
                f"Unknown metric {metric!r}. Available: {list(METRIC_REGISTRY)}"
            )
        rows = []
        for name in self.forecaster_names:
            val = fn(self.actuals.ravel(), self.predictions[name].ravel())
            rows.append({"Forecaster": name, metric.upper(): val})
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """One-row-per-model summary with MSE, MAE, RMSE."""
        frames = [self.metrics(m) for m in METRIC_REGISTRY]
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on="Forecaster")
        return out

    # ------------------------------------------------------------------
    # Diebold-Mariano pairwise tests
    # ------------------------------------------------------------------

    def diebold_mariano(self, loss: str = "se") -> pd.DataFrame:
        """Pairwise Diebold-Mariano test (squared-error loss by default).

        Returns a DataFrame with columns
        ``["Model A", "Model B", "DM t-stat", "DM p-value"]``.
        """
        if loss == "se":
            losses = {
                n: (self.actuals.ravel() - self.predictions[n].ravel()) ** 2
                for n in self.forecaster_names
            }
        elif loss == "ae":
            losses = {
                n: np.abs(self.actuals.ravel() - self.predictions[n].ravel())
                for n in self.forecaster_names
            }
        else:
            raise ValueError(f"Unknown loss {loss!r}; use 'se' or 'ae'")

        rows = []
        for a, b in combinations(self.forecaster_names, 2):
            t_stat, p_val = diebold_mariano_hac(losses[a], losses[b])
            rows.append(
                {
                    "Model A": a,
                    "Model B": b,
                    "DM t-stat": t_stat,
                    "DM p-value": p_val,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Probabilistic evaluation (Phase 1: quantile grids)
    # ------------------------------------------------------------------

    def coverage_table(
        self,
        nominals: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    ) -> pd.DataFrame:
        """Empirical central-interval coverage per forecaster (own quantiles)."""
        return pm.build_coverage_table(
            self.forecaster_names,
            self.actuals,
            self.quantile_predictions,
            self.quantile_levels,
            nominals=nominals,
        )

    def probabilistic_summary(self) -> pd.DataFrame:
        """CRPS, pinball summary, sharpness (90\% PI width) for models with quantiles."""
        rows: list[dict[str, object]] = []
        if not self.quantile_predictions or not self.quantile_levels:
            return pd.DataFrame(
                columns=["Forecaster", "CRPS", "mean_width_90", "pinball_mean_sum"]
            )
        for name in self.forecaster_names:
            if name not in self.quantile_predictions or name not in self.quantile_levels:
                continue
            lv = self.quantile_levels[name]
            qc = self.quantile_predictions[name]
            crps = pm.crps_from_quantiles_grid(self.actuals, lv, qc)
            w90 = pm.mean_interval_width(self.actuals, lv, qc, 0.9)
            ptab = pm.pinball_table(self.actuals, lv, qc)
            pin_sum = float(np.nansum(ptab["pinball_mean"].values))
            rows.append({
                "Forecaster": name,
                "CRPS": crps,
                "mean_width_90": w90,
                "pinball_mean_sum": pin_sum,
            })
        return pd.DataFrame(rows)

    def probabilistic_summary_phase2(
        self,
        nominal: float = 0.9,
        n_samples: int = 512,
        random_state: int | np.random.Generator | None = 42,
    ) -> pd.DataFrame:
        """KDE log score (SciPy) and sharpness vs nominal coverage for one nominal.

        Columns include ``mean_log_score_kde``, ``n_log_score_cells``, and
        sharpness–calibration fields from :func:`build_sharpness_calibration_table`.
        If SciPy is unavailable, ``mean_log_score_kde`` is NaN and
        ``n_log_score_cells`` is 0.
        """
        sharp = pm.build_sharpness_calibration_table(
            self.forecaster_names,
            self.actuals,
            self.quantile_predictions,
            self.quantile_levels,
            nominal=nominal,
        )
        if not self.quantile_predictions or not self.quantile_levels:
            sharp = sharp.copy()
            sharp["mean_log_score_kde"] = np.nan
            sharp["n_log_score_cells"] = 0
            return sharp

        log_cols: list[float] = []
        n_log: list[int] = []
        for name in self.forecaster_names:
            if name not in self.quantile_predictions or name not in self.quantile_levels:
                log_cols.append(float("nan"))
                n_log.append(0)
                continue
            lv = self.quantile_levels[name]
            qc = self.quantile_predictions[name]
            mls, n_c = pm.kde_mean_log_score(
                self.actuals,
                lv,
                qc,
                n_samples=n_samples,
                random_state=random_state,
            )
            log_cols.append(mls)
            n_log.append(n_c)
        out = sharp.copy()
        out["mean_log_score_kde"] = log_cols
        out["n_log_score_cells"] = n_log
        return out

    def plot_sharpness_vs_calibration(
        self,
        nominal: float = 0.9,
        ax=None,
    ):
        """Scatter: mean PI width (sharpness) vs empirical coverage (calibration)."""
        import matplotlib.pyplot as plt

        tab = pm.build_sharpness_calibration_table(
            self.forecaster_names,
            self.actuals,
            self.quantile_predictions,
            self.quantile_levels,
            nominal=nominal,
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
        for _, row in tab.iterrows():
            name = row["Forecaster"]
            w = row["Mean_PI_width"]
            c = row["Empirical_coverage"]
            if not (np.isfinite(w) and np.isfinite(c)):
                continue
            ax.scatter([w], [c], s=80, alpha=0.85, label=name)
        ax.axhline(nominal, color="k", linestyle="--", alpha=0.45, label=f"nominal {nominal:g}")
        ax.set_xlabel("Mean PI width (sharpness)")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(
            f"Sharpness vs calibration ({int(round(nominal * 100))}% PI) — {self.series_name}"
        )
        ax.legend(loc="best", fontsize=8)
        return ax

    def plot_probabilistic_calibration(
        self, forecaster_name: str, ax=None, nominal_grid: np.ndarray | None = None
    ):
        """Reliability: nominal p vs empirical P(y <= Q(p))."""
        import matplotlib.pyplot as plt

        if (
            not self.quantile_predictions
            or forecaster_name not in self.quantile_predictions
        ):
            return None
        lv = self.quantile_levels[forecaster_name]
        qc = self.quantile_predictions[forecaster_name]
        x, y = pm.calibration_curve_points(self.actuals, lv, qc, nominal_grid)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
        ax.plot(x, y, "o-", label=forecaster_name)
        ax.set_xlabel("Nominal quantile")
        ax.set_ylabel("Empirical P(y <= Q(p))")
        ax.set_title(f"Calibration — {forecaster_name} — {self.series_name}")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        return ax

    def plot_pit_histogram(self, forecaster_name: str, ax=None, bins: int = 15):
        """Histogram of PIT values (piecewise-linear CDF from quantiles)."""
        import matplotlib.pyplot as plt

        if (
            not self.quantile_predictions
            or forecaster_name not in self.quantile_predictions
        ):
            return None
        lv = self.quantile_levels[forecaster_name]
        qc = self.quantile_predictions[forecaster_name]
        pits = []
        n_o, q_sz, h = qc.shape
        for i in range(n_o):
            for t in range(h):
                yv = self.actuals[i, t]
                qcol = qc[i, :, t]
                if not np.isfinite(yv) or np.any(~np.isfinite(qcol)):
                    continue
                pits.append(pm.pit_from_quantiles(float(yv), lv, qcol))
        if not pits:
            return None
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pits, bins=bins, range=(0, 1), density=True, alpha=0.75)
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title(f"PIT — {forecaster_name} — {self.series_name}")
        return ax

    # ------------------------------------------------------------------
    # Per-origin error series (for plotting)
    # ------------------------------------------------------------------

    def error_series(self, metric: str = "mse") -> pd.DataFrame:
        """Per-origin metric value for each forecaster.

        Returns a DataFrame indexed by origin with one column per forecaster.
        """
        fn = METRIC_REGISTRY.get(metric)
        if fn is None:
            raise ValueError(f"Unknown metric {metric!r}")
        data: dict[str, list[float]] = {}
        for name in self.forecaster_names:
            vals = []
            for i in range(self.n_origins):
                vals.append(fn(self.actuals[i], self.predictions[name][i]))
            data[name] = vals
        return pd.DataFrame(data, index=range(self.k_first, self.k_first + self.n_origins))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_mse_over_time(self, rolling_window: int = 20, ax=None):
        """Rolling MSE by forecast origin for each model."""
        import matplotlib.pyplot as plt

        errors = self.error_series("mse")
        rolled = errors.rolling(rolling_window, min_periods=1).mean()

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))
        for col in rolled.columns:
            ax.plot(rolled.index, rolled[col], label=col)
        ax.set_xlabel("Forecast origin (k)")
        ax.set_ylabel(f"Rolling MSE (window={rolling_window})")
        ax.set_title(f"Rolling MSE — {self.series_name}")
        ax.legend()
        return ax

    def plot_cumulative_error(self, ax=None):
        """Cumulative squared error curves."""
        import matplotlib.pyplot as plt

        errors = self.error_series("mse")
        cum = errors.cumsum()

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))
        for col in cum.columns:
            ax.plot(cum.index, cum[col], label=col)
        ax.set_xlabel("Forecast origin (k)")
        ax.set_ylabel("Cumulative Squared Error")
        ax.set_title(f"Cumulative Error — {self.series_name}")
        ax.legend()
        return ax

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Export the summary table to CSV."""
        self.summary().to_csv(path, index=False)


class ReplicatedBenchmarkResults:
    """Aggregated results from running the same benchmark across multiple seeds.

    Each element of *per_seed_results* corresponds to one random seed
    (i.e. one realisation of the DGP).

    Attributes:
        dgp_name: Label for the data-generating process (e.g. ``"AR(2)"``).
        seeds: The random seeds used.
        per_seed: List of :class:`BenchmarkResults`, one per seed.
        forecaster_names: Forecaster names (same across all seeds).
    """

    def __init__(
        self,
        dgp_name: str,
        seeds: list[int],
        per_seed: list[BenchmarkResults],
    ) -> None:
        if not per_seed:
            raise ValueError("Need at least one per-seed result")
        self.dgp_name = dgp_name
        self.seeds = seeds
        self.per_seed = per_seed
        self.forecaster_names = per_seed[0].forecaster_names
        self.n_seeds = len(seeds)

    def aggregate_metrics(self) -> pd.DataFrame:
        """Mean and std of each metric across seeds.

        Returns a DataFrame with columns
        ``["Forecaster", "<METRIC>_mean", "<METRIC>_std", ...]``
        for every metric in the registry.
        """
        from benchmark.metrics import METRIC_REGISTRY

        per_seed_summaries = [r.summary() for r in self.per_seed]

        rows = []
        for name in self.forecaster_names:
            row: dict[str, object] = {"Forecaster": name}
            for metric_key in METRIC_REGISTRY:
                col = metric_key.upper()
                vals = np.array([
                    float(s.loc[s["Forecaster"] == name, col].iloc[0])
                    for s in per_seed_summaries
                ])
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append(row)
        return pd.DataFrame(rows)

    def pooled_diebold_mariano(self, loss: str = "se") -> pd.DataFrame:
        """Diebold-Mariano test on pooled (concatenated) errors across all seeds.

        This gives a single test statistic with much higher power than any
        individual seed.
        """
        pooled_actuals = np.concatenate(
            [r.actuals.ravel() for r in self.per_seed]
        )
        pooled_preds: dict[str, np.ndarray] = {}
        for name in self.forecaster_names:
            pooled_preds[name] = np.concatenate(
                [r.predictions[name].ravel() for r in self.per_seed]
            )

        if loss == "se":
            losses = {
                n: (pooled_actuals - pooled_preds[n]) ** 2
                for n in self.forecaster_names
            }
        elif loss == "ae":
            losses = {
                n: np.abs(pooled_actuals - pooled_preds[n])
                for n in self.forecaster_names
            }
        else:
            raise ValueError(f"Unknown loss {loss!r}; use 'se' or 'ae'")

        rows = []
        for a, b in combinations(self.forecaster_names, 2):
            t_stat, p_val = diebold_mariano_hac(losses[a], losses[b])
            rows.append({
                "Model A": a,
                "Model B": b,
                "DM t-stat": t_stat,
                "DM p-value": p_val,
            })
        return pd.DataFrame(rows)

    def per_seed_metric(self, metric: str = "mse") -> pd.DataFrame:
        """Return the metric value per seed per forecaster.

        Returns a DataFrame with columns ``["seed", "Forecaster", "<METRIC>"]``.
        """
        from benchmark.metrics import METRIC_REGISTRY

        fn = METRIC_REGISTRY.get(metric)
        if fn is None:
            raise ValueError(
                f"Unknown metric {metric!r}. Available: {list(METRIC_REGISTRY)}"
            )
        rows = []
        for seed, res in zip(self.seeds, self.per_seed):
            for name in self.forecaster_names:
                val = fn(res.actuals.ravel(), res.predictions[name].ravel())
                rows.append({"seed": seed, "Forecaster": name, metric.upper(): val})
        return pd.DataFrame(rows)

    def replication_scorecard(self) -> pd.DataFrame:
        """Wide scorecard: per-seed MSE/MAE, pooled metrics, and ratios vs best.

        For each forecaster, reports one **MSE** and **MAE** per replication
        seed (same order as :attr:`seeds`), then the **mean and std** of those
        per-seed values, then **pooled** MSE/MAE computed on all evaluation
        points concatenated across seeds (a single summary number per model).

        **Ratios** (pooled, vs the best — smallest — pooled error among models):

        - ``rel_MSE_pooled``: pooled MSE divided by the minimum pooled MSE.
        - ``rel_MAE_pooled``: pooled MAE divided by the minimum pooled MAE.

        The best model on each metric has ratio ``1.0``; larger is worse.
        """
        from benchmark.metrics import mae, mse

        pooled_actuals = np.concatenate([r.actuals.ravel() for r in self.per_seed])
        pooled_preds: dict[str, np.ndarray] = {
            n: np.concatenate([r.predictions[n].ravel() for r in self.per_seed])
            for n in self.forecaster_names
        }
        pooled_mse = {
            n: mse(pooled_actuals, pooled_preds[n]) for n in self.forecaster_names
        }
        pooled_mae = {
            n: mae(pooled_actuals, pooled_preds[n]) for n in self.forecaster_names
        }
        best_mse = min(pooled_mse.values()) if pooled_mse else float("nan")
        best_mae = min(pooled_mae.values()) if pooled_mae else float("nan")

        rows: list[dict[str, object]] = []
        for name in self.forecaster_names:
            row: dict[str, object] = {"Forecaster": name}
            mse_vals: list[float] = []
            mae_vals: list[float] = []
            for sd, res in zip(self.seeds, self.per_seed):
                a = res.actuals.ravel()
                p = res.predictions[name].ravel()
                mse_vals.append(mse(a, p))
                mae_vals.append(mae(a, p))
            for sd, mv in zip(self.seeds, mse_vals):
                row[f"MSE_{sd}"] = mv
            arr_m = np.asarray(mse_vals, dtype=np.float64)
            row["MSE_mean_seeds"] = float(np.mean(arr_m))
            row["MSE_std_seeds"] = (
                float(np.std(arr_m, ddof=1)) if arr_m.size > 1 else 0.0
            )
            mp = pooled_mse[name]
            row["MSE_pooled"] = mp
            for sd, av in zip(self.seeds, mae_vals):
                row[f"MAE_{sd}"] = av
            arr_a = np.asarray(mae_vals, dtype=np.float64)
            row["MAE_mean_seeds"] = float(np.mean(arr_a))
            row["MAE_std_seeds"] = (
                float(np.std(arr_a, ddof=1)) if arr_a.size > 1 else 0.0
            )
            ap = pooled_mae[name]
            row["MAE_pooled"] = ap
            row["rel_MSE_pooled"] = (
                float(mp / best_mse) if best_mse > 0.0 and np.isfinite(best_mse) else float("nan")
            )
            row["rel_MAE_pooled"] = (
                float(ap / best_mae) if best_mae > 0.0 and np.isfinite(best_mae) else float("nan")
            )
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Probabilistic (Monte Carlo)
    # ------------------------------------------------------------------

    def coverage_table(
        self,
        nominals: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    ) -> pd.DataFrame:
        """Coverage per (forecaster, nominal): mean and std across seeds.

        Seeds with no quantiles for a model contribute NaN; ``N_valid`` is the
        mean of per-seed ``N_valid`` counts.
        """
        key_cols = ["Forecaster", "Interval", "Nominal"]
        parts = [r.coverage_table(nominals=nominals) for r in self.per_seed]
        parts = [p for p in parts if p is not None]
        if not parts or all(p.empty for p in parts):
            return pd.DataFrame(
                columns=key_cols
                + ["Empirical_coverage_mean", "Empirical_coverage_std", "N_valid_mean"]
            )
        merged = None
        for i, p in enumerate(parts):
            if p.empty:
                continue
            sub = p[key_cols + ["Empirical_coverage", "N_valid"]].copy()
            sub = sub.rename(
                columns={
                    "Empirical_coverage": f"coverage_seed_{i}",
                    "N_valid": f"n_seed_{i}",
                }
            )
            merged = sub if merged is None else pd.merge(merged, sub, on=key_cols, how="outer")
        if merged is None:
            return pd.DataFrame(
                columns=key_cols
                + ["Empirical_coverage_mean", "Empirical_coverage_std", "N_valid_mean"]
            )
        cov_cols = [c for c in merged.columns if c.startswith("coverage_seed_")]
        n_cols = [c for c in merged.columns if c.startswith("n_seed_")]
        merged["Empirical_coverage_mean"] = merged[cov_cols].mean(axis=1, skipna=True)
        merged["Empirical_coverage_std"] = merged[cov_cols].std(axis=1, ddof=1, skipna=True)
        merged["N_valid_mean"] = merged[n_cols].mean(axis=1, skipna=True) if n_cols else np.nan
        out = merged[key_cols + [
            "Empirical_coverage_mean",
            "Empirical_coverage_std",
            "N_valid_mean",
        ]]
        return out

    def probabilistic_summary_phase2(
        self,
        nominal: float = 0.9,
        n_samples: int = 512,
        random_state: int | np.random.Generator | None = 42,
    ) -> pd.DataFrame:
        """Mean ± std of per-seed :meth:`BenchmarkResults.probabilistic_summary_phase2`."""
        parts: list[pd.DataFrame] = []
        for r, sd in zip(self.per_seed, self.seeds):
            if isinstance(random_state, np.random.Generator):
                rs: int | np.random.Generator | None = random_state
            elif random_state is None:
                rs = int(sd)
            else:
                rs = int(random_state) + int(sd)
            parts.append(
                r.probabilistic_summary_phase2(
                    nominal=nominal,
                    n_samples=n_samples,
                    random_state=rs,
                )
            )
        if not parts or all(p.empty for p in parts):
            return pd.DataFrame(
                columns=[
                    "Forecaster",
                    "Nominal",
                    "Empirical_coverage_mean",
                    "Empirical_coverage_std",
                    "Mean_PI_width_mean",
                    "Mean_PI_width_std",
                    "Coverage_error_mean",
                    "Coverage_error_std",
                    "N_valid_mean",
                    "mean_log_score_kde_mean",
                    "mean_log_score_kde_std",
                    "n_log_score_cells_mean",
                    "n_log_score_cells_std",
                ]
            )
        key = ["Forecaster", "Nominal"]
        merged = parts[0].copy()
        for i, p in enumerate(parts[1:], start=1):
            suffix = f"_seed{i}"
            right = p.rename(
                columns={
                    c: f"{c}{suffix}"
                    for c in p.columns
                    if c not in key
                }
            )
            merged = pd.merge(merged, right, on=key, how="outer")
        stat_cols = [
            "Empirical_coverage",
            "Mean_PI_width",
            "Coverage_error",
            "N_valid",
            "mean_log_score_kde",
            "n_log_score_cells",
        ]
        out_rows: list[dict[str, object]] = []
        for _, row in merged.iterrows():
            rec: dict[str, object] = {
                "Forecaster": row["Forecaster"],
                "Nominal": row["Nominal"],
            }
            for base in stat_cols:
                vals = [
                    float(row[c])
                    for c in merged.columns
                    if c == base or c.startswith(base + "_seed")
                ]
                if not vals:
                    rec[f"{base}_mean"] = np.nan
                    rec[f"{base}_std"] = np.nan
                else:
                    arr = np.array(vals, dtype=np.float64)
                    rec[f"{base}_mean"] = float(np.nanmean(arr))
                    rec[f"{base}_std"] = (
                        float(np.nanstd(arr, ddof=1))
                        if np.sum(np.isfinite(arr)) > 1
                        else 0.0
                    )
            out_rows.append(rec)
        return pd.DataFrame(out_rows)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_metric_distribution(self, metric: str = "mse", ax=None):
        """Box plot of per-seed metric values for each forecaster."""
        import matplotlib.pyplot as plt

        df = self.per_seed_metric(metric)
        col = metric.upper()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        names = self.forecaster_names
        data = [df.loc[df["Forecaster"] == n, col].values for n in names]
        bp = ax.boxplot(data, labels=names, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_ylabel(col)
        ax.set_title(f"{col} distribution across {self.n_seeds} seeds — {self.dgp_name}")
        return ax

    def plot_pooled_cumulative_error(self, ax=None):
        """Cumulative SE across all seeds concatenated, for visual comparison."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        for name in self.forecaster_names:
            pooled_se = np.concatenate([
                (r.actuals.ravel() - r.predictions[name].ravel()) ** 2
                for r in self.per_seed
            ])
            ax.plot(np.cumsum(pooled_se), label=name)
        ax.set_xlabel("Pooled forecast origin")
        ax.set_ylabel("Cumulative Squared Error")
        ax.set_title(f"Pooled Cumulative Error — {self.dgp_name} ({self.n_seeds} seeds)")
        ax.legend()
        return ax

    def plot_recursive_mse(self, ax=None):
        """Recursive (running) MSE plot pooled across seeds.

        At forecast-origin index k the y-value is the mean squared error
        computed over the first k+1 pooled predictions, i.e.
        ``cumsum(SE) / arange(1, N+1)``.  This shows how each forecaster's
        MSE estimate evolves as more predictions accumulate.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        for name in self.forecaster_names:
            pooled_se = np.concatenate([
                (r.actuals.ravel() - r.predictions[name].ravel()) ** 2
                for r in self.per_seed
            ])
            k = np.arange(1, len(pooled_se) + 1)
            ax.plot(k, np.cumsum(pooled_se) / k, label=name)

        ax.set_xlabel("Pooled forecast origin")
        ax.set_ylabel("Recursive MSE")
        ax.set_title(
            f"Recursive MSE — {self.dgp_name} ({self.n_seeds} seeds)"
        )
        ax.legend()
        return ax

    def to_csv(self, path: str | Path) -> None:
        """Export the aggregate table to CSV."""
        self.aggregate_metrics().to_csv(path, index=False)
