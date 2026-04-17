"""BenchmarkResults -- stores predictions/actuals and computes comparisons."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.metrics import METRIC_REGISTRY, diebold_mariano_hac


class BenchmarkResults:
    """Container for the output of a benchmark run.

    Attributes:
        series_name: Name of the :class:`TimeSeries` that was benchmarked.
        horizon: Forecast horizon used.
        k_first: Initial training-window size.
        forecaster_names: Ordered list of forecaster names.
        predictions: ``{forecaster_name: np.ndarray}`` of shape ``(n_origins, horizon)``.
        actuals: ``np.ndarray`` of shape ``(n_origins, horizon)``.
    """

    def __init__(
        self,
        series_name: str,
        horizon: int,
        k_first: int,
        forecaster_names: list[str],
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> None:
        self.series_name = series_name
        self.horizon = horizon
        self.k_first = k_first
        self.forecaster_names = forecaster_names
        self.predictions = predictions
        self.actuals = actuals
        self.n_origins = actuals.shape[0]

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

    def to_csv(self, path: str | Path) -> None:
        """Export the aggregate table to CSV."""
        self.aggregate_metrics().to_csv(path, index=False)
