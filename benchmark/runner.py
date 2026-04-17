"""BenchmarkRunner -- orchestrates the expanding-window horse race."""

from __future__ import annotations

import copy
import warnings
from typing import Callable

import numpy as np

from benchmark.forecasters.base import Forecaster
from benchmark.results import BenchmarkResults, ReplicatedBenchmarkResults
from benchmark.series import TimeSeries


class BenchmarkRunner:
    """Run an expanding-window forecast competition on a single series.

    Parameters:
        series: The :class:`TimeSeries` to benchmark on.
        forecasters: List of :class:`Forecaster` instances to compete.
        k_first: Size of the initial training window.
        horizon: Number of steps ahead to forecast at each origin.
        verbose: Print progress updates.
    """

    def __init__(
        self,
        series: TimeSeries,
        forecasters: list[Forecaster],
        k_first: int = 360,
        horizon: int = 1,
        verbose: bool = True,
    ) -> None:
        if not forecasters:
            raise ValueError("Must provide at least one forecaster")
        self.series = series
        self.forecasters = forecasters
        self.k_first = k_first
        self.horizon = horizon
        self.verbose = verbose

    def run(self) -> BenchmarkResults:
        """Execute the expanding-window evaluation and return results."""
        values = self.series.values
        n = len(values)
        origins = list(range(self.k_first, n - self.horizon + 1))
        n_origins = len(origins)

        if n_origins < 1:
            raise ValueError(
                f"No test origins: series length {n}, k_first={self.k_first}, "
                f"horizon={self.horizon}"
            )

        names = [f.name for f in self.forecasters]
        preds: dict[str, list[np.ndarray]] = {name: [] for name in names}
        actuals_list: list[np.ndarray] = []

        for step, k in enumerate(origins):
            history = values[:k]
            actual = values[k : k + self.horizon]
            actuals_list.append(actual)

            for f in self.forecasters:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pred = f.fit_predict(history, self.horizon)
                except Exception:
                    pred = np.full(self.horizon, np.nan, dtype=np.float64)
                preds[f.name].append(pred)

            if self.verbose and ((step + 1) % max(1, n_origins // 10) == 0 or step == 0):
                print(
                    f"  [{self.series.name}] step {step + 1}/{n_origins} (k={k})",
                    flush=True,
                )

        actuals_arr = np.stack(actuals_list)  # (n_origins, horizon)
        preds_arr = {name: np.stack(preds[name]) for name in names}

        if self.verbose:
            print(f"  [{self.series.name}] done — {n_origins} origins evaluated.", flush=True)

        return BenchmarkResults(
            series_name=self.series.name,
            horizon=self.horizon,
            k_first=self.k_first,
            forecaster_names=names,
            predictions=preds_arr,
            actuals=actuals_arr,
        )


class ReplicatedBenchmarkRunner:
    """Run the same benchmark across multiple random seeds (Monte Carlo replications).

    Parameters:
        dgp_factory: A callable ``(seed: int) -> TimeSeries`` that generates
            a new realisation of the DGP for each seed.
        forecasters: List of :class:`Forecaster` instances to compete.
            Each forecaster is deep-copied per seed so state doesn't leak.
        seeds: Explicit list of integer seeds.  Takes precedence over
            *n_replications* / *base_seed*.
        n_replications: Number of replications (ignored when *seeds* is given).
        base_seed: Starting seed; replications use ``base_seed .. base_seed + n - 1``.
        k_first: Initial training-window size.
        horizon: Forecast horizon.
        verbose: Print per-seed progress.
    """

    def __init__(
        self,
        dgp_factory: Callable[[int], TimeSeries],
        forecasters: list[Forecaster],
        *,
        seeds: list[int] | None = None,
        n_replications: int = 10,
        base_seed: int = 0,
        k_first: int = 360,
        horizon: int = 1,
        verbose: bool = True,
    ) -> None:
        if not forecasters:
            raise ValueError("Must provide at least one forecaster")
        self.dgp_factory = dgp_factory
        self.forecasters = forecasters
        self.seeds = seeds if seeds is not None else list(
            range(base_seed, base_seed + n_replications)
        )
        self.k_first = k_first
        self.horizon = horizon
        self.verbose = verbose

    def run(self) -> ReplicatedBenchmarkResults:
        """Execute the benchmark for every seed and return aggregated results."""
        per_seed: list[BenchmarkResults] = []
        dgp_name: str = ""

        for i, seed in enumerate(self.seeds):
            if self.verbose:
                print(f"=== Replication {i + 1}/{len(self.seeds)}  (seed={seed}) ===")

            ts = self.dgp_factory(seed)
            dgp_name = ts.name

            forecasters_copy = [copy.deepcopy(f) for f in self.forecasters]

            inner = BenchmarkRunner(
                series=ts,
                forecasters=forecasters_copy,
                k_first=self.k_first,
                horizon=self.horizon,
                verbose=self.verbose,
            )
            per_seed.append(inner.run())

        return ReplicatedBenchmarkResults(
            dgp_name=dgp_name,
            seeds=self.seeds,
            per_seed=per_seed,
        )
