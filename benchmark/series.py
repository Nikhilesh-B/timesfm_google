"""Unified TimeSeries container for the benchmarking framework."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def _is_stationary_ar(phi: np.ndarray) -> bool:
    p = len(phi)
    comp = np.zeros((p, p), dtype=np.float64)
    comp[0, :] = phi
    if p > 1:
        comp[1:, :-1] = np.eye(p - 1)
    return bool(np.all(np.abs(np.linalg.eigvals(comp)) < 1.0 - 1e-9))


def _sample_stationary_phi(
    p: int,
    rng: np.random.Generator,
    *,
    low: float = -1.0,
    high: float = 1.0,
    max_tries: int = 50_000,
) -> np.ndarray:
    for _ in range(max_tries):
        phi = rng.uniform(low, high, size=p)
        if _is_stationary_ar(phi):
            return phi
    raise RuntimeError(
        f"Failed to sample stationary AR({p}) coefficients after {max_tries} tries"
    )


@dataclasses.dataclass
class TimeSeries:
    """Thin wrapper around a 1-D numpy array with metadata.

    Attributes:
        values: 1-D float64 array of observations.
        name: Human-readable label used in plots / tables.
        freq: Optional frequency string (e.g. ``"M"``, ``"D"``).
    """

    values: np.ndarray
    name: str = "unnamed"
    freq: str | None = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64).ravel()
        if self.values.size == 0:
            raise ValueError("TimeSeries values must be non-empty")

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"TimeSeries(name={self.name!r}, n={len(self)}, freq={self.freq!r})"

    def expanding_windows(
        self, k_first: int, horizon: int = 1
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(history, actuals)`` pairs for expanding-window evaluation.

        For each origin ``k`` in ``[k_first, len - horizon]`` yields:
        - ``history = values[:k]``
        - ``actuals = values[k : k + horizon]``
        """
        n = len(self.values)
        if k_first < 1:
            raise ValueError("k_first must be >= 1")
        if k_first + horizon > n:
            raise ValueError(
                f"k_first={k_first} + horizon={horizon} exceeds series length {n}"
            )
        for k in range(k_first, n - horizon + 1):
            yield self.values[:k], self.values[k : k + horizon]

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_ar(
        cls,
        p: int = 2,
        n: int = 600,
        seed: int = 42,
        sigma: float = 1.0,
    ) -> TimeSeries:
        """Simulate a stationary AR(p) process."""
        rng = np.random.default_rng(seed)
        phi = _sample_stationary_phi(p, rng)
        eps = rng.normal(0.0, sigma, n)
        y = np.zeros(n, dtype=np.float64)
        y[:p] = eps[:p]
        for t in range(p, n):
            y[t] = phi @ y[t - p : t][::-1] + eps[t]
        return cls(values=y, name=f"AR({p})")

    @classmethod
    def from_arma(
        cls,
        p: int = 2,
        q: int = 2,
        n: int = 600,
        seed: int = 42,
        sigma: float = 1.0,
    ) -> TimeSeries:
        """Simulate a stationary ARMA(p, q) process."""
        rng = np.random.default_rng(seed)
        phi = _sample_stationary_phi(p, rng)
        theta = rng.uniform(-0.8, 0.8, size=q)
        eps = rng.normal(0.0, sigma, n)
        y = np.zeros(n, dtype=np.float64)
        m = max(p, q)
        y[:m] = eps[:m]
        for t in range(m, n):
            ar_part = phi @ y[t - p : t][::-1]
            ma_part = 0.0
            for j in range(1, q + 1):
                if t - j >= 0:
                    ma_part += theta[j - 1] * eps[t - j]
            y[t] = ar_part + ma_part + eps[t]
        return cls(values=y, name=f"ARMA({p},{q})")

    @classmethod
    def from_seasonal(
        cls,
        n: int = 600,
        period: int = 12,
        seed: int = 42,
        sigma: float = 0.5,
    ) -> TimeSeries:
        """Simulate airline-model-style seasonal data with trend + seasonality + noise."""
        rng = np.random.default_rng(seed)
        t = np.arange(n, dtype=np.float64)
        trend = 0.02 * t
        seasonal = np.sin(2 * np.pi * t / period)
        noise = rng.normal(0.0, sigma, n)
        y = trend + seasonal + noise
        return cls(values=y, name=f"Seasonal(m={period})", freq="M")

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        column: str,
        name: str | None = None,
        freq: str | None = None,
    ) -> TimeSeries:
        """Load a single column from a CSV file."""
        df = pd.read_csv(path)
        if column not in df.columns:
            raise KeyError(
                f"Column {column!r} not found. Available: {list(df.columns)}"
            )
        values = df[column].dropna().to_numpy(dtype=np.float64)
        label = name if name is not None else column
        return cls(values=values, name=label, freq=freq)

    @classmethod
    def from_array(
        cls,
        values: np.ndarray | list[float],
        name: str = "custom",
        freq: str | None = None,
    ) -> TimeSeries:
        """Wrap a raw array as a TimeSeries."""
        return cls(values=np.asarray(values, dtype=np.float64), name=name, freq=freq)
