"""SeriesRegistry -- a named catalog of TimeSeries objects."""

from __future__ import annotations

from pathlib import Path

from benchmark.series import TimeSeries


class SeriesRegistry:
    """Class-level registry mapping string keys to :class:`TimeSeries` objects.

    All methods are class methods operating on a shared dict so that
    registrations persist across a notebook session.
    """

    _registry: dict[str, TimeSeries] = {}

    @classmethod
    def register(cls, name: str, series: TimeSeries) -> None:
        """Register a :class:`TimeSeries` under *name* (overwrites if exists)."""
        if not isinstance(series, TimeSeries):
            raise TypeError(f"Expected TimeSeries, got {type(series).__name__}")
        cls._registry[name] = series

    @classmethod
    def register_from_csv(
        cls,
        name: str,
        path: str | Path,
        column: str,
        freq: str | None = None,
    ) -> None:
        """Load a CSV column and register it in one call."""
        ts = TimeSeries.from_csv(path, column=column, name=name, freq=freq)
        cls.register(name, ts)

    @classmethod
    def get(cls, name: str) -> TimeSeries:
        """Retrieve a registered series by name."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise KeyError(
                f"Series {name!r} not found. Available: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """Return a sorted list of all registered series names."""
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Remove all registered series."""
        cls._registry.clear()

    @classmethod
    def register_defaults(cls, n: int = 600, seed: int = 42) -> None:
        """Pre-populate the registry with standard synthetic DGPs."""
        cls.register("AR(2)", TimeSeries.from_ar(p=2, n=n, seed=seed))
        cls.register("AR(5)", TimeSeries.from_ar(p=5, n=n, seed=seed + 1))
        cls.register("ARMA(2,2)", TimeSeries.from_arma(p=2, q=2, n=n, seed=seed + 2))
        cls.register("ARMA(5,5)", TimeSeries.from_arma(p=5, q=5, n=n, seed=seed + 3))
        cls.register(
            "Seasonal(12)", TimeSeries.from_seasonal(n=n, period=12, seed=seed + 4)
        )
