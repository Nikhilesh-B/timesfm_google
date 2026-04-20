"""Time Series Benchmarking Framework.

Usage::

    from benchmark import (
        TimeSeries, SeriesRegistry,
        MeanForecaster, ARIMAForecaster, SSAForecaster, TimesFMForecaster,
        BenchmarkRunner, BenchmarkResults,
    )

Heavy dependencies (statsmodels, TimesFM, etc.) are loaded only when you
import the corresponding symbols, so ``from benchmark import TimeSeries`` stays
lightweight.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

__all__ = [
    "TimeSeries",
    "SeriesRegistry",
    "BenchmarkRunner",
    "ReplicatedBenchmarkRunner",
    "BenchmarkResults",
    "ReplicatedBenchmarkResults",
    "Forecaster",
    "MeanForecaster",
    "ARIMAForecaster",
    "SSAForecaster",
    "TimesFMForecaster",
]

if TYPE_CHECKING:
    from benchmark.forecasters.arima import ARIMAForecaster
    from benchmark.forecasters.base import Forecaster
    from benchmark.forecasters.mean import MeanForecaster
    from benchmark.forecasters.ssa import SSAForecaster
    from benchmark.forecasters.timesfm_forecaster import TimesFMForecaster
    from benchmark.registry import SeriesRegistry
    from benchmark.results import BenchmarkResults, ReplicatedBenchmarkResults
    from benchmark.runner import BenchmarkRunner, ReplicatedBenchmarkRunner
    from benchmark.series import TimeSeries


def __getattr__(name: str) -> Any:
    if name == "TimeSeries":
        from benchmark.series import TimeSeries as _TimeSeries

        out = _TimeSeries
    elif name == "SeriesRegistry":
        from benchmark.registry import SeriesRegistry as _SeriesRegistry

        out = _SeriesRegistry
    elif name == "BenchmarkRunner":
        from benchmark.runner import BenchmarkRunner as _BenchmarkRunner

        out = _BenchmarkRunner
    elif name == "ReplicatedBenchmarkRunner":
        from benchmark.runner import (
            ReplicatedBenchmarkRunner as _ReplicatedBenchmarkRunner,
        )

        out = _ReplicatedBenchmarkRunner
    elif name == "BenchmarkResults":
        from benchmark.results import BenchmarkResults as _BenchmarkResults

        out = _BenchmarkResults
    elif name == "ReplicatedBenchmarkResults":
        from benchmark.results import (
            ReplicatedBenchmarkResults as _ReplicatedBenchmarkResults,
        )

        out = _ReplicatedBenchmarkResults
    elif name == "Forecaster":
        from benchmark.forecasters.base import Forecaster as _Forecaster

        out = _Forecaster
    elif name == "MeanForecaster":
        from benchmark.forecasters.mean import MeanForecaster as _MeanForecaster

        out = _MeanForecaster
    elif name == "ARIMAForecaster":
        from benchmark.forecasters.arima import ARIMAForecaster as _ARIMAForecaster

        out = _ARIMAForecaster
    elif name == "SSAForecaster":
        from benchmark.forecasters.ssa import SSAForecaster as _SSAForecaster

        out = _SSAForecaster
    elif name == "TimesFMForecaster":
        from benchmark.forecasters.timesfm_forecaster import (
            TimesFMForecaster as _TimesFMForecaster,
        )

        out = _TimesFMForecaster
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    setattr(sys.modules[__name__], name, out)
    return out


def __dir__() -> list[str]:
    return sorted(__all__)
