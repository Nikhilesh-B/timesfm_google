"""Time Series Benchmarking Framework.

Usage::

    from benchmark import (
        TimeSeries, SeriesRegistry,
        MeanForecaster, ARIMAForecaster, SSAForecaster, TimesFMForecaster,
        BenchmarkRunner, BenchmarkResults,
    )
"""

from benchmark.series import TimeSeries
from benchmark.registry import SeriesRegistry
from benchmark.runner import BenchmarkRunner, ReplicatedBenchmarkRunner
from benchmark.results import BenchmarkResults, ReplicatedBenchmarkResults
from benchmark.forecasters import (
    Forecaster,
    MeanForecaster,
    ARIMAForecaster,
    SSAForecaster,
    TimesFMForecaster,
)

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
