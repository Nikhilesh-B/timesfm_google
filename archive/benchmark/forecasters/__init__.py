"""Forecaster implementations."""

from archive.benchmark.forecasters.base import Forecaster
from archive.benchmark.forecasters.mean import MeanForecaster
from archive.benchmark.forecasters.arima import ARIMAForecaster
from archive.benchmark.forecasters.ssa import SSAForecaster
from archive.benchmark.forecasters.timesfm_forecaster import TimesFMForecaster

__all__ = [
    "Forecaster",
    "MeanForecaster",
    "ARIMAForecaster",
    "SSAForecaster",
    "TimesFMForecaster",
]
