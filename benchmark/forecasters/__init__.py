"""Forecaster implementations."""

from benchmark.forecasters.base import Forecaster
from benchmark.forecasters.mean import MeanForecaster
from benchmark.forecasters.arima import ARIMAForecaster
from benchmark.forecasters.ssa import SSAForecaster
from benchmark.forecasters.timesfm_forecaster import TimesFMForecaster

__all__ = [
    "Forecaster",
    "MeanForecaster",
    "ARIMAForecaster",
    "SSAForecaster",
    "TimesFMForecaster",
]
