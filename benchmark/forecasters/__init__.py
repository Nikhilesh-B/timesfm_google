"""Forecaster implementations."""

from benchmark.forecasters.base import Forecaster
from benchmark.forecasters.mean import MeanForecaster
from benchmark.forecasters.naive import NaiveBenchmarkForecaster
from benchmark.forecasters.arima import ARIMAForecaster
from benchmark.forecasters.auto_arima import AutoARIMAForecaster
from benchmark.forecasters.ml_bayes_ar import MLBayesARForecaster
from benchmark.forecasters.bayesian_ar import BayesianARForecaster
from benchmark.forecasters.ssa import SSAForecaster
from benchmark.forecasters.timesfm_forecaster import TimesFMForecaster

__all__ = [
    "Forecaster",
    "MeanForecaster",
    "NaiveBenchmarkForecaster",
    "ARIMAForecaster",
    "AutoARIMAForecaster",
    "MLBayesARForecaster",
    "BayesianARForecaster",
    "SSAForecaster",
    "TimesFMForecaster",
]
