"""TimesFMForecaster -- wraps Google TimesFM 2.5 (200M, PyTorch)."""

from __future__ import annotations

import numpy as np

from benchmark.forecasters.base import Forecaster


class TimesFMForecaster(Forecaster):
    """Forecaster backed by TimesFM 2.5 (200M) PyTorch model.

    The model is loaded once on first :meth:`fit` call and reused.
    The ``max_context`` parameter controls how much history (the "prompt")
    is fed to the model -- this is one of the main knobs for benchmarking.

    Parameters:
        max_context: Maximum context window for the model.
        max_horizon: Maximum horizon the compiled model supports.
        repo_id: Hugging Face model identifier.
    """

    name: str = "TimesFM"

    def __init__(
        self,
        max_context: int = 1024,
        max_horizon: int = 256,
        repo_id: str = "google/timesfm-2.5-200m-pytorch",
    ) -> None:
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.repo_id = repo_id
        self._model = None
        self._history: np.ndarray | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        import timesfm

        torch.set_float32_matmul_precision("high")
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.repo_id, torch_compile=False
        )
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )

    def fit(self, history: np.ndarray) -> None:
        self._ensure_model()
        self._history = np.asarray(history, dtype=np.float32)

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None or self._history is None:
            raise RuntimeError("Must call fit() before predict()")
        point_forecast, _ = self._model.forecast(
            horizon=horizon, inputs=[self._history]
        )
        return np.asarray(point_forecast[0, :horizon], dtype=np.float64)
