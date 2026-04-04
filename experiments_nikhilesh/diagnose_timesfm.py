#!/usr/bin/env python3
"""One-shot TimesFM 2.5 diagnostics (versions, naive vs forecast, short-series check).

Run from repo root with the project venv activated:
  python experiments/diagnose_timesfm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Editable install: repo root on path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))


def main() -> None:
  import torch
  import timesfm

  print("torch:", torch.__version__)
  print("timesfm:", timesfm.__path__)

  torch.set_float32_matmul_precision("high")
  inputs = [
      np.linspace(0, 1, 100),
      np.sin(np.linspace(0, 20, 67)),
  ]

  print("\nLoading model (local cache if available)...")
  model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
      "google/timesfm-2.5-200m-pytorch",
      torch_compile=False,
      local_files_only=True,
  )

  print("forecast_naive (per-series decode, no forecast() padding)...")
  naive = model.model.forecast_naive(horizon=12, inputs=inputs)
  for i, arr in enumerate(naive):
    ok = np.isfinite(arr).all()
    print(f"  series {i}: finite={ok} shape={arr.shape}")

  print("forecast() with README-style ForecastConfig...")
  model.compile(
      timesfm.ForecastConfig(
          max_context=1024,
          max_horizon=256,
          normalize_inputs=True,
          use_continuous_quantile_head=True,
          force_flip_invariance=True,
          infer_is_positive=True,
          fix_quantile_crossing=True,
      )
  )
  pf, qf = model.forecast(horizon=12, inputs=list(inputs))
  ok = np.isfinite(pf).all() and np.isfinite(qf).all()
  print(f"  point & quantile finite: {ok}")
  if not ok:
    print("  point_forecast:", pf)
    sys.exit(1)

  print("\nAll checks passed.")


if __name__ == "__main__":
  main()
