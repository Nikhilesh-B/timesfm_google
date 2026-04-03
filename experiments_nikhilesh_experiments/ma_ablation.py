
#!/usr/bin/env python3
"""Ablation: MA(q) vs TimesFM on MA(2) synthetic data.

Grid:
  - n in {10, 100, 1000} total observations (50% train minimum history, 50% test
    via expanding-window one-step-ahead forecasts).
  - q in {2, 10, 20} for ARIMA(0,0,q) — the "appropriate" model here is q=2 (DGP).

TimesFM forecasts depend only on history length, not on q, so we compute each
TimesFM step once per n and reuse across q.

Usage (repo root, venv active):
  python experiments/ma_ablation.py
  python experiments/ma_ablation.py --quick   # n in {10, 100} only (optional)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

try:
  from statsmodels.tools.sm_exceptions import ConvergenceWarning

  warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
  pass
# Statsmodels can be noisy when histories are short or q is large relative to n.
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
  sys.path.insert(0, str(_REPO / "src"))


def simulate_ma2(*, max_n: int, rng: np.random.Generator, theta1: float, theta2: float, sigma: float) -> np.ndarray:
  """Gaussian MA(2) of length max_n."""
  e = rng.normal(0.0, sigma, max_n + 2)
  return e[2 : max_n + 2] + theta1 * e[1 : max_n + 1] + theta2 * e[0:max_n]


def run_ma_forecast(history: np.ndarray, q: int) -> float | None:
  try:
    if len(history) <= q:
      return None
    fit = ARIMA(history.astype(np.float64), order=(0, 0, q)).fit()
    return float(np.asarray(fit.forecast(steps=1)).ravel()[0])
  except Exception:
    return None


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--quick",
    action="store_true",
    help="Run only n=10 and n=100 (skip n=1000 for faster iteration).",
  )
  parser.add_argument(
    "--offline",
    action="store_true",
    help="Load TimesFM weights from Hugging Face cache only (no network).",
  )
  args = parser.parse_args()

  import torch
  import timesfm

  rng = np.random.default_rng(42)
  theta1, theta2 = 0.6, 0.3
  sigma_eps = 1.0
  max_n = 1000
  y_full = simulate_ma2(max_n=max_n, rng=rng, theta1=theta1, theta2=theta2, sigma=sigma_eps)

  ns = [10, 100] if args.quick else [10, 100, 1000]
  qs = [2, 10, 20]

  torch.set_float32_matmul_precision("high")
  print("Loading TimesFM (one-time)...")
  model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    torch_compile=False,
    local_files_only=args.offline,
  )
  model.compile(
    timesfm.ForecastConfig(
      max_context=1024,
      max_horizon=256,
      normalize_inputs=True,
      use_continuous_quantile_head=True,
      force_flip_invariance=True,
      infer_is_positive=False,
      fix_quantile_crossing=True,
    )
  )

  rows: list[dict] = []

  for n in ns:
    y = y_full[:n].astype(np.float64)
    k_start = n // 2
    k_indices = list(range(k_start, n))
    n_test = len(k_indices)
    print(f"\n=== n={n} (train indices 0..{k_start - 1}, {n_test} test steps) ===")

    # TimesFM: one call per forecast origin (depends only on n, not q)
    pred_tf: dict[int, float] = {}
    for j, k in enumerate(k_indices):
      if (j + 1) % max(1, n_test // 10) == 0 or j == 0:
        print(f"  TimesFM step {j + 1}/{n_test} (k={k})")
      history = y[:k].astype(np.float32)
      point_forecast, _ = model.forecast(horizon=1, inputs=[history])
      pred_tf[k] = float(point_forecast[0, 0])

    for q in qs:
      sq_ma: list[float] = []
      sq_tf: list[float] = []
      abs_ma: list[float] = []
      abs_tf: list[float] = []
      ma_wins = 0
      ma_fail = 0
      comparable = 0

      for k in k_indices:
        actual = float(y[k])
        history = y[:k]
        pred_ma = run_ma_forecast(history, q)
        pred_t = pred_tf[k]

        err_tf = actual - pred_t
        sq_tf.append(err_tf**2)
        abs_tf.append(abs(err_tf))

        if pred_ma is None or not np.isfinite(pred_ma):
          ma_fail += 1
          continue

        comparable += 1
        err_ma = actual - pred_ma
        sq_m = err_ma**2
        sq_ma.append(sq_m)
        abs_ma.append(abs(err_ma))
        if sq_m < err_tf**2:
          ma_wins += 1

      mse_ma = float(np.mean(sq_ma)) if sq_ma else float("nan")
      mse_tf = float(np.mean(sq_tf))
      mae_ma = float(np.mean(abs_ma)) if abs_ma else float("nan")
      mae_tf = float(np.mean(abs_tf))
      frac_win = ma_wins / comparable if comparable else float("nan")

      rows.append(
        {
          "n": n,
          "q": q,
          "n_test": n_test,
          "ma_failures": ma_fail,
          "mse_ma": mse_ma,
          "mse_timesfm": mse_tf,
          "mae_ma": mae_ma,
          "mae_timesfm": mae_tf,
          "frac_ma_wins_sq": frac_win,
          "mse_diff_tf_minus_ma": mse_tf - mse_ma if np.isfinite(mse_ma) else float("nan"),
        }
      )
      win_str = f"{frac_win:.1%}" if comparable else "N/A"
      print(
        f"  q={q:2d}  MSE MA={mse_ma:.6f}  MSE TF={mse_tf:.6f}  "
        f"MA wins {win_str}  MA failures={ma_fail}"
      )

  out = pd.DataFrame(rows)
  out_dir = _REPO / "experiments" / "output"
  out_dir.mkdir(parents=True, exist_ok=True)
  csv_path = out_dir / "ma_ablation_results.csv"
  out.to_csv(csv_path, index=False)
  print(f"\nSaved: {csv_path}")

  # --- Narrative summary ---
  print("\n" + "=" * 72)
  print("SUMMARY (MA(2) data generating process; ARIMA(0,0,q) competitor)")
  print("=" * 72)
  print(
    "- **q=2** is the well-specified moving-average order (matches the DGP). "
    "You generally expect the strongest classical baseline there when estimation works."
  )
  print(
    "- **q=10** and **q=20** are misspecified (too many MA lags vs MA(2) truth); MLE "
    "can overfit or be unstable. If **n/2 <= q**, the first forecast origins do not "
    "yet have enough history to fit MA(q) — see `ma_failures` and N/A win rates."
  )
  print(
    "- **TimesFM** is a single pretrained model (no training on this series); "
    "it can beat or lose to a correctly specified short-memory model depending on n "
    "and noise realization."
  )
  print("\nTable (copy from CSV for papers):")
  print(out.to_string(index=False))
  print(
    "\nWhere each approach tended to do well in this run: "
    "compare `mse_ma` vs `mse_timesfm` and `frac_ma_wins_sq` per row; "
    "positive `mse_diff_tf_minus_ma` means lower MSE for MA."
  )


if __name__ == "__main__":
  main()
