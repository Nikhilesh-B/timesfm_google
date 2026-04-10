"""Worker helpers for parallel AR vs TimesFM Monte Carlo (spawn-safe imports)."""

from __future__ import annotations

import os
import queue
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

_TFM = None
_PROGRESS_QUEUE = None


def _task_seed(base_seed: int | None, rep: int, p: int) -> int:
    if base_seed is None:
        return int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)
    return int(base_seed + rep * 10_000 + p * 101)


def _emit_progress(event: str, **payload: object) -> None:
    if _PROGRESS_QUEUE is None:
        return
    msg = {"event": event, **payload}
    try:
        _PROGRESS_QUEUE.put(msg)
    except Exception:
        # Progress must never break the compute path.
        return


def _is_stationary_ar(phi: np.ndarray) -> bool:
    p = len(phi)
    comp = np.zeros((p, p), dtype=np.float64)
    comp[0, :] = phi
    if p > 1:
        comp[1:, :-1] = np.eye(p - 1)
    return bool(np.all(np.abs(np.linalg.eigvals(comp)) < 1.0 - 1e-9))


def sample_stationary_phi(
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
        f"failed to sample stationary AR({p}) phi after {max_tries} tries")


def simulate_ar(
    p: int,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    phi = sample_stationary_phi(p, rng)
    eps = rng.normal(0.0, 1.0, n)
    y = np.zeros(n, dtype=np.float64)
    y[:p] = eps[:p]
    for t in range(p, n):
        ar_sum = 0.0
        for lag in range(1, p + 1):
            ar_sum += phi[lag - 1] * y[t - lag]
        y[t] = ar_sum + eps[t]
    return y, phi


def forecast_ar(history: np.ndarray, p: int) -> float | None:
    try:
        if len(history) <= p:
            return None
        fit = ARIMA(history.astype(np.float64), order=(p, 0, 0)).fit()
        return float(np.asarray(fit.forecast(steps=1)).ravel()[0])
    except Exception:
        return None


def diebold_mariano_hac(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    maxlags: int | None = None,
) -> tuple[float, float]:
    d = np.asarray(loss_a, dtype=np.float64) - \
        np.asarray(loss_b, dtype=np.float64)
    t = len(d)
    if t < 2:
        return float("nan"), float("nan")
    if maxlags is None:
        maxlags = max(1, int(np.floor(4 * (t / 100) ** (2 / 9))))
    res = sm.OLS(d, np.ones(t)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.tvalues[0]), float(res.pvalues[0])


def init_worker(repo_root: str, progress_queue=None) -> None:
    """Load TimesFM once per process."""
    global _TFM, _PROGRESS_QUEUE
    repo = Path(repo_root).resolve()
    _PROGRESS_QUEUE = progress_queue
    sys.path.insert(0, str(repo / "src"))
    try:
        from dotenv import load_dotenv

        load_dotenv(repo / ".env")
    except ImportError:
        pass
    import timesfm as tfm_mod

    torch.set_float32_matmul_precision("high")
    _TFM = tfm_mod.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,
    )
    _TFM.compile(
        tfm_mod.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
    )
    _emit_progress("worker_ready", pid=os.getpid())


def run_one_task(
    rep: int,
    p: int,
    task_seed: int,
    n: int,
    k_first: int,
    task_id: int | None = None,
    progress_every_steps: int = 0,
) -> dict:
    """One Monte Carlo draw: new phi, series, expanding-window MSE + DM."""
    global _TFM
    if _TFM is None:
        raise RuntimeError("init_worker was not called")

    rng = np.random.default_rng(task_seed)
    y, phi = simulate_ar(p, n, rng)
    test_ks = list(range(k_first, n))
    se_ar: list[float] = []
    se_tf: list[float] = []
    n_fail = 0
    n_steps = len(test_ks)

    _emit_progress(
        "task_started",
        pid=os.getpid(),
        rep=rep,
        p=p,
        task_id=task_id,
        task_seed=task_seed,
        n_steps=n_steps,
    )

    for step_idx, k in enumerate(test_ks, start=1):
        actual = float(y[k])
        hist = y[:k].astype(np.float32)
        pred_tf = float(_TFM.forecast(horizon=1, inputs=[hist])[0][0, 0])
        pred_ar = forecast_ar(y[:k], p)
        if pred_ar is None:
            n_fail += 1
        else:
            e_ar = actual - pred_ar
            e_tf = actual - pred_tf
            se_ar.append(e_ar**2)
            se_tf.append(e_tf**2)

        if progress_every_steps > 0 and (
            step_idx % progress_every_steps == 0 or step_idx == n_steps
        ):
            _emit_progress(
                "task_progress",
                pid=os.getpid(),
                rep=rep,
                p=p,
                task_id=task_id,
                done_steps=step_idx,
                total_steps=n_steps,
            )

    se_ar_a = np.asarray(se_ar, dtype=np.float64)
    se_tf_a = np.asarray(se_tf, dtype=np.float64)
    dm_t, dm_p = diebold_mariano_hac(se_ar_a, se_tf_a)

    out = {
        "rep": rep,
        "p_dgp": p,
        "rng_seed": task_seed,
        "phi": ",".join(f"{x:.8f}" for x in phi),
        "n": n,
        "k_first": k_first,
        "n_test": len(test_ks),
        "ar_fit_failures": n_fail,
        "mse_ar": float(np.mean(se_ar_a)) if len(se_ar_a) else float("nan"),
        "mse_timesfm": float(np.mean(se_tf_a)) if len(se_tf_a) else float("nan"),
        "dm_t_stat": dm_t,
        "dm_pvalue_two_sided": dm_p,
    }
    _emit_progress(
        "task_finished",
        pid=os.getpid(),
        rep=rep,
        p=p,
        task_id=task_id,
        ar_fit_failures=n_fail,
    )
    return out


def run_monte_carlo_parallel(
    *,
    rng_seed: int | None = None,
    n_replications: int,
    n: int,
    k_first: int,
    repo_root: str | Path,
    ar_orders: tuple[int, ...] = (2, 5),
    max_workers: int | None = None,
    verbose_progress: bool = False,
    progress_every_tasks: int = 10,
    live_worker_updates: bool = False,
    worker_progress_every_steps: int = 25,
) -> pd.DataFrame:
    """Run all (rep, p) tasks in a process pool; each worker loads TimesFM once."""
    repo_root = Path(repo_root).resolve()
    if not ar_orders:
        raise ValueError("ar_orders must contain at least one AR order")
    tasks: list[tuple[int, int, int, int, int]] = []
    for rep in range(n_replications):
        for p in ar_orders:
            tasks.append((rep, p, _task_seed(rng_seed, rep, p), n, k_first))
    total_tasks = len(tasks)
    if progress_every_tasks < 1:
        progress_every_tasks = 1

    workers = max_workers if max_workers is not None else (os.cpu_count() or 4)

    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    # "spawn" avoids fork issues with CUDA/MPS and is required for robust notebook use on macOS.
    ctx = __import__("multiprocessing").get_context("spawn")
    progress_q = ctx.Queue() if live_worker_updates else None

    def _drain_progress_queue() -> None:
        if progress_q is None:
            return
        while True:
            try:
                msg = progress_q.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            if not verbose_progress:
                continue
            event = msg.get("event", "")
            if event == "worker_ready":
                print(f"[worker {msg.get('pid')}] ready", flush=True)
            elif event == "task_started":
                print(
                    f"[worker {msg.get('pid')}] started rep={msg.get('rep')} "
                    f"p={msg.get('p')} seed={msg.get('task_seed')} "
                    f"steps={msg.get('n_steps')}",
                    flush=True,
                )
            elif event == "task_progress":
                print(
                    f"[worker {msg.get('pid')}] progress rep={msg.get('rep')} p={msg.get('p')} "
                    f"{msg.get('done_steps')}/{msg.get('total_steps')}",
                    flush=True,
                )
            elif event == "task_finished":
                print(
                    f"[worker {msg.get('pid')}] finished rep={msg.get('rep')} p={msg.get('p')} "
                    f"ar_fail={msg.get('ar_fit_failures')}",
                    flush=True,
                )

    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(str(repo_root), progress_q),
    ) as ex:
        fut_to_task: dict = {}
        for idx, t in enumerate(tasks, start=1):
            rep, p, task_seed, _, _ = t
            if verbose_progress:
                print(
                    f"[parallel] allocated {idx}/{total_tasks}: rep={rep} p={p} seed={task_seed}",
                    flush=True,
                )
            fut = ex.submit(
                run_one_task,
                *t,
                idx,
                max(0, int(worker_progress_every_steps)),
            )
            fut_to_task[fut] = (idx, rep, p)
        completed = 0
        start_s = time.perf_counter()
        pending = set(fut_to_task.keys())
        while pending:
            done, pending = wait(pending, timeout=0.5,
                                 return_when=FIRST_COMPLETED)
            _drain_progress_queue()
            for fut in done:
                idx, rep, p = fut_to_task[fut]
                row = fut.result()
                rows.append(row)
                completed += 1
                if verbose_progress and (
                    completed % progress_every_tasks == 0 or completed == total_tasks
                ):
                    elapsed_s = time.perf_counter() - start_s
                    print(
                        f"[parallel] completed {completed}/{total_tasks}: "
                        f"rep={rep} p={p} mse_ar={row['mse_ar']:.6f} "
                        f"mse_tfm={row['mse_timesfm']:.6f} elapsed={elapsed_s:.1f}s",
                        flush=True,
                    )
        _drain_progress_queue()

    df = pd.DataFrame(rows).sort_values(
        ["rep", "p_dgp"]).reset_index(drop=True)
    return df
