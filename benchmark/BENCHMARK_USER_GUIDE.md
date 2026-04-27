# Time series benchmark — user guide

This guide describes the **`benchmark`** Python package and the interactive notebook **[`benchmark_notebook.ipynb`](../benchmark_notebook.ipynb)** in the repository root. Together they let you compare forecasters on synthetic or custom series using an expanding-window evaluation and optional Monte Carlo replications.

---

## 1. Concepts

### 1.1 Time series

A **`TimeSeries`** wraps a one-dimensional `numpy` array of observations plus a name (and optional frequency metadata). Series can be built from arrays, CSV files, or built-in generators (AR, ARMA, seasonal). See [`series.py`](series.py).

### 1.2 Registry

**`SeriesRegistry`** is a session-wide catalog: register series by name, then retrieve them for runs. Defaults include labels such as `AR(2)`, `ARMA(2,2)`, and `Seasonal(12)`. See [`registry.py`](registry.py).

### 1.3 Forecasters

Every forecaster subclasses **`Forecaster`**: implement **`fit(history)`** then **`predict(horizon)`** to return point forecasts. The runner calls **`fit_predict(history, horizon)`** at each evaluation origin. See [`forecasters/base.py`](forecasters/base.py).

### 1.4 Expanding-window benchmark

**`BenchmarkRunner`** walks the series from index **`k_first`** to the end: at each origin **`k`**, it uses **`values[:k]`** as training history, asks each model for **`horizon`** steps ahead, and records errors against **`values[k:k+horizon]`**. Failed fits are turned into **`NaN`** predictions so a single bad model does not stop the whole run. See [`runner.py`](runner.py).

### 1.5 Results

**`BenchmarkResults`** stores predictions and actuals, exposes **`summary()`** (MSE, MAE, RMSE per model), **`diebold_mariano()`** (pairwise tests with HAC errors), and plotting helpers. See [`results.py`](results.py).

### 1.6 Monte Carlo replications

**`ReplicatedBenchmarkRunner`** draws a new series from a **DGP factory** `(seed) -> TimeSeries` for each seed, deep-copies forecasters per seed, and aggregates metrics across seeds. Use this when you care about performance **averaged over random realisations** of a process, not a single path.

**`ReplicatedBenchmarkResults.replication_scorecard()`** returns a **wide** table (one row per forecaster): **MSE** and **MAE** for each seed, their **mean and std** across seeds, **pooled** MSE/MAE on all concatenated forecast errors, and **`rel_MSE_pooled`** / **`rel_MAE_pooled`** (ratio to the best pooled value on that metric, so the winner is `1.0`). For a long ``(seed, Forecaster, metric)`` layout, use **`per_seed_metric("mse")`** or **`per_seed_metric("mae")`**.

---

## 2. Interactive notebook workflow

Open **`benchmark_notebook.ipynb`**.

1. **Imports and secrets**  
   Load dependencies and, if you use TimesFM, set **`HF_TOKEN`** (for example from a local `secrets.py`).

2. **Register series**  
   Call **`SeriesRegistry.clear()`** if you want a clean slate, then **`SeriesRegistry.register_defaults(...)`** and/or **`register`**, **`register_from_csv`**.

3. **Choose a series**  
   Use the series dropdown and preview plot.

4. **Benchmark settings**  
   Set **horizon `h`**, **`k_first`**, and which forecasters to include (checkboxes and sliders).

5. **Run benchmark**  
   Click **Run Benchmark**; inspect the summary table and Diebold–Mariano output.

6. **Plots and export**  
   Use the results cells; optional CSV export is commented in the notebook.

7. **Monte Carlo (optional)**  
   Configure DGP, replication count, seeds, and forecaster toggles, then **Run Replications**.

---

## 3. Forecasters reference

### 3.1 Mean (`MeanForecaster`)

Rolling (or full-history) mean of the last **`window`** observations; the same constant is repeated for each step of **`horizon`**. Simple baseline.

### 3.2 ARIMA (`ARIMAForecaster`)

**`statsmodels`** ARIMA with order **`(p, d, q)`**. Fitted fresh at each origin.

### 3.3 Bayes AR (`BayesianARForecaster`)

Treats **AR(p)** as linear regression on lags with a **Gaussian prior** on coefficients. Inference is **closed-form** (one small linear solve per `fit`): there is **no MCMC**, because the benchmark refits at every origin and sampling would be too slow.

| Parameter | Meaning |
|-----------|--------|
| **`p`** | AR order (number of lags). |
| **`prior_precision`** | **`λ`**: overall tightness. In **ridge** mode, diagonal prior precision is **`λ`** on each coefficient. In **Minnesota** mode, it **scales** the lag-specific precisions below. |
| **`prior_mode`** | **`"ridge"`** (default) or **`"minnesota"`**. |
| **`minnesota_lag_decay_exponent`** | Minnesota only. Precision on lag **j** is **`λ · j^exponent`**. Larger **`exponent`** puts relatively more weight on shrinking **higher** lags. **`0`** gives equal precision on all lags (still distinct from ridge if you use RW centering). |
| **`minnesota_center_rw`** | Minnesota only. If **`True`**, prior mean sets the **first AR coefficient** toward **1** (random-walk style); other coefficients toward **0**. If **`False`**, prior mean is **zero** (decaying diagonal only). |
| **`include_intercept`** | If **`True`**, adds an intercept column; Minnesota uses **`λ`** on the intercept row and RW mean on the **first lag** column after the intercept. |

**Ridge mode** is algebraically **ridge regression** on the lag design: posterior mean **\((X'X + \lambda I)^{-1} X'y\)** with prior mean **0**.

**Minnesota mode** (univariate) follows the same idea as the classic Minnesota **VAR** prior, adapted here: **tighter priors on higher lags**, optional **centering on a unit-root / RW** for the first lag. Multi-step forecasts still **recurse** using the **posterior mean** (plug-in), not full predictive simulation.

Model names in tables look like **`BayesAR(p=2,ridge,λ=0.1)`** or **`BayesAR(p=3,MN,λ=1.0,dec=2.0,μ=RW)`**.

### 3.4 SSA (`SSAForecaster`)

Singular spectrum analysis forecaster (see implementation in [`forecasters/ssa.py`](forecasters/ssa.py)).

### 3.5 TimesFM (`TimesFMForecaster`)

Wraps **TimesFM 2.5** (PyTorch) with Hugging Face **`repo_id`**, **`max_context`**, and **`max_horizon`**. Requires network/token as configured in your environment.

---

## 4. Programmatic example

```python
from benchmark import (
    TimeSeries,
    SeriesRegistry,
    MeanForecaster,
    ARIMAForecaster,
    BayesianARForecaster,
    BenchmarkRunner,
)

SeriesRegistry.clear()
SeriesRegistry.register_defaults(n=600, seed=42)
ts = SeriesRegistry.get("AR(2)")

forecasters = [
    MeanForecaster(window=50),
    ARIMAForecaster(order=(2, 0, 0)),
    BayesianARForecaster(
        p=2,
        prior_precision=1.0,
        prior_mode="minnesota",
        minnesota_lag_decay_exponent=2.0,
        minnesota_center_rw=True,
    ),
]

runner = BenchmarkRunner(series=ts, forecasters=forecasters, k_first=360, horizon=1)
results = runner.run()
print(results.summary())
print(results.diebold_mariano())
```

Heavy imports (**`statsmodels`**, TimesFM, etc.) are loaded **lazily** when you first import the corresponding symbols from **`benchmark`** (see [`__init__.py`](__init__.py)).

---

## 5. Probabilistic evaluation (Phases 1–2)

When a forecaster implements **`predict_quantiles(horizon)`**, the runner stores per-origin quantiles in **`BenchmarkResults.quantile_predictions`** (tensor ``(n_origins, Q, horizon)``) and **`quantile_levels`** per model. **TimesFM** exposes the library’s native P10–P90 fan (mean channel stripped for the quantile tensor).

**Phase 1**

- **`coverage_table()`**: long-format table of empirical **central-interval coverage** vs nominal (50\%, 80\%, 90\%, 95\% by default), with interpolation between reported quantiles. Models without quantiles appear with **Note = no quantiles**.
- **`probabilistic_summary()`**: **CRPS** (integral of pinball loss on a dense τ grid), mean **90\% PI width**, and summed mean pinball losses.
- **`plot_probabilistic_calibration(name)`**, **`plot_pit_histogram(name)`**: reliability and PIT histogram from the piecewise-linear CDF implied by quantiles.

**Phase 2** (requires **[SciPy](https://scipy.org/)** for KDE; install alongside your environment, e.g. `pip install scipy`)

- **`probabilistic_summary_phase2(nominal=0.9, ...)`**: per forecaster, **mean log score** from a **Gaussian KDE** fit to Monte Carlo samples drawn from the piecewise-linear quantile function (inverse-CDF sampling), plus **sharpness** (mean PI width at `nominal`) and **calibration** (empirical coverage vs nominal, and coverage error). If SciPy is missing, **`mean_log_score_kde`** is NaN and **`n_log_score_cells`** is 0.
- **`plot_sharpness_vs_calibration(nominal)`**: scatter of mean PI width vs empirical coverage with a horizontal reference at the nominal coverage.

**`ReplicatedBenchmarkResults.coverage_table()`** averages per-seed coverage statistics (mean ± std across seeds). **`ReplicatedBenchmarkResults.probabilistic_summary_phase2()`** aggregates the Phase 2 table across seeds (mean ± std per column); each seed uses an independent KDE RNG stream derived from **`random_state + seed`** when **`random_state`** is an int.

Implementation: [`probabilistic_metrics.py`](probabilistic_metrics.py).


---

## 6. Metrics and tests

- **Metrics** in **`summary()`**: **MSE**, **MAE**, **RMSE** (see [`metrics.py`](metrics.py)).
- **`diebold_mariano(loss="se"|"ae")`**: pairwise comparison of loss sequences across origins; uses **HAC** standard errors.

---

## 7. Tips

- If **`k_first`** is too large relative to series length **`n`**, you may get **no evaluation origins**; the runner raises a clear error.
- For **`BayesianARForecaster`**, ensure **`len(history) > p`** at each origin; early origins with insufficient history yield **`NaN`** predictions for that model only.
- **TimesFM** is the slowest and most setup-heavy option; use it when you explicitly want that comparison.

For implementation details beyond this guide, browse the **`benchmark/`** package modules cited above.
