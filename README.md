# Alpha Combination Lab

How should a portfolio manager combine many noisy, correlated return signals? This repository is a research
platform for that question, and a pre-registered study that uses it: seventeen combination methods, from equal
weights to gradient boosting, Bayesian model averaging and dynamic model averaging, applied to sixteen published
US-equity signals in the S&P 500, strictly out of sample from 1980 to 2024 and after trading costs. A second
pre-registered test adds seven analyst (I/B/E/S) and options (OptionMetrics) signals.

**Interactive demo:** [rasmuscodes.github.io/projects/alpha-combination](https://rasmuscodes.github.io/projects/alpha-combination/)
· **Full report:** [reports/REPORT.md](reports/REPORT.md) · **Pre-registration:** [experiments/preregistration.md](experiments/preregistration.md)

## What the study found

| | Equal weights | Best complex method | Verdict |
|---|---|---|---|
| Mean monthly rank IC | 0.028 | 0.042 (random forest) | better forecasts; Holm p = 0.09, SPA p = 0.013 |
| Net Sharpe ratio | 0.27 | 0.75 (signal mean-variance) | Holm p = 0.12; explained by covariance and market exposure |
| Net Sharpe, market-hedged (exploratory) | 0.78 | 0.76 (mean-variance) | no method beats equal weights |

- By the pre-registered decision rule, **no complex method beats equal weighting**.
- Flexible models forecast a little better (trees +0.014 rank IC), but trade twice as much; the gain disappears
  after costs, with rolling windows, when signals are used only after publication, and after 2015.
- The best portfolio came from the **covariance** of the signal portfolios, not from forecasting: a
  minimum-variance mix that assumes every signal earns the same does as well. Much of the remaining gap was equal
  weighting's accidental short market position (beta about -0.5 from its low-risk signals).
- Shrinkage toward zero never helped (sixteen weights are well estimated from 350,000 stock-months); adaptivity
  over time (forgetting, regime switching, model averaging) never helped either.
- Synthetic positive controls show the methods find non-linear, regime-dependent and sparse truths when the signal
  is strong (oracle IC 0.10), and mostly cannot at realistic strength (0.03).
- **Second pre-registered test: four analyst and three options signals.** The analyst signals raised equal
  weighting's rank IC by about a third (1990-2024, +0.006, Holm p = 0.06), but no complex method beat equal weights
  on the richer sets (20 signals from 1990, 23 from 2004), raw or market-hedged. After 2004 five estimated linear
  methods did significantly worse than hedged equal weights. See [reports/extension.md](reports/extension.md) and
  section 9 of the report.

## What is in the repository

```
src/alphacomb/
  signals/      record interface (instrument, timestamp, horizon, value, confidence, n_obs, provenance, version),
                cross-sectional transforms, plugin registry, diagnostics, and the sixteen-signal library
  data/         CRSP/Compustat pull and point-in-time build, daily features, Fama-French factors, synthetic panels
  models/       seventeen combiners behind one interface (models/base.py), plus exploratory variants
  validation/   walk-forward engine, purged/embargoed/forward/combinatorial splits, nested model selection
  evaluation/   IC metrics, portfolios, turnover, costs, capacity, factor regressions, statistics
                (Ledoit-Wolf Sharpe test, Holm/BH/Romano-Wolf, Hansen SPA, PSR/DSR, PBO, bootstrap), stability
  registry.py   append-only experiment registry (registry/experiments.jsonl)
experiments/    pre-registration and frozen configurations
scripts/        build_data, build_signals, run_study, run_exploratory, analyze_study, export_demo_fixtures,
                pull_extension, build_extension, analyze_extension, ...
reports/        REPORT.md, generated results.md / exploratory.md / extension.md, tables/, figures/
demo/           the browser lab (engine.js mirrors the Python methods; tests.html checks parity)
tests/          pytest suite (runs without licensed data)
```

## Quick start (no licensed data needed)

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # or: pip install -e .[dev]
pytest                                             # 41 tests, including the look-ahead test for every method
```

Run a method on a synthetic panel where the truth is known:

```python
from alphacomb import models
from alphacomb.data import synthetic
from alphacomb.models.base import Moments
from alphacomb.validation import engine

s = synthetic.make(synthetic.SyntheticConfig(dgp="nonlinear", ic=0.10, seed=1))
mom = Moments.from_panel(s.panel)
cfg = engine.WFConfig(first_test=str(s.panel.dates[120].date()), min_train=60, inner_block=24, tune_every=60)
res = engine.run(models.make("GBM", n_jobs=2), s.panel, mom, cfg)
print(res.ic(s.panel)[res.months].mean())
```

The demo runs locally with `python -m http.server --directory demo` (open `/` for the lab, `/tests.html` for the
engine-vs-Python parity checks).

## Reproducing the study

CRSP and Compustat are licensed and are **not** in this repository. With a WRDS account:

```
python scripts/build_data.py --pull     # raw tables -> data/raw/crsp, processed -> data/processed
python scripts/build_signals.py         # signal records, modelling panel, reports/data_integrity.md
python scripts/run_study.py main        # also: synthetic, robust, leakage, realtime, complexity
python scripts/run_exploratory.py
python scripts/analyze_study.py
```

The second test also needs I/B/E/S and OptionMetrics (WRDS):

```
python scripts/pull_extension.py                  # raw tables -> data/raw/wrds_ext
python scripts/build_extension.py                 # new signals, panel_ext.parquet, reports/data_integrity_extension.md
python scripts/run_study.py main --config ext_a   # then ext_a_base, ext_b, ext_b_base
python scripts/analyze_extension.py               # reports/extension.md
```

Raw files can live elsewhere (`ALPHACOMB_CRSP_RAW`). Runs are checkpointed per job and logged in the registry; an
interrupted experiment resumes where it stopped. The main experiment took about 11 minutes on a laptop, the full set
a few hours.

## Adding a signal

```python
from alphacomb.signals.base import SignalSpec
from alphacomb.signals.registry import register

@register(SignalSpec(name="MY_SIGNAL", family="alternative", description="...", direction=1,
                     reference="...", published=2024, inputs=("vendor.feed",), lag="known at month end"))
def my_signal(bundle):
    return frame_with_instrument_timestamp_value      # validated against the record interface
```

A vendor file in the record format needs no code at all: `ExternalSignal(spec, path="signals.parquet")`.

## Adding a method

Subclass `alphacomb.models.base.Combiner`, implement `fit(data)` and `predict(X, ctx)` (and `update` for online
learners), list its hyperparameter grid from the most to the least regularized, and add it to `METHODS`. The
engine, nested selection, evaluation and statistics then apply unchanged.

## Data and licensing

Only portfolio-level aggregates (monthly long-short returns, ICs, average weights) are published, in `reports/` and
the demo. No security-level CRSP, Compustat, I/B/E/S or OptionMetrics data is included. Code: MIT licence.
