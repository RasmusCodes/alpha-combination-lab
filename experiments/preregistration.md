# Pre-registration: combining sixteen published equity signals

Written 2026-11-21, after the data build and the data-integrity report (`reports/data_integrity.md`: coverage and
signal correlations only) and **before any return-based statistic of any signal or method was computed**. It is
committed to git together with the frozen configuration `experiments/configs/main.yaml`; the commit precedes the
first real-data run in the history, and every run is logged in `registry/experiments.jsonl`.

## Question

Given sixteen imperfect, partly correlated return signals, does any statistical-learning method combine them into
better out-of-sample forecasts and better after-cost portfolios than giving every signal the same weight?

## Data and sample (fixed)

- Universe: S&P 500 members at each formation month's last trading day (CRSP `msp500list`), not yet delisted,
  market cap known: 492-506 stocks a month. Returns include delisting returns (Shumway fill where missing).
- Signals (direction = published sign): MOM +, IMOM +, HIGH52 +, STREV -, LTREV -, BM +, EP +, CFP +, GPA +, ROA +,
  ACC -, AG -, NSI -, IVOL -, BETA -, SEAS +. Compustat items usable six months after fiscal year end.
- Each month every signal is ranked, mapped to normal scores, multiplied by its sign; missing = 0 (neutral).
- Target for model fitting: next-month total return as within-month normal scores.
- Panel 1967-01 to 2024-12. Out-of-sample formation months **1980-01 to 2024-11 (539 months)**.

## Protocol (fixed)

- Walk-forward, expanding window. Parameters re-estimated every January on all formation months whose outcome is
  known, minus a one-month embargo. Hyperparameters re-selected every five years (1980, 1985, ... 2020) by nested
  forward-chaining validation inside the training window (three 36-month blocks, each trained only on earlier
  months), criterion mean monthly rank IC, rule "best".
- Online methods (DMA, ONLINE_EG, ONLINE_RLS) also learn each month's outcome as soon as it is known.
- Methods and grids: as in `src/alphacomb/models` at this commit. Simple: EW (benchmark), HEW (equal per family),
  ICW. Complex: OLS, RIDGE, LASSO, ENET, PCR, MVO, BMA, BSHRINK, RF, GBM, DMA, REGIME, ONLINE_EG, ONLINE_RLS.
- Portfolio: rank-weighted dollar-neutral long-short ($1 long, $1 short), rebalanced monthly, drift-adjusted
  turnover. Costs: era schedule in `evaluation/costs.py` (base) and twice that (high), scaled per stock by
  liquidity. Net = gross minus costs.

## Hypotheses and tests

Primary (confirmatory). Benchmark = EW. Sixteen comparisons per hypothesis.

- **H1 (economic).** For each method m: Sharpe(net, base costs, m) = Sharpe(net, EW). Two-sided Ledoit-Wolf (2008)
  studentized stationary bootstrap (mean block 6 months, 5,000 draws). Adjusted with Holm (FWER 5%); Benjamini-
  Hochberg and Romano-Wolf reported alongside.
- **H2 (predictive).** For each method m: mean monthly rank IC(m) = mean rank IC(EW). Newey-West t test on the
  monthly difference; same adjustments.
- **Decision rule.** "A complex method beats equal weighting" only if some complex method has a positive difference
  with Holm-adjusted p < 0.05 in H1 or H2 **and** Hansen's SPA test (consistent p-value) of the best complex method
  against EW rejects at 5% for the same metric. "Equal weighting beats method m" if the difference is negative with
  Holm p < 0.05. Otherwise: no detectable difference.

Power (stated as ranges because the correlation between a method's and EW's returns is unknown in advance): with
539 months, a two-sided 5% test has 80% power for an annualized Sharpe difference of about 0.19 / 0.32 / 0.42 when
that correlation is 0.9 / 0.7 / 0.5; with the Holm bar for 16 tests, about 0.25 / 0.44 / 0.57. Rank-IC differences
are measured more precisely (roughly 0.005-0.01 detectable).

Secondary (confirmatory, reported whatever they show):

- **H3** Hansen SPA (consistent, lower and upper p) for the best of all 16 against EW, on net returns and rank IC.
- **H4** Deflated Sharpe ratio of the best method's net Sharpe, with the number of trials taken from the registry
  (all completed real-data configurations) and their Sharpe-ratio variance; also with the effective number of trials.
- **H5** Probability of backtest overfitting (CSCV, 16 blocks) over the fixed-hyperparameter strategy set of E6.
- **H6** Complexity: within each tunable family, the Spearman correlation between complexity and out-of-sample rank
  IC across its fixed configurations (E6). Descriptive.
- **H7** REGIME vs OLS (its pooled counterpart) on rank IC: does conditioning weights on market regimes help?
  Two-sided Newey-West test. Results by volatility regime, market state and NBER recession are descriptive.
- **H8** Real-time signal availability (E5): a signal enters only from January of its publication year. H1 and H2
  are repeated against real-time EW. The gap between hindsight and real-time EW is descriptive.
- **H9** Leakage (E4): for RIDGE, ENET and GBM, the optimism of the inner validation score (selected configuration's
  validation rank IC minus the next five years' realized rank IC) is compared across forward-chaining, purged
  K-fold (embargo 12) and naive shuffled K-fold. Expected order: naive > purged >= forward. Paired t test over tuning
  events.

## Experiments

- **E1 synthetic** (positive controls; development, not counted as real-data trials): seven DGPs
  (`data/synthetic.py`) x IC 0.03 / 0.10 x three seeds, N = 300, T = 360, the same protocol scaled down. Expected:
  EW within noise of the best on `linear_equal`; regressions beat EW on `linear_unequal`, `sparse`, `wrong_sign` at
  IC 0.10; trees beat linear methods on `nonlinear` at IC 0.10; REGIME beats OLS on `regime`; adaptive methods beat
  expanding OLS on `decay`. If these fail, the machinery, not the market, is the suspect.
- **E2 main**: all methods, the protocol above (H1-H4, H7).
- **E3 robustness** (secondary): rolling 10-year windows; the one-standard-error selection rule; equal-weighted
  quintile portfolios; high costs; subperiods 1980-1999, 2000-2014, 2015-2024.
- **E4 leakage** (H9). **E5 real time** (H8). **E6 complexity** (H5, H6): every grid configuration run with fixed
  hyperparameters.
- Capacity: net Sharpe at $10M-$100B with square-root impact. Factor exposures: Fama-French five factors plus
  momentum and short-term reversal. Feature importance: out-of-sample permutation drop in rank IC by signal and family.

## Guards

Look-ahead: transforms within month only; walk-forward with purge and embargo; a test perturbs everything after a
date and checks earlier forecasts do not move (`tests/test_validation.py`). Survivorship: point-in-time membership,
delisting returns. Leakage: nested selection inside training windows only. Repeated experimentation: the registry
counts every configuration; the DSR uses that count. Instability: selection and weight stability are reported.

## Deviations

Any change after the first confirmatory run is logged in the registry as `exploratory` and listed in the report's
deviations section, with its reason.
