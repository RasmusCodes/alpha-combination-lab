# Pre-registration: earnings-revision and options signals

Written 2026-11-22/23 after the main study (E1-E6, X1-X3). The data were pulled (approved 2026-11-22) and the
data-integrity report (`reports/data_integrity_extension.md`: coverage and correlations only) was written BEFORE any
return-based statistic of the new signals was computed. This file and `experiments/configs/ext_*.yaml` are committed
before the first extension run; the git history shows the order.

## Question

The main study found that sixteen price- and accounting-based signals behave like about ten independent ones, and
that no complex method beats equal weighting. Do signals from two other information sources, analysts and option
markets, (a) add predictive information to the combination, and (b) give flexible combination methods something
equal weighting cannot exploit?

## Data

| Source | Table | Filter | Rows |
|---|---|---|---|
| I/B/E/S | `ibes.statsumu_epsus` (unadjusted summary) | EPS, FY1 / FY2 / next quarter, tickers linked to S&P 500 ever-members | pulled |
| I/B/E/S | `ibes.actu_epsus` (unadjusted actuals) | quarterly EPS with announcement dates | pulled |
| WRDS link | `wrdsapps.ibcrsphist` | ticker to PERMNO, score <= 3, used inside its date range | pulled |
| OptionMetrics | `optionm.vsurfd1996-2025` | month-end trading days, 30-day surface, deltas +-20, +-25, +-50, +-80 | pulled |
| OptionMetrics | `optionm.stdopd1996-2025` | month-end trading days, 30-day standardized options (cross-check only) | pulled |
| WRDS link | `wrdsapps.opcrsphist` | secid to PERMNO, score <= 3; one secid per PERMNO-month (best score, most data) | pulled |

64 tables, 4.17 million rows in all (`data/raw/wrds_ext/manifest.json`, not published).

**Share basis.** The standard I/B/E/S file restates history for later stock splits, which CRSP files ending in 2024
do not know about, so the UNADJUSTED files are used: a per-share value is divided by the same-date CRSP price, and
when CRSP's cumulative price factor changes (a split) inside the window a value depends on, the value is blank.

## New signals (direction = published sign)

| Signal | Family | Sign | Definition | Reference |
|---|---|---|---|---|
| REV1 | earnings | + | change in the mean FY1 EPS forecast from last month's to this month's statistics period (same fiscal year) / month-end price; blank after a split in either month | Chan, Jegadeesh & Lakonishok (1996) |
| REVB | earnings | + | (FY1 estimates revised up - down) / number of estimates, at least 3 | Chan, Jegadeesh & Lakonishok (1996) |
| SUE | earnings | + | latest quarterly surprise announced in the last three months: (actual - last median forecast for the quarter made within 120 days before the announcement) / month-end price; blank after a split since the forecast | Livnat & Mendenhall (2006) |
| DISP | earnings | - | FY1 forecast standard deviation / abs(mean), at least 3 estimates, abs(mean) >= $0.01 | Diether, Malloy & Scherbina (2002) |
| IVSKEW | options | - | 30-day IV of the delta -20 put minus the delta +50 call | Xing, Zhang & Zhao (2010) |
| IVSPREAD | options | + | 30-day IV of the delta +50 call minus the delta -50 put | Cremers & Weinbaum (2010) |
| VRP | options | + | realized volatility of daily returns in month t (annualized) minus the mean at-the-money 30-day IV | Bali & Hovakimian (2009) |

Same pipeline as the main study: within-month normal scores, multiplied by the sign, missing = 0.

## Experiments (configs in `experiments/configs/`)

| Config | Signals | Panel from | Out of sample | Months |
|---|---|---|---|---|
| `ext_a` | 20: the 16 + REV1, REVB, SUE, DISP | 1980-01 | 1990-01 to 2024-11 | 419 |
| `ext_a_base` | the 16 | 1980-01 | 1990-01 to 2024-11 | 419 |
| `ext_b` | 23: the 20 + IVSKEW, IVSPREAD, VRP | 1996-01 | 2004-01 to 2024-11 | 251 |
| `ext_b_base` | the 16 | 1996-01 | 2004-01 to 2024-11 | 251 |

Protocol, methods, grids, costs and statistics as in the main study, with two changes for the shorter windows,
identical in extended and baseline runs: at least 84 training months, and 24-month inner validation blocks. BMA
enumerates all 2^K models exactly (streamed: about 1 million at K = 20, 8.4 million at K = 23).

## Hypotheses and tests

- **H10 (new information, primary).** On each window, mean monthly rank IC of EW over the extended set minus EW over
  the sixteen (same months): Newey-West t test, two-sided, Holm over the two windows. Secondary: net Sharpe difference
  (Ledoit-Wolf bootstrap) and the same on ex-ante market-hedged net returns.
- **H11 (complexity with richer signals, primary).** Within `ext_a` and within `ext_b`, the main study's decision rule
  against extended EW: some complex method has a positive difference with Holm p < 0.05 (16 comparisons) on net
  Sharpe or rank IC, and the SPA test of the best complex method rejects at 5% for the same metric. For net Sharpe the
  SPA is run on volatility-scaled returns (Sharpe-consistent; the main study's X3); raw-return SPA is also reported.
- **H12 (hedged).** As H11 on ex-ante market-hedged net returns (36-month beta of each method's own returns; the
  main study's X1, now confirmatory).
- **Descriptive.** Correlation of the new signals with the sixteen, effective number of signals (integrity report);
  permutation importance by family; each complex method's gain over EW in the extended run minus its gain in the
  baseline run (does richer information help complex methods more than equal weights?); subperiods.

Power: 419 months is close to the main study's 539; 251 months detects only about 1.3 times larger Sharpe
differences. A null on H11 in `ext_b` is therefore weak evidence.

## Deviations

Anything changed after the first extension run is logged as exploratory in the registry and listed in the report.
