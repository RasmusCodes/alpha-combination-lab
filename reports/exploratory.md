# Exploratory analyses (decided after the confirmatory results; not pre-registered)

## X2: what drives MVO

|  | rank_ic | sharpe_gross | sharpe_net | turnover | net_minus_EW | p |
|---|---|---|---|---|---|---|
| EW | 0.0280 | 0.42 | 0.27 | 6.2 |  |  |
| MVO | 0.0369 | 1.23 | 0.75 | 10.7 | 0.49 | 0.003 |
| MINVAR | 0.0303 | 1.12 | 0.76 | 7.5 | 0.50 | 0.001 |
| MVO_SAMPLE | 0.0362 | 1.34 | 0.88 | 9.5 | 0.61 | 0.001 |
| RISKPAR | 0.0294 | 0.63 | 0.44 | 6.1 | 0.18 | 0.000 |

## X1: market-hedged net returns (ex-ante 36-month beta)

Months 1983-01-31 to 2024-11-30 (503; the first 36 out-of-sample months estimate the first hedge ratio).

|  | sharpe_net | sharpe_hedged | beta_full | hedged_minus_EW | p | p_holm |
|---|---|---|---|---|---|---|
| EW | 0.26 | 0.78 | -0.47 |  |  |  |
| HEW | 0.30 | 0.81 | -0.44 | 0.03 | 0.487 | 1.000 |
| ICW | 0.26 | 0.65 | -0.35 | -0.13 | 0.131 | 1.000 |
| OLS | 0.40 | 0.63 | -0.18 | -0.15 | 0.253 | 1.000 |
| RIDGE | 0.36 | 0.68 | -0.24 | -0.10 | 0.412 | 1.000 |
| LASSO | 0.23 | 0.41 | -0.11 | -0.37 | 0.011 | 0.150 |
| ENET | 0.22 | 0.41 | -0.11 | -0.37 | 0.009 | 0.144 |
| PCR | 0.35 | 0.60 | -0.20 | -0.18 | 0.150 | 1.000 |
| MVO | 0.79 | 0.76 | 0.03 | -0.02 | 0.932 | 1.000 |
| BMA | 0.30 | 0.48 | -0.15 | -0.30 | 0.029 | 0.319 |
| BSHRINK | 0.31 | 0.47 | -0.16 | -0.31 | 0.026 | 0.312 |
| RF | 0.44 | 0.68 | -0.22 | -0.10 | 0.468 | 1.000 |
| GBM | 0.49 | 0.69 | -0.16 | -0.09 | 0.556 | 1.000 |
| DMA | 0.29 | 0.53 | -0.15 | -0.25 | 0.115 | 1.000 |
| REGIME | 0.48 | 0.63 | -0.09 | -0.15 | 0.334 | 1.000 |
| ONLINE_EG | 0.11 | 0.46 | -0.26 | -0.32 | 0.010 | 0.150 |
| ONLINE_RLS | 0.24 | 0.43 | -0.11 | -0.35 | 0.018 | 0.228 |
| MINVAR | 0.78 | 0.85 | -0.04 | 0.07 | 0.707 |  |

## X3: SPA on volatility-scaled net returns (Sharpe-consistent)

Complex methods vs equal weights, 503 months: best MVO, consistent p = 0.038 (lower 0.036, upper 0.038).
