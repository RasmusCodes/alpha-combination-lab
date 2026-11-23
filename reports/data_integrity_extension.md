# Data integrity: signal extension (no performance statistics)

Built 2026-11-23 in 52 s from I/B/E/S (unadjusted summary and actuals) and OptionMetrics (30-day surface at month ends), linked to CRSP through the WRDS link tables. Written before any return-based statistic of the new signals was computed.

## Coverage of the new signals (share of S&P 500 universe rows with a value)

|  | REV1 | REVB | SUE | DISP | IVSKEW | IVSPREAD | VRP |
|---|---|---|---|---|---|---|---|
| 1975-1979 | 0.67 | 0.68 | 0.00 | 0.68 | 0.00 | 0.00 | 0.00 |
| 1980-1984 | 0.82 | 0.88 | 0.03 | 0.88 | 0.00 | 0.00 | 0.00 |
| 1985-1989 | 0.88 | 0.98 | 0.75 | 0.97 | 0.00 | 0.00 | 0.00 |
| 1990-1994 | 0.89 | 0.98 | 0.92 | 0.98 | 0.00 | 0.00 | 0.00 |
| 1995-1999 | 0.88 | 0.99 | 0.93 | 0.99 | 0.78 | 0.78 | 0.78 |
| 2000-2004 | 0.90 | 0.99 | 0.94 | 0.99 | 0.99 | 0.99 | 0.99 |
| 2005-2009 | 0.90 | 0.98 | 0.94 | 0.98 | 0.99 | 0.99 | 0.99 |
| 2010-2014 | 0.91 | 0.99 | 0.95 | 0.99 | 1.00 | 1.00 | 1.00 |
| 2015-2019 | 0.91 | 0.99 | 0.95 | 0.99 | 1.00 | 1.00 | 1.00 |
| 2020-2024 | 0.91 | 0.99 | 0.95 | 0.99 | 0.98 | 0.98 | 0.98 |

## Correlation of the new signals with the sixteen (oriented, average within-month rank correlation, 1996 onward)

|  | MOM | IMOM | HIGH52 | STREV | LTREV | BM | EP | CFP | GPA | ROA | ACC | AG | NSI | IVOL | BETA | SEAS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REV1 | 0.28 | 0.13 | 0.28 | -0.11 | -0.02 | -0.12 | -0.11 | -0.13 | 0.01 | 0.01 | 0.03 | 0.01 | -0.00 | 0.07 | 0.03 | 0.01 |
| REVB | 0.31 | 0.13 | 0.31 | -0.12 | -0.04 | -0.14 | -0.13 | -0.15 | 0.03 | 0.02 | 0.03 | 0.00 | -0.00 | 0.07 | 0.02 | 0.01 |
| SUE | 0.14 | 0.06 | 0.13 | -0.08 | 0.07 | 0.04 | -0.01 | 0.01 | -0.02 | -0.06 | 0.05 | 0.06 | 0.02 | -0.02 | -0.04 | -0.01 |
| DISP | 0.14 | 0.06 | 0.27 | -0.02 | -0.23 | -0.31 | 0.04 | -0.12 | 0.23 | 0.29 | -0.15 | -0.08 | 0.16 | 0.33 | 0.26 | 0.04 |
| IVSKEW | 0.04 | 0.04 | 0.01 | 0.02 | -0.06 | -0.10 | -0.01 | -0.02 | 0.12 | 0.11 | 0.04 | -0.02 | 0.05 | -0.00 | 0.06 | 0.01 |
| IVSPREAD | -0.01 | -0.00 | -0.05 | 0.03 | -0.01 | -0.01 | -0.00 | -0.01 | 0.02 | 0.02 | 0.01 | -0.01 | 0.03 | -0.02 | -0.03 | 0.00 |
| VRP | -0.00 | -0.02 | -0.03 | -0.06 | -0.06 | -0.01 | 0.02 | -0.00 | -0.05 | 0.01 | -0.00 | -0.04 | -0.03 | -0.42 | -0.08 | 0.01 |

## Among the new signals

|  | REV1 | REVB | SUE | DISP | IVSKEW | IVSPREAD | VRP |
|---|---|---|---|---|---|---|---|
| REV1 | 1.00 | 0.78 | 0.33 | 0.11 | 0.02 | -0.01 | -0.01 |
| REVB | 0.78 | 1.00 | 0.34 | 0.13 | 0.02 | -0.02 | -0.01 |
| SUE | 0.33 | 0.34 | 1.00 | -0.06 | -0.00 | -0.01 | -0.01 |
| DISP | 0.11 | 0.13 | -0.06 | 1.00 | 0.04 | -0.01 | 0.00 |
| IVSKEW | 0.02 | 0.02 | -0.00 | 0.04 | 1.00 | 0.40 | -0.04 |
| IVSPREAD | -0.01 | -0.02 | -0.01 | -0.01 | 0.40 | 1.00 | -0.00 |
| VRP | -0.01 | -0.01 | -0.01 | 0.00 | -0.04 | -0.00 | 1.00 |

## Effective number of independent signals

- Sixteen original signals, 1996 onward: 10.02
- Twenty (with earnings), 1985 onward: 11.65
- Twenty-three (with options), 1996 onward: 14.07

## Cross-checks

- OptionMetrics standardized at-the-money 30-day IV vs the surface at delta 50: 264,536 secid-months, correlation 0.993 (calls) / 0.996 (puts), median absolute difference 0.0011 (calls).
- I/B/E/S FY1 consensus rows linked to S&P 500 PERMNOs: 464,238; quarterly surprises: 124,329; OptionMetrics PERMNO-months: 262,769.

## Specs

| name | family | direction | published | records | first |
|---|---|---|---|---|---|
| REV1 | earnings | 1 | 1996 | 256512 | 1976-02-29 |
| REVB | earnings | 1 | 1996 | 279700 | 1976-01-31 |
| SUE | earnings | 1 | 2006 | 221500 | 1984-05-31 |
| DISP | earnings | -1 | 2002 | 279590 | 1976-01-31 |
| IVSKEW | options | -1 | 2010 | 172703 | 1996-01-31 |
| IVSPREAD | options | 1 | 2010 | 172703 | 1996-01-31 |
| VRP | options | 1 | 2009 | 172679 | 1996-01-31 |