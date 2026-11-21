"""Trading-cost model: one-way cost per unit traded = era base cost x stock liquidity multiplier (+ market impact
for capacity analysis).

Era base cost (one-way, bp, for the median S&P 500 stock): half-spread plus commissions and fees for an institutional
trader. Anchors: Jones (2002, "A Century of Stock Market Liquidity and Trading Costs") measures half-spread plus
one-way commission on large US stocks of about 0.84% on average over 1925-2000 and below 0.50% after 1991 (retail-like
commission schedules, so an upper bound for institutions); decimalization (2001) roughly halved large-cap spreads;
TAQ effective spreads of large US stocks were around 5-12 bp in 2003-2017 with crisis widening in 2008-09 and
March 2020. The schedule is an assumption, not a measurement: results are also shown at 2x (the 'high' case) and as
break-even costs.

Stock multiplier: (median member dollar volume / own dollar volume) ^ 0.25, clipped to [0.5, 3], using trailing
dollar volume known at formation.

Impact (capacity only): square-root law, cost = Y * sigma_daily * sqrt(trade dollars / average daily dollar volume),
Y = 1, the whole trade executed in one day (conservative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ERA_ONE_WAY_BPS = [
    ("1960-01-01", 50.0),   # fixed commissions until May 1975
    ("1975-05-01", 30.0),   # negotiated commissions, eighth ticks
    ("1991-01-01", 15.0),   # Jones (2002): total friction below 0.5% after 1991 (retail-like), lower for institutions
    ("2001-01-01", 8.0),    # decimalization
    ("2004-01-01", 6.0),
    ("2008-01-01", 9.0),    # financial crisis
    ("2010-01-01", 5.0),
    ("2020-01-01", 6.0),    # COVID volatility
    ("2021-01-01", 5.0),
]


def era_cost(dates: pd.DatetimeIndex, multiplier: float = 1.0) -> np.ndarray:
    """One-way base cost (fraction) for each date."""
    out = np.full(len(dates), np.nan)
    for start, bps in ERA_ONE_WAY_BPS:
        out[dates >= pd.Timestamp(start)] = bps * 1e-4 * multiplier
    return out


def stock_cost(base: np.ndarray, dvol: np.ndarray, offsets: np.ndarray, elasticity: float = 0.25,
               lo: float = 0.5, hi: float = 3.0) -> np.ndarray:
    """Per-row one-way cost: base (per row) x liquidity multiplier from trailing dollar volume (NaN -> median)."""
    out = np.array(base, dtype=float, copy=True)
    for t in range(len(offsets) - 1):
        a, b = offsets[t], offsets[t + 1]
        v = dvol[a:b]
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() < 5:
            continue
        med = np.median(v[ok])
        mult = np.ones(b - a)
        mult[ok] = np.clip((med / v[ok]) ** elasticity, lo, hi)
        out[a:b] *= mult
    return out
