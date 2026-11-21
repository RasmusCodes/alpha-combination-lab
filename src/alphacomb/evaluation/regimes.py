"""Market regimes known at each formation month's close (used by the regime ensemble and for evaluation).

volatility  'high' when the trailing 63-trading-day realized volatility of the market exceeds its median over the
            previous 120 month ends (rolling, at least 60), else 'low'. (An expanding median was tried first, before
            any results existed: because market volatility trended up after the calm 1960s it labelled 68% of months
            'high', so the rolling version was adopted.)
state       'down' when the trailing 24-month cumulative market return is negative (Cooper, Gutierrez & Hameed 2004),
            else 'up'
regime      volatility x state (four cells)
nber        NBER recession months (peak + 1 .. trough). Announced with long lags, so used ONLY to slice results,
            never as a model input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NBER_PEAK_TROUGH = [("1960-04", "1961-02"), ("1969-12", "1970-11"), ("1973-11", "1975-03"), ("1980-01", "1980-07"),
                    ("1981-07", "1982-11"), ("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06"),
                    ("2020-02", "2020-04")]


def nber_months(index: pd.DatetimeIndex) -> np.ndarray:
    m = index.to_period("M")
    out = np.zeros(len(index), dtype=bool)
    for pk, tr in NBER_PEAK_TROUGH:
        out |= (m > pd.Period(pk, "M")) & (m <= pd.Period(tr, "M"))
    return out


def market_regimes(mkt_daily: pd.Series, month_ends: pd.DatetimeIndex, vol_window: int = 63,
                   state_months: int = 24, median_months: int = 120, burn_in_months: int = 60) -> pd.DataFrame:
    """mkt_daily: daily total market return (decimal), indexed by date. Returns a frame indexed by month_ends."""
    r = mkt_daily.dropna().sort_index()
    vol = r.rolling(vol_window, min_periods=int(vol_window * 0.8)).std() * np.sqrt(252)
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    me = month_ends.to_period("M")
    v = vol.groupby(vol.index.to_period("M")).last().reindex(me)
    cum = (1 + monthly).rolling(state_months, min_periods=state_months).apply(np.prod, raw=True) - 1
    cum = cum.reindex(me)
    med = v.rolling(median_months, min_periods=burn_in_months).median()
    vol_state = np.where(v.isna() | med.isna(), None, np.where(v > med, "high", "low"))
    state = np.where(cum.isna(), None, np.where(cum < 0, "down", "up"))
    regime = [f"{a}-{b}" if a is not None and b is not None else None for a, b in zip(vol_state, state)]
    out = pd.DataFrame({"mkt_vol": v.to_numpy(), "vol_median": med.to_numpy(), "volatility": vol_state,
                        "mkt_24m": cum.to_numpy(), "state": state, "regime": regime,
                        "nber": nber_months(month_ends)}, index=month_ends)
    return out
