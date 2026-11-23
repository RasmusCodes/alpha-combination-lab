"""Shared helpers for the built-in signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

LIBRARY = ["MOM", "IMOM", "HIGH52", "STREV", "LTREV", "BM", "EP", "CFP", "GPA", "ROA", "ACC", "AG", "NSI",
           "IVOL", "BETA", "SEAS"]
EXT_EARNINGS = ["REV1", "REVB", "SUE", "DISP"]        # I/B/E/S (extension)
EXT_OPTIONS = ["IVSKEW", "IVSPREAD", "VRP"]          # OptionMetrics (extension)

CRSP_MONTHLY = "crsp.msf+msedelist+msenames+msp500list"
CRSP_DAILY = "crsp.dsf"
COMPUSTAT = "comp.funda+crsp.ccmxpf_lnkhist"
FRENCH = "french.factors_daily"
FUND_LAG = "annual record usable from the end of the 6th month after fiscal year end, for up to 18 months"


def cum_log_return(R: pd.DataFrame, first_lag: int, last_lag: int, min_obs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compounded return over months t-last_lag .. t-first_lag (missing months count as 0, need >= min_obs).
    Returns (cumulative return, observation count)."""
    L = np.log1p(R)
    width = last_lag - first_lag + 1
    s = L.rolling(width, min_periods=1).sum().shift(first_lag)
    n = L.notna().astype(float).rolling(width, min_periods=1).sum().shift(first_lag)
    return np.expm1(s).where(n >= min_obs), n


def fund_confidence(age: pd.Series) -> np.ndarray:
    """1 for accounting data at most 12 months past fiscal year end, 0.75 when older (still within the 18-month cap)."""
    return np.where(age <= 12, 1.0, np.where(age <= 18, 0.75, 0.0))
