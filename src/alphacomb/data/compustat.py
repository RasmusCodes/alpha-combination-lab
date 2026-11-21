"""Compustat annual fundamentals, made point in time.

A fiscal year's record becomes usable at the end of the SIXTH month after its fiscal year end and stays usable for
at most 18 months (Fama-French style reporting lags; conservative because 10-K deadlines were 90 days for most of the
sample). Lagged items (for growth and accruals) come from the same firm's previous fiscal year when it ended 9-15
months earlier.

  be        Fama-French book equity: stockholders' equity (seq, else ceq + pstk, else at - lt) + deferred taxes
            (txditc) - preferred stock (pstkrv, else pstkl, else pstk)
  gp        gross profitability (revt - cogs) / at                          Novy-Marx (2013)
  roa       ib / at                                                        Haugen & Baker (1996)
  acc       balance-sheet accruals [(d act - d che) - (d lct - d dlc) - dp] / average at   Sloan (1996); taxes payable
            is not in the pull, so its change is omitted
  ag        asset growth at / at_prev - 1                                  Cooper, Gulen & Schill (2008)
  earn, cf  ib and ib + dp (divided by market equity at formation for E/P and CF/P)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_MONTHS, MAX_AGE_MONTHS = 6, 18


def annual_inputs(fl: pd.DataFrame) -> pd.DataFrame:
    f = fl.sort_values(["permno", "datadate"]).reset_index(drop=True)
    se = f["seq"].fillna(f["ceq"] + f["pstk"].fillna(0.0)).fillna(f["at"] - f["lt"])
    ps = f["pstkrv"].fillna(f["pstkl"]).fillna(f["pstk"]).fillna(0.0)
    be = se + f["txditc"].fillna(0.0) - ps
    g = f.groupby("permno")
    prev = {c: g[c].shift(1) for c in ("at", "act", "che", "lct", "dlc", "datadate")}
    gap = (f["datadate"].dt.to_period("M").astype("int64") - prev["datadate"].dt.to_period("M").astype("int64"))
    gap = gap.where(prev["datadate"].notna())
    lag_ok = (gap >= 9) & (gap <= 15)
    at, at0 = f["at"], prev["at"]
    d_wc = (f["act"] - prev["act"]) - (f["che"] - prev["che"]) - ((f["lct"] - prev["lct"]) - (f["dlc"].fillna(0)
                                                                                           - prev["dlc"].fillna(0)))
    acc = ((d_wc - f["dp"].fillna(0.0)) / ((at + at0) / 2)).where(lag_ok & (at > 0) & (at0 > 0))
    ag = (at / at0 - 1).where(lag_ok & (at0 > 0) & (at > 0))
    out = pd.DataFrame({
        "permno": f["permno"].astype("int64"), "datadate": f["datadate"],
        "avail": (f["datadate"] + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(LAG_MONTHS)).astype("datetime64[ns]"),
        "be": be, "gp": ((f["revt"] - f["cogs"]) / at).where(at > 0), "roa": (f["ib"] / at).where(at > 0),
        "acc": acc, "ag": ag, "earn": f["ib"], "cf": f["ib"] + f["dp"].fillna(0.0)})
    return out


def attach(rows: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    """rows: permno, month (formation month ends). Adds the latest annual record available at the month end (avail
    <= month, at most MAX_AGE_MONTHS after its fiscal year end); 'fund_age' is that age in months."""
    left = rows[["permno", "month"]].reset_index(drop=True).assign(_row=lambda x: np.arange(len(x)))
    left = left.astype({"permno": "int64", "month": "datetime64[ns]"})
    right = ann.sort_values("avail")
    x = pd.merge_asof(left.sort_values("month"), right, left_on="month", right_on="avail", by="permno",
                      direction="backward")
    age = (x["month"].dt.to_period("M").astype("int64") - x["datadate"].dt.to_period("M").astype("int64"))
    x["fund_age"] = age.where(x["datadate"].notna())
    stale = ~(x["fund_age"] <= MAX_AGE_MONTHS)
    for c in ("be", "gp", "roa", "acc", "ag", "earn", "cf"):
        x[c] = x[c].where(~stale)
    x.loc[stale, "fund_age"] = np.nan
    return x.sort_values("_row").drop(columns=["_row", "avail"]).reset_index(drop=True)
