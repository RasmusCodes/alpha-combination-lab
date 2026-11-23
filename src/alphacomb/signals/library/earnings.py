"""Analyst signals from I/B/E/S (signal extension; not part of the pre-registered sixteen).

Per-share forecasts and actuals are UNADJUSTED, so they are scaled by the same-date CRSP price, and any month in
which CRSP's cumulative price factor changed (a split) between the forecast and the formation date blanks the
value instead of mixing share bases. A consensus is known at the close of its statistics-period month; an actual
from its announcement date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphacomb.signals.base import SignalSpec
from alphacomb.signals.registry import register

IBES = "ibes.statsumu_epsus+actu_epsus+wrdsapps.ibcrsphist"
CRSP_PRICE = "crsp.msf (price, cfacpr)"


def _long(w: pd.DataFrame, name: str) -> pd.Series:
    s = w.stack()
    s.index.names = ["month", "permno"]
    return s.rename(name)


def _price_and_splits(b) -> pd.DataFrame:
    """(month, permno) -> |price| and whether CRSP's cumulative price factor changed in that month."""
    prc = b.wide("prc").abs()
    cf = b.wide("cfacpr")
    split = (cf.ne(cf.shift(1)) & cf.notna() & cf.shift(1).notna()).astype(float).where(cf.notna())
    cum = split.fillna(0).cumsum()
    out = pd.concat([_long(prc, "price"), _long(split, "split"), _long(cum, "cumsplit")], axis=1).reset_index()
    return out


def _records(b, x: pd.DataFrame, value: str, conf: str | float = 1.0, nobs: str | float = 1) -> pd.DataFrame:
    u = b.universe[["permno", "month"]]
    y = u.merge(x, on=["permno", "month"], how="left")
    return pd.DataFrame({"instrument": y["permno"].to_numpy(), "timestamp": y["month"].to_numpy(),
                         "value": y[value].to_numpy(dtype=float),
                         "confidence": (y[conf] if isinstance(conf, str) else conf),
                         "n_obs": (y[nobs] if isinstance(nobs, str) else nobs)})


@register(SignalSpec(
    name="REV1", family="earnings", direction=1, published=1996, min_history=2,
    description="One-month change in the mean FY1 EPS forecast (same fiscal year) divided by the month-end price; "
                "blank when a split occurred in this or the previous month",
    reference="Chan, Jegadeesh & Lakonishok (1996), Journal of Finance", inputs=(IBES, CRSP_PRICE),
    lag="consensus of the statistics period in month t and t-1 (mid-month), known at the close of t"))
def rev1(b):
    f = b.ibes_fy1.sort_values(["permno", "month"]).reset_index(drop=True)
    prev = f.groupby("permno")[["month", "fpedats", "meanest"]].shift(1)
    ok = (prev["month"] == f["month"] - pd.offsets.MonthEnd(1)) & (prev["fpedats"] == f["fpedats"])
    ps = _price_and_splits(b)
    x = f.merge(ps, on=["permno", "month"], how="left")
    prev_split = ps.assign(month=ps["month"] + pd.offsets.MonthEnd(1))[["permno", "month", "split"]]
    x = x.merge(prev_split.rename(columns={"split": "split_prev"}), on=["permno", "month"], how="left")
    good = ok & (x["split"] == 0) & (x["split_prev"] == 0) & (x["price"] > 0)
    x["v"] = ((f["meanest"] - prev["meanest"]) / x["price"]).where(good)
    x["conf"] = np.clip(x["numest"] / 5.0, 0, 1)
    return _records(b, x, "v", "conf", "numest")


@register(SignalSpec(
    name="REVB", family="earnings", direction=1, published=1996, min_history=1,
    description="Revision breadth: (FY1 estimates revised up - revised down) / number of estimates, at least 3 "
                "estimates",
    reference="Chan, Jegadeesh & Lakonishok (1996), Journal of Finance", inputs=(IBES,),
    lag="consensus of the statistics period in month t, known at the close of t"))
def revb(b):
    f = b.ibes_fy1.copy()
    f["v"] = ((f["numup"] - f["numdown"]) / f["numest"]).where(f["numest"] >= 3)
    f["conf"] = np.clip(f["numest"] / 5.0, 0, 1)
    return _records(b, f, "v", "conf", "numest")


@register(SignalSpec(
    name="DISP", family="earnings", direction=-1, published=2002, min_history=1,
    description="Dispersion of FY1 EPS forecasts: standard deviation / |mean| (at least 3 estimates, |mean| >= $0.01)",
    reference="Diether, Malloy & Scherbina (2002), Journal of Finance", inputs=(IBES,),
    lag="consensus of the statistics period in month t, known at the close of t"))
def disp(b):
    f = b.ibes_fy1.copy()
    f["v"] = (f["stdev"] / f["meanest"].abs()).where((f["numest"] >= 3) & (f["meanest"].abs() >= 0.01))
    f["conf"] = np.clip(f["numest"] / 5.0, 0, 1)
    return _records(b, f, "v", "conf", "numest")


@register(SignalSpec(
    name="SUE", family="earnings", direction=1, published=2006, min_history=1,
    description="Latest quarterly EPS surprise announced in the last three months: (actual - last median forecast "
                "for that quarter made before the announcement and within 120 days of it) / month-end price; "
                "blank if a split occurred between the forecast and the formation date",
    reference="Livnat & Mendenhall (2006), Journal of Accounting Research", inputs=(IBES, CRSP_PRICE),
    lag="announcement date on or before the formation month end"))
def sue(b):
    s = b.ibes_sue.dropna(subset=["anndats", "actual", "forecast", "fstat"])
    s = s[s["fstat"] >= s["anndats"] - pd.Timedelta(days=120)].sort_values("anndats")
    u = b.universe[["permno", "month"]].sort_values("month")
    x = pd.merge_asof(u, s, left_on="month", right_on="anndats", by="permno", direction="backward")
    x = x[x["anndats"].notna() & (x["anndats"] > x["month"] - pd.offsets.MonthEnd(3))]
    ps = _price_and_splits(b)
    x = x.merge(ps[["permno", "month", "price", "cumsplit"]], on=["permno", "month"], how="left")
    fm = (x["fstat"] + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")
    before = ps[["permno", "month", "cumsplit"]].rename(columns={"month": "fm", "cumsplit": "cum_f"})
    before["fm"] = before["fm"] + pd.offsets.MonthEnd(1)    # cumulative splits up to the month BEFORE the forecast
    x = x.assign(fm=fm).merge(before, on=["permno", "fm"], how="left")
    nosplit = (x["cumsplit"] - x["cum_f"].fillna(0)) == 0
    x["v"] = ((x["actual"] - x["forecast"]) / x["price"]).where(nosplit & (x["price"] > 0))
    return _records(b, x, "v", 1.0, 1)
