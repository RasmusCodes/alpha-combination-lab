"""OptionMetrics IvyDB US (WRDS): pull and monthly implied-volatility features for the options-signal extension.

Pull (approved 2026-11-22): for securities linked (WRDS opcrsphist, score <= 3) to S&P 500 ever-members, on each
month's last trading day only: the 30-day volatility surface at deltas +-20, +-25, +-50, +-80 (vsurfd) and the
30-day standardized at-the-money options (stdopd), 1996 to 2025, plus the link table. Puts carry negative deltas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphacomb.data.wrds import MEMBERS

YEARS = range(1996, 2026)
SECIDS = f"(select distinct secid from wrdsapps.opcrsphist where permno in {MEMBERS} and score <= 3)"
DELTAS = "(20, 25, 50, 80, -20, -25, -50, -80)"
TEXT = {"cp_flag": str}


def _month_ends(table: str) -> str:
    return f"(select max(date) from {table} group by date_trunc('month', date))"


def jobs() -> list:
    out = [("om_link", f"select secid, sdate, edate, permno, score from wrdsapps.opcrsphist where permno in {MEMBERS}",
            ["sdate", "edate"], None)]
    for y in YEARS:
        v, s = f"optionm.vsurfd{y}", f"optionm.stdopd{y}"
        out.append((f"om_vsurfd_{y}", f"select secid, date, days, delta, impl_volatility, cp_flag from {v} "
                                      f"where secid in {SECIDS} and days = 30 and delta in {DELTAS} "
                                      f"and date in {_month_ends(v)}", ["date"], TEXT))
        out.append((f"om_stdopd_{y}", f"select secid, date, days, forward_price, strike_price, impl_volatility, delta, "
                                      f"cp_flag from {s} where secid in {SECIDS} and days = 30 "
                                      f"and date in {_month_ends(s)}", ["date"], TEXT))
    return out


def link_permno(df: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """PERMNO for each (secid, date) inside the link's validity, best score first; when several secids map to one
    PERMNO on a date, keep the one with the best score and then the most rows."""
    lk = link[link["score"] <= 3].copy()
    lk["edate"] = lk["edate"].fillna(pd.Timestamp("2099-12-31"))
    x = df.merge(lk[["secid", "permno", "sdate", "edate", "score"]], on="secid", how="inner")
    x = x[(x["date"] >= x["sdate"]) & (x["date"] <= x["edate"])]
    return x.drop(columns=["sdate", "edate"])


def monthly_iv(surf: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """One row per (permno, formation month): 30-day implied volatilities from the surface at the month's last
    trading day: call50 / put50 (at the money), put20 (out-of-the-money put)."""
    s = surf.copy()
    s["key"] = np.where(s["cp_flag"] == "C", "call", "put") + s["delta"].abs().astype(int).astype(str)
    w = s.pivot_table(index=["secid", "date"], columns="key", values="impl_volatility", aggfunc="first").reset_index()
    w = link_permno(w, link)
    w["n_ok"] = w[["call50", "put50", "put20"]].notna().sum(axis=1)
    w = w.sort_values(["permno", "date", "score", "n_ok"], ascending=[True, True, True, False])
    w = w.drop_duplicates(["permno", "date"])
    w["month"] = (w["date"] + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")
    keep = ["permno", "month", "date", "secid"] + [c for c in ("call20", "call25", "call50", "call80", "put20", "put25",
                                                                 "put50", "put80") if c in w]
    return w[keep].reset_index(drop=True)
