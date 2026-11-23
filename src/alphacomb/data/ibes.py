"""I/B/E/S analyst data (WRDS): pull and point-in-time monthly features for the earnings-signal extension.

Pull (approved 2026-11-22): the UNADJUSTED summary file (statsumu_epsus: EPS, FY1 / FY2 / next quarter) and
unadjusted quarterly actuals (actu_epsus), for I/B/E/S tickers linked (WRDS ICLINK, score <= 3) to S&P 500
ever-members, plus the link table. Unadjusted values are on the share basis of their own date, so they are scaled
by the same-date CRSP price; changes across dates are put on one share basis with CRSP's cumulative adjustment
factor (the standard adjusted I/B/E/S file is restated for later splits, which CRSP files ending in 2024 do not
know about).

Timing: a consensus with statistics period `statpers` (mid-month) is known at that month's close; an actual is
known from its announcement date `anndats`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphacomb.data.wrds import MEMBERS

TICKERS = f"(select distinct ticker from wrdsapps.ibcrsphist where permno in {MEMBERS} and score <= 3)"
TEXT = {"ticker": str, "fpi": str, "fiscalp": str, "measure": str, "curcode": str, "curr_act": str,
        "pdicity": str, "ncusip": str}


def jobs() -> list:
    return [
        ("ibes_link", f"select ticker, permno, ncusip, sdate, edate, score from wrdsapps.ibcrsphist "
                      f"where permno in {MEMBERS}", ["sdate", "edate"], TEXT),
        ("ibes_statsumu", f"select ticker, statpers, fpi, fiscalp, numest, numup, numdown, medest, meanest, stdev, "
                          f"highest, lowest, fpedats, curcode, usfirm from ibes.statsumu_epsus "
                          f"where ticker in {TICKERS} and measure = 'EPS' and fpi in ('1','2','6')",
         ["statpers", "fpedats"], TEXT),
        ("ibes_actu", f"select ticker, pends, pdicity, anndats, value, curr_act, usfirm from ibes.actu_epsus "
                      f"where ticker in {TICKERS} and measure = 'EPS' and pdicity = 'QTR'", ["pends", "anndats"], TEXT),
    ]


def link_permno(df: pd.DataFrame, link: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Attach the PERMNO whose link (score <= 3) is valid on each row's date; best score wins, then the latest."""
    lk = link[link["score"] <= 3].copy()
    lk["edate"] = lk["edate"].fillna(pd.Timestamp("2099-12-31"))
    x = df.reset_index(drop=True).assign(_row=lambda d: np.arange(len(d))).merge(
        lk[["ticker", "permno", "sdate", "edate", "score"]], on="ticker", how="inner")
    x = x[(x[date_col] >= x["sdate"]) & (x[date_col] <= x["edate"])]
    x = x.sort_values(["_row", "score", "sdate"], ascending=[True, True, False]).drop_duplicates("_row")
    return x.drop(columns=["sdate", "edate", "_row"]).reset_index(drop=True)


def monthly_fy1(stat: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """One row per (permno, formation month): the FY1 consensus whose statistics period falls in that month."""
    s = stat[(stat["fpi"] == "1") & (stat["usfirm"] == 1) & (stat["curcode"].fillna("USD") == "USD")].copy()
    s = link_permno(s, link, "statpers")
    s["month"] = (s["statpers"] + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")
    s = s.sort_values(["permno", "month", "statpers"]).drop_duplicates(["permno", "month"], keep="last")
    return s[["permno", "month", "statpers", "fpedats", "numest", "numup", "numdown", "meanest", "medest", "stdev"]]


def quarterly_surprises(stat: pd.DataFrame, actu: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Each quarterly announcement with the last median forecast for that quarter made before it (fpi 6 = the next
    quarter to be reported), both unadjusted; columns permno, anndats, pends, actual, forecast, fstat."""
    a = actu[(actu["usfirm"] == 1) & (actu["curr_act"].fillna("USD") == "USD")].dropna(subset=["anndats", "value"])
    a = a.rename(columns={"value": "actual"})[["ticker", "pends", "anndats", "actual"]]
    f = stat[(stat["fpi"] == "6") & (stat["usfirm"] == 1)][["ticker", "statpers", "fpedats", "medest"]]
    f = f.rename(columns={"fpedats": "pends"}).dropna(subset=["medest"])
    m = a.merge(f, on=["ticker", "pends"], how="inner")
    m = m[m["statpers"] < m["anndats"]]
    m = m.sort_values(["ticker", "pends", "statpers"]).drop_duplicates(["ticker", "pends", "anndats"], keep="last")
    m = m.rename(columns={"medest": "forecast", "statpers": "fstat"})
    return link_permno(m, link, "anndats")[["permno", "pends", "anndats", "actual", "forecast", "fstat"]]
