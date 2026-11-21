"""Monthly features from CRSP daily files, computed from per-month sufficient statistics (one pass over the yearly
files, never the whole daily table in memory).

  ivol    idiosyncratic volatility: std of daily residuals from a Fama-French 3-factor regression within the month
          (>= 15 days; Ang, Hodrick, Xing & Zhang 2006)
  beta    market beta from daily excess returns over the trailing 12 months (>= 120 days)
  sigma   daily return volatility over the trailing 3 months (>= 40 days): impact model input
  high52  month-end split-adjusted price / its maximum over the trailing 12 months (George & Hwang 2004)
  dvol    mean daily dollar volume over the trailing 3 months ($): cost and capacity input
Every feature at month t uses daily data up to that month's last trading day only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphacomb.data import crsp

SUMS = ["n", "sx", "sy", "sxx", "sxy", "syy",                     # CAPM sums (x = Mkt-RF, y = excess return)
        "n3", "a11", "a12", "a13", "a22", "a23", "a33", "b0", "b1", "b2", "b3", "s1", "s2", "s3", "yy3",  # FF3
        "nr", "sr", "srr", "pmax", "plast", "dv", "ndv"]


def month_sums(d: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["date"] = d["date"].astype("datetime64[ns]")
    d["ret"] = crsp.clean_returns(d["ret"])
    d = d.join(fac[["Mkt-RF", "SMB", "HML", "RF"]], on="date", how="left")
    d["month"] = d["date"] + pd.offsets.MonthEnd(0)
    px = d["prc"].abs()
    d["padj"] = (px / d["cfacpr"]).where((px > 0) & (d["cfacpr"] > 0))
    d["dvol_d"] = (px * d["vol"]).where((px > 0) & (d["vol"] >= 0))
    ok = d["ret"].notna() & d["Mkt-RF"].notna()
    y = (d["ret"] - d["RF"]).where(ok)
    x = d["Mkt-RF"].where(ok)
    s, h = d["SMB"].where(ok), d["HML"].where(ok)
    one = ok.astype(float)
    cols = {"n": one, "sx": x, "sy": y, "sxx": x * x, "sxy": x * y, "syy": y * y,
            "n3": one, "a11": x * x, "a12": x * s, "a13": x * h, "a22": s * s, "a23": s * h, "a33": h * h,
            "b0": y, "b1": x * y, "b2": s * y, "b3": h * y, "s1": x, "s2": s, "s3": h, "yy3": y * y,
            "nr": d["ret"].notna().astype(float), "sr": d["ret"], "srr": d["ret"] ** 2,
            "dv": d["dvol_d"], "ndv": d["dvol_d"].notna().astype(float)}
    g = pd.DataFrame(cols).fillna(0.0)
    g["permno"], g["month"] = d["permno"].to_numpy(), d["month"].to_numpy()
    out = g.groupby(["permno", "month"]).sum()
    pr = d.groupby(["permno", "month"])["padj"]
    out["pmax"] = pr.max()
    last = d.dropna(subset=["padj"]).sort_values("date").groupby(["permno", "month"])["padj"].last()
    out["plast"] = last
    return out.reset_index()


def _ivol(row_sums: pd.DataFrame) -> np.ndarray:
    """Residual std of y on [1, x, s, h] within the month from the sums (NaN when < 15 days or singular)."""
    S = row_sums
    n = S["n3"].to_numpy()
    A = np.zeros((len(S), 4, 4))
    A[:, 0, 0] = n
    A[:, 0, 1] = A[:, 1, 0] = S["s1"]
    A[:, 0, 2] = A[:, 2, 0] = S["s2"]
    A[:, 0, 3] = A[:, 3, 0] = S["s3"]
    A[:, 1, 1] = S["a11"]
    A[:, 1, 2] = A[:, 2, 1] = S["a12"]
    A[:, 1, 3] = A[:, 3, 1] = S["a13"]
    A[:, 2, 2] = S["a22"]
    A[:, 2, 3] = A[:, 3, 2] = S["a23"]
    A[:, 3, 3] = S["a33"]
    b = np.stack([S["b0"], S["b1"], S["b2"], S["b3"]], axis=1)
    out = np.full(len(S), np.nan)
    ok = n >= 15
    det = np.linalg.det(A[ok])
    good = np.zeros(len(S), dtype=bool)
    good[np.flatnonzero(ok)[np.abs(det) > 1e-18]] = True
    coef = np.linalg.solve(A[good], b[good][..., None])[..., 0]
    rss = S["yy3"].to_numpy()[good] - (coef * b[good]).sum(1)
    out[good] = np.sqrt(np.maximum(rss, 0) / (n[good] - 4))
    return out


def build(raw_dir=None, factors: pd.DataFrame | None = None, log=print) -> pd.DataFrame:
    """Monthly features for every permno-month with daily data."""
    from alphacomb.data import french
    fac = factors if factors is not None else french.daily_factors()
    parts = []
    for f in crsp.dsf_files(raw_dir):
        d = pd.read_parquet(f, columns=["permno", "date", "ret", "prc", "vol", "cfacpr"])
        parts.append(month_sums(d, fac))
        log(f"daily sums: {f.name}")
    S = pd.concat(parts, ignore_index=True).sort_values(["permno", "month"]).reset_index(drop=True)
    S["ivol"] = _ivol(S)
    months = pd.DatetimeIndex(np.unique(S["month"]), name="month")
    wide = {c: S.pivot(index="month", columns="permno", values=c).reindex(months)
            for c in ("n", "sx", "sy", "sxx", "sxy", "nr", "sr", "srr", "pmax", "plast", "dv", "ndv")}
    r12 = {c: wide[c].rolling(12, min_periods=1).sum() for c in ("n", "sx", "sy", "sxx", "sxy")}
    n12 = r12["n"]
    cov = r12["sxy"] - r12["sx"] * r12["sy"] / n12
    var = r12["sxx"] - r12["sx"] ** 2 / n12
    beta = (cov / var).where((n12 >= 120) & (var > 0))
    r3 = {c: wide[c].rolling(3, min_periods=1).sum() for c in ("nr", "sr", "srr", "dv", "ndv")}
    sig = np.sqrt(((r3["srr"] - r3["sr"] ** 2 / r3["nr"]) / (r3["nr"] - 1)).clip(lower=0)).where(r3["nr"] >= 40)
    dvol = (r3["dv"] / r3["ndv"]).where(r3["ndv"] >= 20)
    hi = wide["pmax"].rolling(12, min_periods=6).max()
    high52 = (wide["plast"] / hi).where(hi > 0)

    def long(w, name):
        return w.stack().rename(name)

    feats = pd.concat([long(beta, "beta"), long(sig, "sigma"), long(dvol, "dvol"), long(high52, "high52"),
                       long(n12, "beta_days")], axis=1)
    feats.index.names = ["month", "permno"]
    feats = feats.reset_index()
    feats = feats[feats[["beta", "sigma", "dvol", "high52"]].notna().any(axis=1)]
    feats = feats.merge(S[["permno", "month", "ivol", "n3"]].rename(columns={"n3": "ivol_days"}),
                        on=["permno", "month"], how="outer")
    return feats.sort_values(["permno", "month"]).reset_index(drop=True)
