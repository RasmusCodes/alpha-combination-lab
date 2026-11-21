"""Predictive metrics, computed month by month on the cross-section.

IC        Pearson correlation of the forecast with the next-month return
rank IC   Spearman correlation (Pearson of within-month ranks): the primary predictive metric
R2_oos    out-of-sample R^2 of the rank-gauss target, pooled over rows: 1 - SSE / sum(y^2)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else np.nan


def monthly_ic(pred: np.ndarray, ret: np.ndarray, offsets: np.ndarray, months=None, rank: bool = True,
               min_n: int = 20) -> np.ndarray:
    """(T,) array of within-month correlations between pred and ret (NaN where fewer than min_n valid pairs or the
    month is not requested)."""
    T = len(offsets) - 1
    out = np.full(T, np.nan)
    it = range(T) if months is None else months
    for t in it:
        a, b = offsets[t], offsets[t + 1]
        p, r = pred[a:b], ret[a:b]
        ok = np.isfinite(p) & np.isfinite(r)
        if ok.sum() < min_n:
            continue
        p, r = p[ok], r[ok]
        if rank:
            p, r = rankdata(p), rankdata(r)
        out[t] = _corr(p, r)
    return out


def ic_summary(ic: np.ndarray, nw_lags: int | None = None) -> dict:
    from alphacomb.evaluation.stats import nw_tstat
    x = ic[np.isfinite(ic)]
    if len(x) < 3:
        return {"mean": np.nan, "sd": np.nan, "ir": np.nan, "t": np.nan, "hit": np.nan, "n": len(x)}
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "ir": float(x.mean() / x.std(ddof=1)),
            "t": float(nw_tstat(x, nw_lags)), "hit": float((x > 0).mean()), "n": int(len(x))}


def r2_oos(y: np.ndarray, yhat: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(yhat)
    if ok.sum() == 0:
        return np.nan
    return float(1.0 - np.sum((y[ok] - yhat[ok]) ** 2) / np.sum(y[ok] ** 2))
