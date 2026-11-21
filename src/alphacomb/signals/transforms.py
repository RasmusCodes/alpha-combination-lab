"""Cross-sectional transforms. Every function works within one formation month at a time (groups given by offsets
into month-sorted arrays), so no statistic ever crosses dates: a transform cannot leak future information.

offsets: int array of length T+1; rows offsets[t]:offsets[t+1] belong to month t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def offsets_from_sorted(keys: np.ndarray) -> np.ndarray:
    """Group boundaries of a sorted key array (e.g. month index per row)."""
    keys = np.asarray(keys)
    if len(keys) == 0:
        return np.array([0], dtype=np.int64)
    if (keys[1:] < keys[:-1]).any():
        raise ValueError("keys must be sorted")
    change = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    return np.concatenate([[0], change, [len(keys)]]).astype(np.int64)


def _groups(offsets):
    for t in range(len(offsets) - 1):
        yield t, offsets[t], offsets[t + 1]


def winsorize(x: np.ndarray, offsets: np.ndarray, q: float = 0.01) -> np.ndarray:
    """Clip each month's values at its q and 1-q quantiles (NaN ignored and kept)."""
    out = np.array(x, dtype=float, copy=True)
    if q <= 0:
        return out
    for _, a, b in _groups(offsets):
        v = out[a:b]
        ok = np.isfinite(v)
        if ok.sum() < 3:
            continue
        lo, hi = np.quantile(v[ok], [q, 1 - q])
        v[ok] = np.clip(v[ok], lo, hi)
    return out


def zscore(x: np.ndarray, offsets: np.ndarray, min_n: int = 5) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for _, a, b in _groups(offsets):
        v = x[a:b]
        ok = np.isfinite(v)
        if ok.sum() < min_n:
            continue
        m, s = v[ok].mean(), v[ok].std()
        if s > 0:
            out[a:b][ok] = (v[ok] - m) / s
    return out


def rank_uniform(x: np.ndarray, offsets: np.ndarray, min_n: int = 5) -> np.ndarray:
    """Average ranks mapped to (0, 1): (rank - 0.5) / n within each month; NaN stays NaN."""
    out = np.full(len(x), np.nan)
    for _, a, b in _groups(offsets):
        v = x[a:b]
        ok = np.isfinite(v)
        n = int(ok.sum())
        if n < min_n:
            continue
        out[a:b][ok] = (rankdata(v[ok], method="average") - 0.5) / n
    return out


def rank_gauss(x: np.ndarray, offsets: np.ndarray, min_n: int = 5) -> np.ndarray:
    """Normal scores of within-month ranks, re-standardized to mean 0 / variance 1 (robust to outliers)."""
    u = rank_uniform(x, offsets, min_n)
    g = ndtri(u)
    return zscore(g, offsets, min_n)


def demean_within(x: np.ndarray, offsets: np.ndarray, groups: np.ndarray, min_n: int = 2) -> np.ndarray:
    """Subtract the (month, group) mean (e.g. industry neutralization); groups with fewer than min_n values -> 0."""
    out = np.array(x, dtype=float, copy=True)
    g = np.asarray(groups)
    for _, a, b in _groups(offsets):
        v, gg = out[a:b], g[a:b]
        ok = np.isfinite(v)
        for k in np.unique(gg[ok]):
            sel = ok & (gg == k)
            if sel.sum() >= min_n:
                v[sel] -= v[sel].mean()
            else:
                v[sel] = 0.0
    return out


def fill_missing(x: np.ndarray, how: str = "zero") -> np.ndarray:
    """After standardization, 0 is the cross-sectional mean: 'zero' treats a missing signal as neutral."""
    if how == "zero":
        return np.where(np.isfinite(x), x, 0.0)
    if how == "keep":
        return np.asarray(x, dtype=float)
    raise ValueError(f"unknown missing-data rule {how!r}")


@dataclass(frozen=True)
class Normalization:
    """The pre-specified signal pipeline: winsorize -> standardize -> (orient) -> fill missing."""
    winsor: float = 0.01
    method: str = "rank_gauss"          # 'rank_gauss' | 'zscore' | 'rank'
    missing: str = "zero"
    min_n: int = 20                     # fewer valid values in a month -> the signal is missing that month

    def apply(self, x: np.ndarray, offsets: np.ndarray, direction: int = 1) -> np.ndarray:
        v = winsorize(x, offsets, self.winsor) if self.method == "zscore" else np.asarray(x, dtype=float)
        if self.method == "zscore":
            v = zscore(v, offsets, self.min_n)
        elif self.method == "rank_gauss":
            v = rank_gauss(v, offsets, self.min_n)
        elif self.method == "rank":
            v = rank_uniform(v, offsets, self.min_n) - 0.5
            v = zscore(v, offsets, self.min_n)
        else:
            raise ValueError(f"unknown standardization {self.method!r}")
        return fill_missing(direction * v, self.missing)
