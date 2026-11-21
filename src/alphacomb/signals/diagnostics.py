"""Signal diagnostics that use no returns: coverage, cross-sectional correlation structure, effective number.

effective number of independent signals
  participation ratio  (sum lambda)^2 / sum lambda^2 of the average correlation matrix's eigenvalues
  exchangeable         K / (1 + (K - 1) rho_bar)
Both equal K for uncorrelated signals and approach 1 as signals become copies of each other. With equal ICs c and
equal pairwise correlation rho, the best linear combination of K signals has IC c * sqrt(K / (1 + (K - 1) rho)):
ten signals with rho = 0.8 forecast only about 10% better than one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def coverage(frame: pd.DataFrame, names: list[str], by: str = "decade") -> pd.DataFrame:
    d = frame[["date"] + names].copy()
    d["key"] = (d["date"].dt.year // 10 * 10).astype(str) + "s" if by == "decade" else d["date"].dt.year
    return d.groupby("key")[names].apply(lambda g: g.notna().mean()).round(3)


def mean_rank_correlation(frame: pd.DataFrame, names: list[str], min_n: int = 30) -> pd.DataFrame:
    """Time-series average of within-month Spearman correlations (pairwise complete)."""
    K = len(names)
    acc, cnt = np.zeros((K, K)), np.zeros((K, K))
    for _, g in frame.groupby("date"):
        X = g[names].to_numpy(float)
        R = np.full_like(X, np.nan)
        for k in range(K):
            ok = np.isfinite(X[:, k])
            if ok.sum() >= min_n:
                R[ok, k] = rankdata(X[ok, k])
        Rm = pd.DataFrame(R).corr(min_periods=min_n).to_numpy()
        good = np.isfinite(Rm)
        acc[good] += Rm[good]
        cnt[good] += 1
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame(out, index=names, columns=names)


def effective_number(C: np.ndarray) -> dict:
    C = np.asarray(C, float)
    K = len(C)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
    rho = (np.nansum(C) - K) / (K * (K - 1))
    return {"K": K, "eigenvalues": lam.tolist(), "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "mean_corr": float(rho), "exchangeable": float(K / (1 + (K - 1) * rho)),
            "first_eigen_share": float(lam[0] / lam.sum())}


def combined_ic(c: float, K: int, rho: float) -> float:
    """IC of the equal-weighted (and, by symmetry, optimal) combination of K signals with equal IC c and equal
    pairwise correlation rho."""
    return c * np.sqrt(K / (1 + (K - 1) * rho))
