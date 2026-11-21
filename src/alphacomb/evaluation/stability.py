"""Stability diagnostics: do weights, selected hyperparameters and method rankings hold still?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def weight_stability(W: np.ndarray, months: np.ndarray, every: int = 12) -> dict:
    """W: (T, K) weights by month. Compares the weight vectors at successive refits (every `every` months):
    mean correlation, mean L1 change of the normalized vectors, and how often a weight changes sign."""
    idx = [t for t in months[::every] if np.isfinite(W[t]).all()]
    if len(idx) < 3:
        return {"n_refits": len(idx)}
    V = np.array([W[t] / np.abs(W[t]).sum() if np.abs(W[t]).sum() > 0 else W[t] for t in idx])
    cors = [np.corrcoef(V[i], V[i + 1])[0, 1] for i in range(len(V) - 1)]
    l1 = [np.abs(V[i + 1] - V[i]).sum() for i in range(len(V) - 1)]
    flips = [(np.sign(V[i + 1]) != np.sign(V[i])) & (V[i] != 0) for i in range(len(V) - 1)]
    return {"n_refits": len(idx), "mean_corr": float(np.nanmean(cors)), "mean_l1_change": float(np.mean(l1)),
            "sign_flip_rate": float(np.mean(flips))}


def monthly_weight_turnover(W: np.ndarray, months: np.ndarray) -> float:
    V = W[months]
    ok = np.isfinite(V).all(1)
    V = V[ok]
    if len(V) < 3:
        return np.nan
    s = np.abs(V).sum(1, keepdims=True)
    V = V / np.where(s > 0, s, 1)
    return float(np.mean(np.abs(np.diff(V, axis=0)).sum(1)))


def selection_stability(selections: list[dict]) -> dict:
    """How often the nested-CV choice changes between refits, and how concentrated the choices are."""
    keys = [tuple(sorted(s["params"].items())) for s in selections if s.get("params")]
    if len(keys) < 2:
        return {"n_refits": len(keys)}
    changes = float(np.mean([a != b for a, b in zip(keys[:-1], keys[1:])]))
    counts = pd.Series([str(k) for k in keys]).value_counts(normalize=True)
    entropy = float(-(counts * np.log(counts)).sum())
    return {"n_refits": len(keys), "change_rate": changes, "n_distinct": int(len(counts)),
            "modal_share": float(counts.iloc[0]), "modal": counts.index[0], "entropy": entropy}


def rank_stability(perf: pd.DataFrame) -> dict:
    """perf: rows = subperiods, columns = methods (higher = better). Mean pairwise Kendall tau between the method
    rankings of different subperiods and Kendall's W (coefficient of concordance)."""
    P = perf.dropna(axis=1)
    taus = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            taus.append(kendalltau(P.iloc[i], P.iloc[j]).statistic)
    R = P.rank(axis=1).to_numpy()
    m, n = R.shape
    S = ((R.sum(0) - R.sum(0).mean()) ** 2).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n)) if n > 1 else np.nan
    return {"mean_tau": float(np.mean(taus)) if taus else np.nan, "kendall_w": float(W)}
