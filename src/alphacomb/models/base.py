"""Common interface of every combination method, and the per-month sufficient statistics they share.

A Combiner turns the K signals of one month into one score per instrument. Batch methods are fit on a training set
of formation months (TrainData) and predict until the next refit; online methods additionally see every month's
outcome as soon as it is known (update). Linear methods work from per-month moments (X'X, X'y, ...), so a refit or a
cross-validation fold costs a sum over months; row-level learners (trees) take the rows themselves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphacomb.panel import Panel


@dataclass
class Moments:
    """Per-month statistics over rows with a known target (finite y)."""
    G: np.ndarray        # (T, K, K) X'X
    c: np.ndarray        # (T, K)    X'y
    yy: np.ndarray       # (T,)      y'y
    n: np.ndarray        # (T,)      rows
    fm: np.ndarray       # (T, K)    cross-sectional multivariate OLS slopes (Fama-MacBeth), NaN if not estimable
    ic: np.ndarray       # (T, K)    corr(signal, y): with rank-gauss y and signals, a rank IC
    fret: np.ndarray     # (T, K)    signal "factor return": mean(x_k * (ret - mean ret))

    @classmethod
    def from_panel(cls, p: Panel) -> "Moments":
        T, K = p.T, p.K
        G, c, yy, n = np.zeros((T, K, K)), np.zeros((T, K)), np.zeros(T), np.zeros(T)
        fm, ic, fret = np.full((T, K), np.nan), np.full((T, K), np.nan), np.full((T, K), np.nan)
        for t in range(T):
            r = p.rows(t, t + 1)
            X, y, ret = p.X[r], p.y[r], p.ret[r]
            ok = np.isfinite(y)
            if ok.sum() < K + 5:
                continue
            Xo, yo = X[ok], y[ok]
            G[t], c[t], yy[t], n[t] = Xo.T @ Xo, Xo.T @ yo, yo @ yo, ok.sum()
            sx = Xo.std(axis=0)
            good = sx > 0
            ic[t, good] = ((Xo[:, good] - Xo[:, good].mean(0)) * (yo - yo.mean())[:, None]).mean(0) / (
                sx[good] * yo.std())
            try:
                if np.linalg.cond(G[t]) < 1e8:
                    fm[t] = np.linalg.solve(G[t], c[t])
            except np.linalg.LinAlgError:
                pass
            okr = np.isfinite(ret)
            if okr.sum() > K:
                rr = ret[okr] - ret[okr].mean()
                fret[t] = (X[okr] * rr[:, None]).mean(0)
        return cls(G=G, c=c, yy=yy, n=n, fm=fm, ic=ic, fret=fret)

    def sums(self, months: np.ndarray):
        m = np.asarray(months, dtype=int)
        return self.G[m].sum(0), self.c[m].sum(0), float(self.yy[m].sum()), float(self.n[m].sum())


@dataclass
class TrainData:
    """A training set: the panel, its moments and the formation months the model may learn from."""
    panel: Panel
    moments: Moments
    months: np.ndarray
    seed: int = 0
    cache: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return self.panel.K

    def sums(self):
        if "sums" not in self.cache:
            self.cache["sums"] = self.moments.sums(self.months)
        return self.cache["sums"]

    def rows(self, max_rows: int | None = None):
        """Row-level training data (finite y only). Months are contiguous in memory, so this is a few slices.
        max_rows: deterministic subsample (seeded) for learners whose cost grows with rows."""
        idx = np.concatenate([np.arange(self.panel.offsets[t], self.panel.offsets[t + 1]) for t in self.months])
        idx = idx[np.isfinite(self.panel.y[idx])]
        if max_rows is not None and len(idx) > max_rows:
            rng = np.random.default_rng(self.seed)
            idx = np.sort(rng.choice(idx, size=max_rows, replace=False))
        return self.panel.X[idx], self.panel.y[idx], idx

    def month_info(self) -> pd.DataFrame | None:
        mi = self.panel.month_info
        return None if mi is None else mi.iloc[self.months]

    def deff(self) -> float:
        """Design effect of pooling correlated cross-sections: average ratio of the Fama-MacBeth variance of the
        coefficients (which reflects cross-sectional dependence) to the naive pooled-OLS variance. >= 1."""
        if "deff" in self.cache:
            return self.cache["deff"]
        G, c, yy, n = self.sums()
        fm = self.moments.fm[self.months]
        fm = fm[np.isfinite(fm).all(1)]
        out = 1.0
        if len(fm) > 24 and n > 0:
            try:
                b = np.linalg.solve(G, c)
                s2 = max((yy - 2 * b @ c + b @ G @ b) / n, 1e-12)
                v_naive = np.diag(s2 * np.linalg.inv(G))
                v_fm = fm.var(axis=0, ddof=1) / len(fm)
                ratio = v_fm / np.maximum(v_naive, 1e-300)
                out = float(max(1.0, np.median(ratio)))
            except np.linalg.LinAlgError:
                out = 1.0
        self.cache["deff"] = out
        return out


@dataclass
class MonthContext:
    t: int
    date: pd.Timestamp
    info: dict


class Combiner:
    """Base class. Subclasses set name, family_label, grid (hyperparameter configurations ordered from the most to
    the least regularized) and implement fit/predict (and update for online learners)."""

    name = "base"
    label = "Base"
    kind = "linear"                    # 'simple' | 'linear' | 'bayes' | 'tree' | 'dynamic'
    grid: list[dict] = []
    online = False
    needs_rows = False

    def __init__(self, **params):
        self.params = dict(params)
        self.beta: np.ndarray | None = None

    def clone(self, **params) -> "Combiner":
        p = dict(self.params)
        p.update(params)
        return type(self)(**p)

    def fit(self, data: TrainData) -> "Combiner":
        raise NotImplementedError

    def predict(self, X: np.ndarray, ctx: MonthContext | None = None) -> np.ndarray:
        return X @ self.beta

    def update(self, t: int, panel: Panel, moments: Moments) -> None:
        """Online learners: incorporate formation month t (its outcome is now known)."""

    def weights(self) -> np.ndarray | None:
        """Linear-equivalent signal weights (None for non-linear learners)."""
        return None if self.beta is None else np.asarray(self.beta, dtype=float)

    def complexity(self) -> float | None:
        """Effective degrees of freedom where defined."""
        return None

    def describe(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in sorted(self.params.items()))})"
