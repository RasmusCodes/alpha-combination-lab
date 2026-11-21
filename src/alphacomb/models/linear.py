"""Simple and linear combinations, all solved from per-month moments.

Penalty conventions follow scikit-learn (per-observation loss):
    (1 / 2n) ||y - X b||^2 + lam * alpha * ||b||_1 + lam * (1 - alpha) / 2 * ||b||^2
so ridge is alpha = 0 and lasso alpha = 1. Grids are ordered from the most to the least regularized model.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from alphacomb.models.base import Combiner, TrainData


# ----------------------------------------------------------------------------- solvers
def enet_moments(G: np.ndarray, c: np.ndarray, n: float, lam: float, alpha: float,
                 b0: np.ndarray | None = None, tol: float = 1e-12, max_iter: int = 20000) -> np.ndarray:
    """Coordinate descent for the elastic net on sufficient statistics (X'X, X'y, n)."""
    K = len(c)
    Gn, cn = G / n, c / n
    diag = np.diag(Gn).copy()
    b = np.zeros(K) if b0 is None else np.array(b0, dtype=float)
    l1, l2 = lam * alpha, lam * (1.0 - alpha)
    for _ in range(max_iter):
        maxd = 0.0
        for j in range(K):
            if diag[j] <= 0:
                b[j] = 0.0
                continue
            rj = cn[j] - Gn[j] @ b + diag[j] * b[j]
            new = np.sign(rj) * max(abs(rj) - l1, 0.0) / (diag[j] + l2)
            d = abs(new - b[j])
            maxd = max(maxd, d)
            b[j] = new
        if maxd < tol:
            break
    return b


def ridge_moments(G, c, n, lam) -> np.ndarray:
    K = len(c)
    A = G / n + lam * np.eye(K)
    try:
        return np.linalg.solve(A, c / n)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, c / n, rcond=None)[0]


# ----------------------------------------------------------------------------- simple benchmarks
class EqualWeight(Combiner):
    name, label, kind = "EW", "Equal weights", "simple"

    def fit(self, data: TrainData):
        self.beta = np.full(data.K, 1.0 / data.K)
        return self

    def complexity(self):
        return 0.0


class HierarchicalEqualWeight(Combiner):
    """Equal weight per FAMILY, then equal within the family: ten correlated value signals count as one view."""
    name, label, kind = "HEW", "Family-equal weights", "simple"

    def fit(self, data: TrainData):
        fam = data.panel.family_index()
        b = np.zeros(data.K)
        for idx in fam.values():
            b[idx] = 1.0 / (len(fam) * len(idx))
        self.beta = b
        return self

    def complexity(self):
        return 0.0


class ICWeighted(Combiner):
    """Weights proportional to each signal's average rank IC over the training months, negatives set to zero."""
    name, label, kind = "ICW", "IC-weighted", "simple"

    def fit(self, data: TrainData):
        ic = np.nanmean(data.moments.ic[data.months], axis=0)
        w = np.where(np.isfinite(ic), np.maximum(ic, 0.0), 0.0)
        self.beta = w / w.sum() if w.sum() > 0 else np.full(data.K, 1.0 / data.K)
        return self

    def complexity(self):
        return float((self.beta > 0).sum())


# ----------------------------------------------------------------------------- regressions
class OLS(Combiner):
    name, label, kind = "OLS", "Linear regression", "linear"

    def fit(self, data: TrainData):
        G, c, _, n = data.sums()
        self.beta = ridge_moments(G, c, n, 1e-10)
        return self

    def complexity(self):
        return float(len(self.beta))


class Ridge(Combiner):
    name, label, kind = "RIDGE", "Ridge", "linear"
    grid = [{"lam": v} for v in (30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01)]

    def __init__(self, lam: float = 1.0, **kw):
        super().__init__(lam=lam, **kw)

    def fit(self, data: TrainData):
        G, c, _, n = data.sums()
        self.beta = ridge_moments(G, c, n, self.params["lam"])
        self._d = np.linalg.eigvalsh(G / n)
        return self

    def complexity(self):
        lam = self.params["lam"]
        return float(np.sum(self._d / (self._d + lam)))


class ElasticNet(Combiner):
    """Elastic net; lam is given as a fraction of lam_max (the smallest penalty that zeroes every coefficient)."""
    name, label, kind = "ENET", "Elastic net", "linear"
    grid = [{"frac": f, "alpha": a} for f in (0.6, 0.3, 0.1, 0.03, 0.01) for a in (0.9, 0.5, 0.1)]

    def __init__(self, frac: float = 0.1, alpha: float = 0.5, **kw):
        super().__init__(frac=frac, alpha=alpha, **kw)

    def fit(self, data: TrainData):
        G, c, _, n = data.sums()
        a = self.params["alpha"]
        lam_max = np.max(np.abs(c / n)) / max(a, 1e-3)
        self.beta = enet_moments(G, c, n, self.params["frac"] * lam_max, a)
        return self

    def complexity(self):
        return float(np.sum(np.abs(self.beta) > 0))


class Lasso(ElasticNet):
    name, label, kind = "LASSO", "Lasso", "linear"
    grid = [{"frac": f, "alpha": 1.0} for f in (0.9, 0.6, 0.3, 0.1, 0.03, 0.01)]

    def __init__(self, frac: float = 0.1, alpha: float = 1.0, **kw):
        super().__init__(frac=frac, alpha=1.0, **kw)


class PCR(Combiner):
    """Principal-component regression: regress the target on the first k principal components of the signals."""
    name, label, kind = "PCR", "Principal components", "linear"
    grid = [{"k": k} for k in (1, 2, 3, 4, 6, 8, 12, 16)]

    def __init__(self, k: int = 3, **kw):
        super().__init__(k=k, **kw)

    def fit(self, data: TrainData):
        G, c, _, n = data.sums()
        R = G / n
        evals, evecs = np.linalg.eigh(R)
        order = np.argsort(evals)[::-1]
        k = int(min(self.params["k"], data.K))
        V, lam = evecs[:, order[:k]], evals[order[:k]]
        gamma = (V.T @ (c / n)) / np.maximum(lam, 1e-12)
        self.beta = V @ gamma
        self.evals = evals[order]
        return self

    def complexity(self):
        return float(min(self.params["k"], len(self.beta)))


class SignalMVO(Combiner):
    """Mean-variance weights on the signals' factor returns, w ~ Sigma^-1 mu (Ledoit-Wolf covariance): the
    Grinold-Kahn view of signal combination as a portfolio of signal portfolios."""
    name, label, kind = "MVO", "Signal mean-variance", "linear"

    def fit(self, data: TrainData):
        f = data.moments.fret[data.months]
        f = f[np.isfinite(f).all(1)]
        if len(f) < 2 * data.K:
            self.beta = np.full(data.K, 1.0 / data.K)
            return self
        mu = f.mean(0)
        S = LedoitWolf().fit(f).covariance_
        self.beta = np.linalg.solve(S, mu)
        self.beta = self.beta / np.abs(self.beta).sum()
        return self

    def complexity(self):
        return float(len(self.beta))
