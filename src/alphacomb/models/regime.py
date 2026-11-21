"""Regime-conditioned ensemble: a separate set of signal weights for each ex-ante market regime, shrunk toward
the pooled (all-regime) weights.

    beta_r = argmin  (1/n_r) sum_{t in r} ||y_t - X_t beta||^2 + kappa ||beta - beta_pool||^2

Large kappa -> one set of weights for every regime (the pooled regression); small kappa -> fully regime-specific
weights. The regime label of each formation month must be known at that month's close (month_info['regime']);
the study uses market volatility (above / below its expanding median) crossed with market state (trailing 24-month
market return up / down).
"""
from __future__ import annotations

import numpy as np

from alphacomb.models.base import Combiner, MonthContext, TrainData
from alphacomb.models.linear import ridge_moments


class RegimeEnsemble(Combiner):
    name, label, kind = "REGIME", "Regime-conditioned ensemble", "dynamic"
    grid = [{"kappa": k} for k in (10.0, 1.0, 0.3, 0.1, 0.03, 0.01)]

    def __init__(self, kappa: float = 0.1, column: str = "regime", **kw):
        super().__init__(kappa=kappa, column=column, **kw)

    def fit(self, data: TrainData):
        G, c, _, n = data.sums()
        self.pool = ridge_moments(G, c, n, 1e-8)
        mi = data.month_info()
        self.by_regime = {}
        if mi is None or self.params["column"] not in mi:
            self.beta = self.pool
            return self
        labels = mi[self.params["column"]].to_numpy()
        kappa = self.params["kappa"]
        K = data.K
        for r in np.unique(labels[~pd_isna(labels)]):
            m = data.months[labels == r]
            if len(m) == 0:
                continue
            Gr, cr, _, nr = data.moments.sums(m)
            if nr <= 0:
                continue
            A = Gr / nr + kappa * np.eye(K)
            self.by_regime[r] = np.linalg.solve(A, cr / nr + kappa * self.pool)
        self.beta = self.pool
        return self

    def _beta_for(self, ctx: MonthContext | None):
        if ctx is None:
            return self.pool
        r = ctx.info.get(self.params["column"])
        return self.by_regime.get(r, self.pool)

    def predict(self, X, ctx=None):
        self.beta = self._beta_for(ctx)
        return X @ self.beta

    def complexity(self):
        return float(len(self.by_regime) * len(self.pool) / (1.0 + self.params["kappa"]))


def pd_isna(a):
    import pandas as pd
    return pd.isna(a)
