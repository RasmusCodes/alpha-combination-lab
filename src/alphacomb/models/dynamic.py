"""Methods whose weights move every month as outcomes arrive.

DMA         Dynamic model averaging (Raftery, Karny & Ettler 2010; Koop & Korobilis 2012) adapted to panels: one
            time-varying-parameter regression of the target on the signals for every non-empty subset of signal
            FAMILIES. Coefficients follow random walks (forgetting factor lambda), model probabilities are
            discounted (forgetting factor alpha) and updated with each month's predictive likelihood. The month's
            cross-section enters through its moments scaled to the effective sample size (rows / design effect),
            because rows within a month share common shocks. lambda = alpha = 0.99 are the literature defaults.
ONLINE_RLS  Recursive least squares with exponential forgetting (an online regression; forget = 1 is expanding OLS
            updated monthly).
ONLINE_EG   Multiplicative weights (Hedge / exponentiated gradient on gains) over the signals as experts:
            w_k ~ exp(eta * cumulative rank IC_k). eta = 0 is equal weighting; large eta follows recent winners.
"""
from __future__ import annotations

from itertools import combinations
from math import log, pi

import numpy as np

from alphacomb.models.base import Combiner, Moments, TrainData
from alphacomb.models.linear import ridge_moments


class DMA(Combiner):
    name, label, kind = "DMA", "Dynamic model averaging", "dynamic"
    online = True

    def __init__(self, lam: float = 0.99, alpha: float = 0.99, p0: float = 0.01, sigma2: float = 1.0, **kw):
        super().__init__(lam=lam, alpha=alpha, p0=p0, sigma2=sigma2, **kw)

    def _models(self, panel):
        fam = list(panel.family_index().items())
        models = []
        for r in range(1, len(fam) + 1):
            for subset in combinations(range(len(fam)), r):
                idx = sorted(i for j in subset for i in fam[j][1])
                models.append(np.array(idx))
        return models

    def fit(self, data: TrainData):
        p = self.params
        self.deff = data.deff()
        self.models = self._models(data.panel)
        self.state = [(np.zeros(len(m)), p["p0"] * np.eye(len(m))) for m in self.models]
        self.logpi = np.full(len(self.models), -log(len(self.models)))
        self.K = data.K
        for t in np.sort(data.months):
            self._step(int(t), data.moments)
        return self

    def _step(self, t: int, mom: Moments):
        if mom.n[t] <= 0:
            return
        p, d = self.params, self.deff
        lam, s2 = p["lam"], p["sigma2"]
        G, c, yy, n = mom.G[t] / d, mom.c[t] / d, mom.yy[t] / d, mom.n[t] / d
        prior = p["alpha"] * self.logpi
        prior -= np.logaddexp.reduce(prior)
        ll = np.empty(len(self.models))
        new_state = []
        for i, idx in enumerate(self.models):
            b, P = self.state[i]
            P = P / lam
            Gm, cm = G[np.ix_(idx, idx)], c[idx]
            Pinv = np.linalg.inv(P)
            A = Pinv + Gm / s2
            ee = yy - 2 * b @ cm + b @ Gm @ b
            xe = (cm - Gm @ b) / s2
            Ainv = np.linalg.inv(A)
            _, ld_p = np.linalg.slogdet(P)
            _, ld_a = np.linalg.slogdet(A)
            logdet = n * log(s2) + ld_p + ld_a
            quad = ee / s2 - xe @ Ainv @ xe
            ll[i] = -0.5 * (n * log(2 * pi) + logdet + quad)
            new_state.append((Ainv @ (Pinv @ b + cm / s2), Ainv))
        self.state = new_state
        post = prior + ll
        self.logpi = post - np.logaddexp.reduce(post)
        self.last_t = t

    def _beta(self):
        prior = self.params["alpha"] * self.logpi
        w = np.exp(prior - np.logaddexp.reduce(prior))
        beta = np.zeros(self.K)
        for wi, idx, (b, _) in zip(w, self.models, self.state):
            beta[idx] += wi * b
        return beta, w

    def predict(self, X, ctx=None):
        self.beta, _ = self._beta()
        return X @ self.beta

    def update(self, t, panel, moments):
        self._step(int(t), moments)

    def family_probabilities(self, panel) -> dict[str, float]:
        """Posterior inclusion probability of each family under the current model weights."""
        _, w = self._beta()
        fam = panel.family_index()
        out = {}
        for f, idx in fam.items():
            out[f] = float(sum(wi for wi, m in zip(w, self.models) if set(idx) <= set(m.tolist())))
        return out

    def weights(self):
        b, _ = self._beta()
        return b


class OnlineRLS(Combiner):
    name, label, kind = "ONLINE_RLS", "Online least squares (forgetting)", "dynamic"
    online = True
    grid = [{"forget": f} for f in (1.0, 0.995, 0.99, 0.98, 0.95)]

    def __init__(self, forget: float = 0.99, ridge: float = 1e-4, **kw):
        super().__init__(forget=forget, ridge=ridge, **kw)

    def fit(self, data: TrainData):
        K = data.K
        self.Gs, self.cs, self.ns = np.zeros((K, K)), np.zeros(K), 0.0
        for t in np.sort(data.months):
            self._step(int(t), data.moments)
        self._solve()
        return self

    def _step(self, t, mom):
        f = self.params["forget"]
        self.Gs = f * self.Gs + mom.G[t]
        self.cs = f * self.cs + mom.c[t]
        self.ns = f * self.ns + mom.n[t]

    def _solve(self):
        self.beta = ridge_moments(self.Gs, self.cs, max(self.ns, 1.0), self.params["ridge"])

    def update(self, t, panel, moments):
        self._step(int(t), moments)
        self._solve()

    def complexity(self):
        f = self.params["forget"]
        return float("inf") if f >= 1 else 1.0 / (1.0 - f)     # memory length in months


class OnlineHedge(Combiner):
    name, label, kind = "ONLINE_EG", "Online multiplicative weights", "dynamic"
    online = True
    grid = [{"eta": e} for e in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)]

    def __init__(self, eta: float = 0.2, **kw):
        super().__init__(eta=eta, **kw)

    def fit(self, data: TrainData):
        ic = data.moments.ic[np.sort(data.months)]
        self.cum = np.nansum(np.where(np.isfinite(ic), ic, 0.0), axis=0)
        self._solve()
        return self

    def _solve(self):
        z = self.params["eta"] * (self.cum - self.cum.max())
        w = np.exp(z)
        self.beta = w / w.sum()

    def update(self, t, panel, moments):
        ic = moments.ic[int(t)]
        self.cum = self.cum + np.where(np.isfinite(ic), ic, 0.0)
        self._solve()

    def complexity(self):
        return float(self.params["eta"])
