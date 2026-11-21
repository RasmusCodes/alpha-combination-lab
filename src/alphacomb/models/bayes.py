"""Bayesian combinations.

BMA      Bayesian model averaging over every subset of signals (2^K linear models) with Zellner's g-prior
         (g = effective sample size, the unit-information prior) and a beta-binomial(1, 1) prior on model size
         (Ley & Steel 2009), so every model size is equally likely a priori. Pooled panel rows are not independent
         (common monthly shocks), so the sample size in the marginal likelihood is the EFFECTIVE one: rows divided
         by the design effect estimated from Fama-MacBeth vs pooled-OLS coefficient variances (TrainData.deff).
BSHRINK  Empirical-Bayes shrinkage of the Fama-MacBeth slopes toward EQUAL WEIGHTS: prior beta ~ N(kappa * 1,
         tau^2 I); kappa and tau^2 maximize the marginal likelihood of the average slope vector, whose sampling
         covariance is estimated from the slopes' time series. Weak evidence -> equal weights; strong evidence ->
         the regression weights.
"""
from __future__ import annotations

from itertools import combinations
from math import comb, log

import numpy as np

from alphacomb.models.base import Combiner, TrainData


class BMA(Combiner):
    name, label, kind = "BMA", "Bayesian model averaging", "bayes"

    def __init__(self, max_k: int = 16, **kw):
        super().__init__(max_k=max_k, **kw)

    def fit(self, data: TrainData):
        G, c, yy, n = data.sums()
        K = len(c)
        if K > self.params["max_k"]:
            raise ValueError(f"BMA enumerates 2^K models; K={K} exceeds max_k={self.params['max_k']}")
        ne = max(n / data.deff(), K + 2.0)
        g = ne
        logpost, betas, incl = [], [], []
        for p in range(0, K + 1):
            prior = -log(K + 1) - log(comb(K, p))
            if p == 0:
                logpost.append(np.array([prior + 0.0]))
                betas.append(np.zeros((1, K)))
                incl.append(np.zeros((1, K), dtype=bool))
                continue
            S = np.array(list(combinations(range(K), p)))
            Gs = G[S[:, :, None], S[:, None, :]]
            cs = c[S]
            try:
                bs = np.linalg.solve(Gs, cs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                bs = np.stack([np.linalg.lstsq(a, b, rcond=None)[0] for a, b in zip(Gs, cs)])
            r2 = np.clip((bs * cs).sum(1) / yy, 0.0, 1.0 - 1e-12)
            lml = 0.5 * (ne - 1 - p) * np.log1p(g) - 0.5 * (ne - 1) * np.log1p(g * (1 - r2))
            logpost.append(lml + prior)
            full = np.zeros((len(S), K))
            np.put_along_axis(full, S, bs * (g / (1 + g)), axis=1)
            betas.append(full)
            m = np.zeros((len(S), K), dtype=bool)
            np.put_along_axis(m, S, True, axis=1)
            incl.append(m)
        lp = np.concatenate(logpost)
        w = np.exp(lp - lp.max())
        w /= w.sum()
        B, M = np.vstack(betas), np.vstack(incl)
        self.beta = w @ B
        self.inclusion = w @ M
        self.expected_size = float(w @ M.sum(1))
        self.top_model_prob = float(w.max())
        self.n_eff = ne
        return self

    def complexity(self):
        return self.expected_size


class BayesShrink(Combiner):
    name, label, kind = "BSHRINK", "Bayesian shrinkage to equal weights", "bayes"
    TAU2_GRID = np.concatenate([[0.0], np.logspace(-9, -1, 65)])

    def fit(self, data: TrainData):
        B = data.moments.fm[data.months]
        B = B[np.isfinite(B).all(1)]
        K = data.K
        one = np.ones(K)
        if len(B) < K + 12:
            self.beta = one / K
            self.tau2 = 0.0
            return self
        T = len(B)
        bbar = B.mean(0)
        S = np.cov(B.T, ddof=1) / T
        best = (-np.inf, None)
        for tau2 in self.TAU2_GRID:
            V = S + tau2 * np.eye(K)
            sign, logdet = np.linalg.slogdet(V)
            if sign <= 0:
                continue
            W = np.linalg.inv(V)
            kappa = (one @ W @ bbar) / (one @ W @ one)
            r = bbar - kappa * one
            ll = -0.5 * (logdet + r @ W @ r)
            if ll > best[0]:
                best = (ll, (tau2, kappa, W, r))
        tau2, kappa, W, r = best[1]
        self.beta = kappa * one + tau2 * (W @ r)
        self.tau2, self.kappa = float(tau2), float(kappa)
        self.shrink_df = float(1.0 + np.trace(tau2 * W))
        return self

    def complexity(self):
        return getattr(self, "shrink_df", 1.0)
