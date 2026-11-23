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

from itertools import combinations, islice
from math import comb, log

import numpy as np

from alphacomb.models.base import Combiner, TrainData


class BMA(Combiner):
    name, label, kind = "BMA", "Bayesian model averaging", "bayes"

    def __init__(self, max_k: int = 16, chunk: int = 200_000, **kw):
        super().__init__(max_k=max_k, chunk=chunk, **kw)

    def fit(self, data: TrainData):
        """Exact enumeration of all 2^K models, streamed in chunks with a running log-sum-exp, so memory stays
        bounded for K above 16 (1M models at K = 20, 8.4M at K = 23)."""
        G, c, yy, n = data.sums()
        K = len(c)
        if K > self.params["max_k"]:
            raise ValueError(f"BMA enumerates 2^K models; K={K} exceeds max_k={self.params['max_k']}")
        ne = max(n / data.deff(), K + 2.0)
        g = ne
        shrink = g / (1 + g)
        run_max, Z = -np.inf, 0.0
        Bsum, Isum, Esum, top = np.zeros(K), np.zeros(K), 0.0, -np.inf

        def absorb(lp, bfull, mask):
            nonlocal run_max, Z, Bsum, Isum, Esum, top
            mx = max(run_max, float(lp.max()))
            scale = np.exp(run_max - mx) if np.isfinite(run_max) else 0.0
            w = np.exp(lp - mx)
            Z = Z * scale + w.sum()
            Bsum = Bsum * scale + w @ bfull
            Isum = Isum * scale + w @ mask
            Esum = Esum * scale + float(w @ mask.sum(1))
            run_max = mx
            top = max(top, float(lp.max()))

        for p in range(0, K + 1):
            prior = -log(K + 1) - log(comb(K, p))
            if p == 0:
                absorb(np.array([prior]), np.zeros((1, K)), np.zeros((1, K)))
                continue
            gen = combinations(range(K), p)
            while True:
                block = list(islice(gen, self.params["chunk"]))
                if not block:
                    break
                S = np.array(block)
                Gs = G[S[:, :, None], S[:, None, :]]
                cs = c[S]
                try:
                    bs = np.linalg.solve(Gs, cs[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    bs = np.stack([np.linalg.lstsq(a, b, rcond=None)[0] for a, b in zip(Gs, cs)])
                r2 = np.clip((bs * cs).sum(1) / yy, 0.0, 1.0 - 1e-12)
                lml = 0.5 * (ne - 1 - p) * np.log1p(g) - 0.5 * (ne - 1) * np.log1p(g * (1 - r2))
                full = np.zeros((len(S), K))
                np.put_along_axis(full, S, bs * shrink, axis=1)
                mask = np.zeros((len(S), K))
                np.put_along_axis(mask, S, 1.0, axis=1)
                absorb(lml + prior, full, mask)
        self.beta = Bsum / Z
        self.inclusion = Isum / Z
        self.expected_size = Esum / Z
        self.top_model_prob = float(np.exp(top - run_max) / Z)
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
