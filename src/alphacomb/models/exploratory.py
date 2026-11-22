"""Exploratory variants used to explain the confirmatory results (NOT part of the pre-registered method set; runs
are logged in the registry as 'exploratory' and counted as trials).

MINVAR       minimum-variance mix of the signal portfolios, w ~ Sigma^-1 1 (Ledoit-Wolf): the covariance part of
             MVO with every signal assumed to earn the same
MVO_SAMPLE   MVO with the sample covariance (no shrinkage)
RISKPAR      inverse-volatility weights on the signal portfolios, w ~ 1 / sd_k (no correlations, no means)
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from alphacomb.models.base import Combiner, TrainData


def _fret(data: TrainData):
    f = data.moments.fret[data.months]
    return f[np.isfinite(f).all(1)]


class MinVar(Combiner):
    name, label, kind = "MINVAR", "Minimum-variance signal mix", "linear"

    def fit(self, data: TrainData):
        f = _fret(data)
        S = LedoitWolf().fit(f).covariance_
        b = np.linalg.solve(S, np.ones(data.K))
        self.beta = b / np.abs(b).sum()
        return self


class MVOSample(Combiner):
    name, label, kind = "MVO_SAMPLE", "Signal mean-variance, sample covariance", "linear"

    def fit(self, data: TrainData):
        f = _fret(data)
        b = np.linalg.solve(np.cov(f.T, ddof=1), f.mean(0))
        self.beta = b / np.abs(b).sum()
        return self


class RiskParity(Combiner):
    name, label, kind = "RISKPAR", "Inverse-volatility signal weights", "simple"

    def fit(self, data: TrainData):
        sd = _fret(data).std(0, ddof=1)
        b = 1.0 / sd
        self.beta = b / b.sum()
        return self


EXPLORATORY = {c.name: c for c in (MinVar, MVOSample, RiskParity)}
