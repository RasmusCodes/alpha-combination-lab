"""Model library: every combination method behind the Combiner interface, addressable by name."""
from __future__ import annotations

from alphacomb.models.base import Combiner, Moments, MonthContext, TrainData
from alphacomb.models.bayes import BMA, BayesShrink
from alphacomb.models.dynamic import DMA, OnlineHedge, OnlineRLS
from alphacomb.models.linear import (OLS, PCR, ElasticNet, EqualWeight, HierarchicalEqualWeight, ICWeighted, Lasso,
                                     Ridge, SignalMVO)
from alphacomb.models.regime import RegimeEnsemble
from alphacomb.models.trees import GradientBoosting, RandomForest

METHODS: dict[str, type[Combiner]] = {c.name: c for c in (
    EqualWeight, HierarchicalEqualWeight, ICWeighted, OLS, Ridge, Lasso, ElasticNet, PCR, BMA, BayesShrink, SignalMVO,
    RandomForest, GradientBoosting, DMA, RegimeEnsemble, OnlineHedge, OnlineRLS)}

# Pre-specified order for tables: simple -> linear -> Bayesian -> trees -> dynamic
ORDER = ["EW", "HEW", "ICW", "OLS", "RIDGE", "LASSO", "ENET", "PCR", "MVO", "BMA", "BSHRINK", "RF", "GBM", "DMA",
         "REGIME", "ONLINE_EG", "ONLINE_RLS"]


def make(name: str, **params) -> Combiner:
    try:
        cls = METHODS[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; known: {sorted(METHODS)}") from None
    return cls(**params)


__all__ = ["Combiner", "Moments", "MonthContext", "TrainData", "METHODS", "ORDER", "make"]
