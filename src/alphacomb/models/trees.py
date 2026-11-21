"""Tree ensembles on the pooled rows (LightGBM, deterministic mode).

RF   random forest: LightGBM's 'rf' boosting (bagged trees fit to the target and averaged), row subsample per tree.
GBM  gradient boosting: shallow trees, learning rate 0.05, stochastic row/feature subsampling, L2 leaf penalty.
     The number of rounds is evaluated in stages from one fit during model selection (predict_staged).

Both learn the rank-gauss target from the K signals, so they can find non-linear shapes and interactions that the
linear methods cannot. Grids are ordered from the most to the least constrained model.
"""
from __future__ import annotations

import numpy as np

from alphacomb.models.base import Combiner, TrainData

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


def _base_params(seed: int, n_jobs: int) -> dict:
    return {"objective": "regression", "verbose": -1, "seed": seed, "deterministic": True,
            "force_row_wise": True, "num_threads": n_jobs}


class RandomForest(Combiner):
    name, label, kind = "RF", "Random forest", "tree"
    needs_rows = True
    grid = [{"depth": d} for d in (2, 4, 6)]

    def __init__(self, depth: int = 4, feature_fraction: float = 0.5, trees: int = 200, min_leaf: int = 100,
                 bagging_fraction: float = 0.1, n_jobs: int = 2, max_rows: int | None = None, seed: int = 0, **kw):
        super().__init__(depth=depth, feature_fraction=feature_fraction, trees=trees, min_leaf=min_leaf,
                         bagging_fraction=bagging_fraction, n_jobs=n_jobs, max_rows=max_rows, seed=seed, **kw)

    def fit(self, data: TrainData):
        p = self.params
        X, y, _ = data.rows(p["max_rows"])
        params = _base_params(p["seed"] + data.seed, p["n_jobs"]) | {
            "boosting": "rf", "num_leaves": 2 ** p["depth"], "max_depth": p["depth"],
            "min_data_in_leaf": p["min_leaf"], "feature_fraction": p["feature_fraction"],
            "bagging_fraction": p["bagging_fraction"], "bagging_freq": 1}
        self.booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=p["trees"])
        self.beta = None
        return self

    def predict(self, X, ctx=None):
        return self.booster.predict(X)

    def complexity(self):
        return float(self.params["depth"])


class GradientBoosting(Combiner):
    name, label, kind = "GBM", "Gradient boosting", "tree"
    needs_rows = True
    staged_param = "rounds"
    grid = [{"depth": d, "rounds": r} for d in (1, 2, 3, 5) for r in (25, 50, 100, 200, 400)]

    def __init__(self, depth: int = 2, rounds: int = 100, learning_rate: float = 0.05, min_leaf: int = 200,
                 l2: float = 10.0, feature_fraction: float = 0.8, bagging_fraction: float = 0.5, n_jobs: int = 2,
                 max_rows: int | None = None, seed: int = 0, **kw):
        super().__init__(depth=depth, rounds=rounds, learning_rate=learning_rate, min_leaf=min_leaf, l2=l2,
                         feature_fraction=feature_fraction, bagging_fraction=bagging_fraction, n_jobs=n_jobs,
                         max_rows=max_rows, seed=seed, **kw)

    def _params(self, seed):
        p = self.params
        return _base_params(seed, p["n_jobs"]) | {
            "boosting": "gbdt", "learning_rate": p["learning_rate"], "num_leaves": 2 ** p["depth"],
            "max_depth": p["depth"], "min_data_in_leaf": p["min_leaf"], "lambda_l2": p["l2"],
            "feature_fraction": p["feature_fraction"], "bagging_fraction": p["bagging_fraction"],
            "bagging_freq": 1}

    def fit(self, data: TrainData, rounds: int | None = None):
        X, y, _ = data.rows(self.params["max_rows"])
        self.booster = lgb.train(self._params(self.params["seed"] + data.seed), lgb.Dataset(X, y),
                                 num_boost_round=int(rounds or self.params["rounds"]))
        self.beta = None
        return self

    def predict(self, X, ctx=None):
        return self.booster.predict(X, num_iteration=int(self.params["rounds"]))

    def predict_staged(self, X, rounds_list):
        return {r: self.booster.predict(X, num_iteration=int(r)) for r in rounds_list}

    def complexity(self):
        return float(self.params["depth"] * self.params["rounds"])
