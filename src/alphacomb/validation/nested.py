"""Nested model selection: choose a method's hyperparameters using only its training window.

For each configuration in the method's grid, the inner folds (validation blocks inside the training window) are
predicted month by month exactly as the outer walk-forward would predict them (online learners keep learning
through the block), and scored by the mean monthly rank IC. Rule 'best' takes the highest mean; rule '1se' takes
the most regularized configuration (grids are ordered most -> least regularized) within one standard error of it.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from alphacomb.evaluation.metrics import monthly_ic
from alphacomb.models.base import Combiner, Moments, MonthContext, TrainData
from alphacomb.panel import Panel
from alphacomb.validation.splits import Fold


def month_context(panel: Panel, t: int) -> MonthContext:
    info = {} if panel.month_info is None else panel.month_info.iloc[t].to_dict()
    return MonthContext(t=t, date=panel.dates[t], info=info)


def predict_months(model: Combiner, panel: Panel, moments: Moments, months: np.ndarray, trained_through: int,
                   out: np.ndarray | None = None) -> np.ndarray:
    """Row-level predictions for `months` (sorted). Online models first absorb every month up to t-1."""
    out = np.full(panel.n, np.nan) if out is None else out
    seen = trained_through
    for t in np.sort(months):
        if model.online:
            for s in range(seen + 1, t):
                model.update(s, panel, moments)
            seen = max(seen, t - 1)
        r = panel.rows(t, t + 1)
        out[r] = model.predict(panel.X[r], month_context(panel, t))
    return out


def score(pred: np.ndarray, panel: Panel, months: np.ndarray) -> np.ndarray:
    ic = monthly_ic(pred, panel.ret, panel.offsets, months=months)
    return ic[months]


def select(proto: Combiner, panel: Panel, moments: Moments, folds: list[Fold], rule: str = "best",
           seed: int = 0) -> tuple[dict, list[dict]]:
    grid = proto.grid
    if not grid or not folds:
        return {}, []
    staged = getattr(proto, "staged_param", None)
    scores: dict[int, list[np.ndarray]] = defaultdict(list)
    if staged:
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, cfg in enumerate(grid):
            groups[tuple(sorted((k, v) for k, v in cfg.items() if k != staged))].append(i)
        for key, idxs in groups.items():
            stages = sorted({grid[i][staged] for i in idxs})
            for f in folds:
                m = proto.clone(**dict(key), **{staged: max(stages)})
                m.fit(TrainData(panel, moments, f.train, seed=seed))
                preds = {s: np.full(panel.n, np.nan) for s in stages}
                for t in f.val:
                    r = panel.rows(t, t + 1)
                    for s, p in m.predict_staged(panel.X[r], stages).items():
                        preds[s][r] = p
                for i in idxs:
                    scores[i].append(score(preds[grid[i][staged]], panel, f.val))
    else:
        for i, cfg in enumerate(grid):
            for f in folds:
                m = proto.clone(**cfg).fit(TrainData(panel, moments, f.train, seed=seed))
                pred = predict_months(m, panel, moments, f.val, int(f.train.max()))
                scores[i].append(score(pred, panel, f.val))
    table = []
    for i, cfg in enumerate(grid):
        allm = np.concatenate(scores[i])
        allm = allm[np.isfinite(allm)]
        fold_means = [float(np.nanmean(s)) if np.isfinite(s).any() else np.nan for s in scores[i]]
        table.append({"config": cfg, "mean": float(allm.mean()) if len(allm) else np.nan,
                      "se": float(allm.std(ddof=1) / np.sqrt(len(allm))) if len(allm) > 1 else np.nan,
                      "folds": fold_means})
    means = np.array([row["mean"] for row in table])
    if not np.isfinite(means).any():
        return dict(grid[0]), table
    best = int(np.nanargmax(means))
    if rule == "1se":
        bar = means[best] - table[best]["se"]
        best = next(i for i in range(len(grid)) if np.isfinite(means[i]) and means[i] >= bar)
    elif rule != "best":
        raise ValueError(f"unknown selection rule {rule!r}")
    return dict(grid[best]), table
