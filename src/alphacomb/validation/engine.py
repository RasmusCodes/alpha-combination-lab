"""Walk-forward engine: the only way a model's forecasts are produced for evaluation.

At each refit date r (every `refit_every` months from first_test):
  1. training months = formation months whose labels are known by r, minus the embargo (expanding or rolling)
  2. hyperparameters are selected by nested CV inside those months only (validation.nested)
  3. the model is fit on all training months and predicts every month until the next refit; online models also
     absorb each month's outcome as soon as it is known (month t's outcome before predicting t + 1)
Nothing computed from months after a prediction date can reach that prediction: tests/test_no_lookahead.py
perturbs future data and checks that past forecasts do not change.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from alphacomb.evaluation.metrics import monthly_ic
from alphacomb.models.base import Combiner, Moments, TrainData
from alphacomb.panel import Panel
from alphacomb.validation import nested, splits


@dataclass
class WFConfig:
    first_test: str = "1980-01-31"
    last_test: str | None = None
    refit_every: int = 12
    horizon: int = 1
    embargo: int = 1
    window: int | None = None          # None = expanding; else rolling length in months
    min_train: int = 120
    first_train: str | None = None     # earliest formation month allowed in training
    inner: str = "forward"             # 'forward' | 'purged_kfold' | 'naive_kfold' | 'none'
    inner_folds: int = 3
    inner_block: int = 36
    inner_embargo: int = 1
    kfold_embargo: int = 12
    rule: str = "best"
    tune_every: int = 12               # months between hyperparameter re-selections (multiple of refit_every)
    importance: bool = False
    importance_groups: dict | None = None
    seed: int = 0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "importance_groups"}


@dataclass
class WFResult:
    method: str
    pred: np.ndarray                      # row-level OOS forecasts (NaN outside the test months)
    months: np.ndarray                    # OOS month indices
    weights: np.ndarray                   # (T, K) linear-equivalent weights used in each month (NaN if none)
    selections: list = field(default_factory=list)
    importance: dict = field(default_factory=dict)   # group -> (T,) rank-IC drop when the group is permuted
    seconds: float = 0.0

    def ic(self, panel: Panel, rank: bool = True) -> np.ndarray:
        return monthly_ic(self.pred, panel.ret, panel.offsets, months=self.months, rank=rank)


def inner_folds(cfg: WFConfig, train: np.ndarray) -> list[splits.Fold]:
    if cfg.inner == "forward":
        return splits.forward_folds(train, cfg.inner_folds, cfg.inner_block, cfg.horizon, cfg.inner_embargo,
                                    min_train=min(cfg.min_train, 60))
    if cfg.inner == "purged_kfold":
        return splits.purged_kfold(train, max(cfg.inner_folds, 3), cfg.horizon, cfg.kfold_embargo)
    if cfg.inner == "naive_kfold":
        return splits.naive_kfold(train, max(cfg.inner_folds, 3), cfg.seed)
    if cfg.inner == "none":
        return []
    raise ValueError(f"unknown inner CV scheme {cfg.inner!r}")


def run(proto: Combiner, panel: Panel, moments: Moments, cfg: WFConfig, fixed: dict | None = None,
        log=None, available=None) -> WFResult:
    """Walk-forward forecasts of one method. fixed: hyperparameters to use instead of nested selection.
    available: optional function date -> signal names usable at that refit (real-time signal availability); the
    model is then selected, fit and applied on that subset only (each subset's moments are computed once)."""
    if available is not None:
        return _run_available(proto, panel, moments, cfg, fixed, log, available)
    t_start = time.time()
    first = 0 if cfg.first_train is None else int(panel.dates.searchsorted(np.datetime64(cfg.first_train)))
    sched = splits.walk_forward_refits(panel.dates, cfg.first_test, cfg.last_test, cfg.refit_every)
    pred = np.full(panel.n, np.nan)
    W = np.full((panel.T, panel.K), np.nan)
    sel, imp = [], {}
    rng = np.random.default_rng(cfg.seed + 7)
    groups = cfg.importance_groups or {}
    if cfg.importance:
        imp = {g: np.full(panel.T, np.nan) for g in groups}
    oos = []
    params, last_tune = None, None
    for r, test in sched:
        train = splits.train_window(r, first, cfg.horizon, cfg.embargo, cfg.window)
        train = train[train >= first]
        if len(train) < cfg.min_train:
            if log:
                log(f"{proto.name}: skip refit {panel.dates[r].date()} (only {len(train)} training months)")
            continue
        tuned = False
        if fixed is not None:
            params, table = dict(fixed), []
        elif params is None or last_tune is None or r - last_tune >= cfg.tune_every or not proto.grid:
            params, table = nested.select(proto, panel, moments, inner_folds(cfg, train), cfg.rule, cfg.seed)
            last_tune, tuned = r, True
        else:
            table = []
        data = TrainData(panel, moments, train, seed=cfg.seed)
        model = proto.clone(**params).fit(data)
        sel.append({"refit": str(panel.dates[r].date()), "params": params, "cv": table, "tuned": tuned,
                    "deff": data.deff(), "complexity": model.complexity(), "n_train_months": int(len(train))})
        seen = int(train.max())
        for t in test:
            if model.online:
                for s in range(seen + 1, t):
                    model.update(s, panel, moments)
                seen = max(seen, t - 1)
            rows = panel.rows(t, t + 1)
            ctx = nested.month_context(panel, t)
            X = panel.X[rows]
            p = model.predict(X, ctx)
            pred[rows] = p
            w = model.weights()
            if w is not None:
                W[t] = w
            if cfg.importance and len(X) > 20:
                # every group's permuted copy of the month in ONE predict call (tree prediction has a large
                # per-call overhead); the same row permutation is applied to all columns of a group
                base = _rank_ic(p, panel.ret[rows])
                names = list(groups)
                stack = np.empty((len(names) * len(X), X.shape[1]))
                for j, g in enumerate(names):
                    Xp = X.copy()
                    perm = rng.permutation(len(X))
                    Xp[:, groups[g]] = X[perm][:, groups[g]]
                    stack[j * len(X):(j + 1) * len(X)] = Xp
                ps = model.predict(stack, ctx)
                for j, g in enumerate(names):
                    imp[g][t] = base - _rank_ic(ps[j * len(X):(j + 1) * len(X)], panel.ret[rows])
            oos.append(t)
        if log:
            log(f"{proto.name}: {panel.dates[r].date()} params={params} n_train={len(train)}")
    return WFResult(method=proto.name, pred=pred, months=np.array(oos, dtype=int), weights=W, selections=sel,
                    importance=imp, seconds=time.time() - t_start)


def _run_available(proto, panel, moments, cfg, fixed, log, available) -> WFResult:
    """Walk-forward where the usable signal set changes over time: one ordinary run per distinct signal set over
    the refits that use it, stitched together (hyperparameters re-selected whenever the set changes)."""
    t_start = time.time()
    sched = splits.walk_forward_refits(panel.dates, cfg.first_test, cfg.last_test, cfg.refit_every)
    sets: dict[tuple, list] = {}
    for r, test in sched:
        names = tuple(n for n in panel.names if n in set(available(panel.dates[r])))
        sets.setdefault(names, []).append((r, test))
    pred = np.full(panel.n, np.nan)
    W = np.full((panel.T, panel.K), np.nan)
    sel, oos = [], []
    for names, blocks in sets.items():
        if not names:
            continue
        sub = panel.subset_signals(list(names))
        idx = [panel.names.index(n) for n in names]
        sub_m = moments if len(names) == panel.K else Moments.from_panel(sub)
        first, last = blocks[0][1][0], blocks[-1][1][-1]
        c = WFConfig(**{**cfg.__dict__, "first_test": str(panel.dates[first].date()),
                        "last_test": str(panel.dates[last].date()), "importance": False})
        res = run(proto, sub, sub_m, c, fixed=fixed, log=log)
        keep = np.concatenate([np.arange(panel.offsets[t], panel.offsets[t + 1]) for t in res.months])
        pred[keep] = res.pred[keep]
        W[np.ix_(res.months, idx)] = res.weights[res.months]
        for s in res.selections:
            s["signals"] = list(names)
        sel += res.selections
        oos += list(res.months)
    order = np.argsort(oos)
    return WFResult(method=proto.name, pred=pred, months=np.array(oos, dtype=int)[order], weights=W,
                    selections=sorted(sel, key=lambda s: s["refit"]), seconds=time.time() - t_start)


def _rank_ic(p, r):
    from scipy.stats import rankdata
    ok = np.isfinite(p) & np.isfinite(r)
    if ok.sum() < 20:
        return np.nan
    a, b = rankdata(p[ok]), rankdata(r[ok])
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))
