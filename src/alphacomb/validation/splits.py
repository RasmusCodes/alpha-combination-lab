"""Time-series splits that respect the information flow.

Formation month s has a label (the return over the next `horizon` months) that is known at the end of month
s + horizon. A model predicting at the close of month t may therefore learn only from s <= t - horizon (the PURGE);
the EMBARGO removes a further gap of months to blunt serial dependence.

walk_forward_refits  outer schedule: refit dates and the months each fitted model predicts
forward_folds        inner folds for nested model selection: validation blocks at the end of the training window,
                     each trained only on months before it (no future information even inside the training window)
purged_kfold         López de Prado's purged K-fold with an embargo after each validation block (uses training
                     months on both sides of the block)
naive_kfold          shuffled K-fold with no purge or embargo: kept only to MEASURE leakage, never to select models
cpcv                 combinatorial purged cross-validation splits
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    val: np.ndarray


def last_trainable(t_pred: int, horizon: int = 1, embargo: int = 0) -> int:
    """Last formation month whose label is known (and outside the embargo) when predicting at month t_pred."""
    return t_pred - horizon - embargo


def walk_forward_refits(dates: pd.DatetimeIndex, first_test, last_test=None, every: int = 12) -> list[tuple[int, np.ndarray]]:
    """[(refit month index, months predicted by that fit)], refitting every `every` months from first_test."""
    t0 = int(dates.searchsorted(pd.Timestamp(first_test) + pd.offsets.MonthEnd(0)))
    t1 = len(dates) if last_test is None else int(dates.searchsorted(pd.Timestamp(last_test) + pd.offsets.MonthEnd(0),
                                                                     side="right"))
    out = []
    for r in range(t0, t1, every):
        out.append((r, np.arange(r, min(r + every, t1))))
    return out


def train_window(t_refit: int, first: int = 0, horizon: int = 1, embargo: int = 1, window: int | None = None) -> np.ndarray:
    end = last_trainable(t_refit, horizon, embargo)
    start = first if window is None else max(first, end - window + 1)
    return np.arange(start, end + 1)


def forward_folds(train: np.ndarray, n_folds: int = 3, block: int = 36, horizon: int = 1, embargo: int = 1,
                  min_train: int = 60) -> list[Fold]:
    train = np.sort(np.asarray(train))
    folds = []
    for j in range(n_folds, 0, -1):
        v_end = len(train) - (j - 1) * block
        v_start = v_end - block
        if v_start <= 0:
            continue
        val = train[v_start:v_end]
        cut = val[0] - horizon - embargo
        tr = train[train <= cut]
        if len(tr) >= min_train:
            folds.append(Fold(tr, val))
    return folds


def _purge_mask(train: np.ndarray, val: np.ndarray, horizon: int, embargo: int) -> np.ndarray:
    v0, v1 = val.min(), val.max()
    overlap = (train + horizon > v0) & (train < v1 + horizon)       # label windows overlap the validation labels
    embargoed = (train > v1) & (train <= v1 + horizon - 1 + embargo)
    return ~(overlap | embargoed)


def purged_kfold(train: np.ndarray, n_folds: int = 5, horizon: int = 1, embargo: int = 12) -> list[Fold]:
    train = np.sort(np.asarray(train))
    blocks = np.array_split(train, n_folds)
    folds = []
    for b in blocks:
        rest = np.setdiff1d(train, b)
        keep = _purge_mask(rest, b, horizon, embargo)
        folds.append(Fold(rest[keep], b))
    return folds


def naive_kfold(train: np.ndarray, n_folds: int = 5, seed: int = 0) -> list[Fold]:
    train = np.asarray(train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train)
    return [Fold(np.sort(np.setdiff1d(train, v)), np.sort(v)) for v in np.array_split(perm, n_folds)]


def cpcv(months: np.ndarray, n_groups: int = 6, k_test: int = 2, horizon: int = 1, embargo: int = 1) -> list[Fold]:
    """Combinatorial purged CV: every choice of k_test of n_groups contiguous groups is a validation set."""
    months = np.sort(np.asarray(months))
    groups = np.array_split(months, n_groups)
    folds = []
    for test_ids in combinations(range(n_groups), k_test):
        val = np.concatenate([groups[i] for i in test_ids])
        tr = np.setdiff1d(months, val)
        keep = np.ones(len(tr), dtype=bool)
        for i in test_ids:
            keep &= _purge_mask(tr, groups[i], horizon, embargo)
        folds.append(Fold(tr[keep], val))
    return folds
