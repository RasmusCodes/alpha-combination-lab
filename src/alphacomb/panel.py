"""Panel: month-sorted long arrays (one row per instrument and formation month) shared by every model.

Rows of month t are offsets[t]:offsets[t+1]. X holds the K signals after the pre-specified cross-sectional pipeline
(standardized within the month, oriented by the literature sign, missing -> 0). y is the training target: the
next-month return turned into within-month normal scores (rank-gauss), NaN where the return is unknown. ret keeps the
raw next-month return for portfolio evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from alphacomb.signals.base import SignalSpec
from alphacomb.signals.transforms import Normalization, offsets_from_sorted, rank_gauss


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    offsets: np.ndarray
    ids: np.ndarray
    X: np.ndarray
    y: np.ndarray
    ret: np.ndarray
    names: list[str]
    families: list[str]
    mask: np.ndarray                                   # signal available before missing-value fill
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    month_info: pd.DataFrame | None = None             # per-month state (regimes ...), index = dates
    specs: list[SignalSpec] | None = None

    # ------------------------------------------------------------------ shape helpers
    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def K(self) -> int:
        return self.X.shape[1]

    def rows(self, t0: int, t1: int) -> slice:
        """Rows of months t0 .. t1-1 (contiguous because rows are month-sorted)."""
        return slice(int(self.offsets[t0]), int(self.offsets[t1]))

    def month_of_row(self) -> np.ndarray:
        return np.repeat(np.arange(self.T), np.diff(self.offsets))

    def t_of(self, date) -> int:
        """Index of the formation month containing date (month end)."""
        d = pd.Timestamp(date) + pd.offsets.MonthEnd(0)
        return int(self.dates.get_loc(d))

    def family_index(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for k, f in enumerate(self.families):
            out.setdefault(f, []).append(k)
        return out

    # ------------------------------------------------------------------ subsets
    def subset_signals(self, names: list[str]) -> "Panel":
        idx = [self.names.index(nm) for nm in names]
        return replace(self, X=self.X[:, idx], mask=self.mask[:, idx], names=[self.names[i] for i in idx],
                       families=[self.families[i] for i in idx],
                       specs=[self.specs[i] for i in idx] if self.specs else None)

    def subset_months(self, t0: int, t1: int) -> "Panel":
        r = self.rows(t0, t1)
        off = self.offsets[t0:t1 + 1] - self.offsets[t0]
        mi = self.month_info.iloc[t0:t1] if self.month_info is not None else None
        return replace(self, dates=self.dates[t0:t1], offsets=off, ids=self.ids[r], X=self.X[r], y=self.y[r],
                       ret=self.ret[r], mask=self.mask[r], extras={k: v[r] for k, v in self.extras.items()},
                       month_info=mi)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_frame(cls, df: pd.DataFrame, specs: list[SignalSpec], norm: Normalization | None = None,
                   date_col: str = "date", id_col: str = "instrument", ret_col: str = "ret_next",
                   extras: tuple[str, ...] = (), month_info: pd.DataFrame | None = None,
                   min_signals: int = 1) -> "Panel":
        """df: one row per (date, instrument) with a raw column per spec.name and the next-month return.
        Rows with fewer than min_signals available signals are dropped (before normalization)."""
        norm = norm or Normalization()
        names = [s.name for s in specs]
        d = df.sort_values([date_col, id_col]).reset_index(drop=True)
        avail = d[names].notna().sum(axis=1)
        d = d[avail >= min_signals].reset_index(drop=True)
        dates = pd.DatetimeIndex(d[date_col].astype("datetime64[ns]"))
        uniq = pd.DatetimeIndex(np.unique(dates.to_numpy()))
        tidx = uniq.get_indexer(dates)
        offsets = offsets_from_sorted(tidx)
        raw = d[names].to_numpy(dtype=float)
        mask = np.isfinite(raw)
        X = np.empty_like(raw)
        for k, s in enumerate(specs):
            X[:, k] = norm.apply(raw[:, k], offsets, s.direction)
            mask[:, k] &= np.isfinite(rank_gauss(raw[:, k], offsets, norm.min_n))   # month had enough values
        ret = d[ret_col].to_numpy(dtype=float)
        y = rank_gauss(ret, offsets, min_n=norm.min_n)
        ex = {c: d[c].to_numpy() for c in extras}
        mi = month_info.reindex(uniq) if month_info is not None else None
        return cls(dates=uniq, offsets=offsets, ids=d[id_col].to_numpy(), X=X, y=y, ret=ret, names=names,
                   families=[s.family for s in specs], mask=mask, extras=ex, month_info=mi, specs=list(specs))

    def to_frame(self) -> pd.DataFrame:
        f = pd.DataFrame(self.X, columns=self.names)
        f.insert(0, "date", np.repeat(self.dates.to_numpy(), np.diff(self.offsets)))
        f.insert(1, "instrument", self.ids)
        f["y"], f["ret_next"] = self.y, self.ret
        for k, v in self.extras.items():
            f[k] = v
        return f
