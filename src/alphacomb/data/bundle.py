"""DataBundle: what signal functions read. Loads the processed CRSP/Compustat files lazily and offers wide
(month x permno) views for rolling computations plus the formation universe to deliver records on."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from alphacomb import paths
from alphacomb.data import compustat


class CRSPBundle:
    def __init__(self, processed: Path | str = paths.PROCESSED):
        self.dir = Path(processed)
        self._wide: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ raw frames
    @cached_property
    def monthly(self) -> pd.DataFrame:
        m = pd.read_parquet(self.dir / "monthly.parquet")
        return m.sort_values(["permno", "month"]).drop_duplicates(["permno", "month"], keep="last")

    @cached_property
    def daily(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "daily_features.parquet")

    @cached_property
    def funda(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "funda_annual.parquet")

    @cached_property
    def ibes_fy1(self) -> pd.DataFrame:
        """Monthly FY1 consensus per PERMNO (extension; built by scripts/build_extension.py)."""
        return pd.read_parquet(self.dir / "ibes_fy1.parquet")

    @cached_property
    def ibes_sue(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "ibes_surprises.parquet")

    @cached_property
    def om_iv(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "om_iv.parquet")

    @cached_property
    def months(self) -> pd.DatetimeIndex:
        lo, hi = self.monthly["month"].min(), self.monthly["month"].max()
        return pd.date_range(lo, hi, freq="ME")

    @cached_property
    def universe(self) -> pd.DataFrame:
        """Formation universe: S&P 500 member at the month's last trading day, not yet delisted, market cap known."""
        m = self.monthly
        u = m[m["member"] & m["alive"] & (m["mcap"] > 0)]
        return u[["permno", "month", "mcap", "ff48", "permco"]].sort_values(["month", "permno"]).reset_index(drop=True)

    # ------------------------------------------------------------------ helpers
    def wide(self, col: str, frame: str = "monthly") -> pd.DataFrame:
        key = f"{frame}:{col}"
        if key not in self._wide:
            src = self.monthly if frame == "monthly" else self.daily
            w = src.pivot_table(index="month", columns="permno", values=col, aggfunc="last")
            self._wide[key] = w.reindex(self.months)
        return self._wide[key]

    def on_universe(self, value: pd.DataFrame, confidence: pd.DataFrame | float | None = None,
                    n_obs: pd.DataFrame | float | None = None) -> pd.DataFrame:
        """Wide (month x permno) arrays -> long records on the formation universe."""
        u = self.universe
        cols = value.columns
        ti = value.index.get_indexer(u["month"])
        pj = cols.get_indexer(u["permno"])
        ok = (ti >= 0) & (pj >= 0)
        V = np.full(len(u), np.nan)
        V[ok] = value.to_numpy()[ti[ok], pj[ok]]
        out = pd.DataFrame({"instrument": u["permno"].to_numpy(), "timestamp": u["month"].to_numpy(), "value": V})
        for name, arr in (("confidence", confidence), ("n_obs", n_obs)):
            if isinstance(arr, pd.DataFrame):
                a = np.full(len(u), np.nan)
                ai, aj = arr.index.get_indexer(u["month"]), arr.columns.get_indexer(u["permno"])
                okk = (ai >= 0) & (aj >= 0)
                a[okk] = arr.to_numpy()[ai[okk], aj[okk]]
                out[name] = a
            elif arr is not None:
                out[name] = arr
        if "confidence" in out:
            out["confidence"] = out["confidence"].clip(0, 1).fillna(0.0)
        return out

    def fundamentals(self) -> pd.DataFrame:
        """Annual inputs attached to universe rows (point in time), plus permco-level market equity."""
        if not hasattr(self, "_fund"):
            u = self.universe
            x = compustat.attach(u, self.funda)
            me = self.monthly.groupby(["permco", "month"])["mcap"].sum(min_count=1)
            x["me"] = me.reindex(pd.MultiIndex.from_arrays([u["permco"], u["month"]])).to_numpy()
            self._fund = x
        return self._fund
