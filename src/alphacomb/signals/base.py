"""The standardized signal interface.

Every signal, whatever its source (prices, accounting data, an alternative-data vendor, an options feed), is delivered
as RECORDS with the same eight fields:

  instrument   security key (CRSP PERMNO in the equity study; any integer id for synthetic or external data)
  timestamp    formation date: the month end at whose close the value is known (no later information used)
  horizon      forecast horizon in months (1 = next month's return)
  value        the raw signal value (before any cross-sectional transform; orientation is in the spec)
  confidence   0..1, how much the producer trusts this value (data completeness, staleness, estimation noise)
  n_obs        historical observations behind the value (daily returns in a window, months of history, ...)
  provenance   where the inputs came from (tables and construction id)
  version      construction version: the spec's version plus a fingerprint of the code that produced it

A SignalSpec carries the metadata that is the same for every record: family, the expected sign from the literature
(direction), the reference and its publication year (used for real-time signal availability), the inputs and the
availability lag. New signals are added by registering a function (signals/registry.py) or by loading a vendor file
in the record format (ExternalSignal); nothing else in the platform changes.
"""
from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

RECORD_COLUMNS = ["instrument", "timestamp", "horizon", "value", "confidence", "n_obs", "provenance", "version"]
FAMILIES = ("momentum", "reversal", "value", "quality", "investment", "lowrisk", "seasonality",
            "earnings", "alternative", "options", "macro", "event", "synthetic", "other")


@dataclass(frozen=True)
class SignalSpec:
    name: str
    family: str
    description: str
    direction: int = 1                  # +1: higher raw value -> higher expected return (literature prior)
    horizon: int = 1                    # months
    reference: str = ""
    published: int | None = None        # publication year of the reference (real-time availability)
    inputs: tuple[str, ...] = ()        # source tables / fields
    lag: str = ""                       # availability rule, in words
    version: str = "1.0"
    min_history: int = 0                # months of history needed before the first value

    def __post_init__(self):
        if self.direction not in (-1, 1):
            raise ValueError(f"{self.name}: direction must be +1 or -1")
        if self.family not in FAMILIES:
            raise ValueError(f"{self.name}: unknown family {self.family!r}; known: {FAMILIES}")
        if self.horizon < 1:
            raise ValueError(f"{self.name}: horizon must be >= 1 month")

    def as_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "description": self.description,
                "direction": self.direction, "horizon": self.horizon, "reference": self.reference,
                "published": self.published, "inputs": list(self.inputs), "lag": self.lag,
                "version": self.version, "min_history": self.min_history}


def code_fingerprint(fn: Callable) -> str:
    """Short hash of a function's source: the 'model version' changes by itself when the construction changes."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:10]


def validate_records(df: pd.DataFrame, spec: SignalSpec | None = None) -> pd.DataFrame:
    """Check a record frame against the interface; returns it with canonical dtypes and order.

    Raises ValueError on: missing columns, null keys, duplicate (instrument, timestamp), non-month-end timestamps,
    confidence outside [0, 1], negative n_obs, horizon different from the spec's."""
    missing = [c for c in RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"signal records lack columns {missing}")
    out = df[RECORD_COLUMNS].copy()
    if out["instrument"].isna().any() or out["timestamp"].isna().any():
        raise ValueError("null instrument or timestamp in signal records")
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ns]")
    if not (out["timestamp"] == out["timestamp"] + pd.offsets.MonthEnd(0)).all():
        raise ValueError("timestamps must be month ends (formation dates)")
    if out.duplicated(["instrument", "timestamp"]).any():
        raise ValueError("duplicate (instrument, timestamp) in signal records")
    out["value"] = out["value"].astype("float64")
    out["confidence"] = out["confidence"].astype("float64")
    c = out["confidence"].dropna()
    if ((c < 0) | (c > 1)).any():
        raise ValueError("confidence must lie in [0, 1]")
    out["n_obs"] = out["n_obs"].fillna(0).astype("int64")
    if (out["n_obs"] < 0).any():
        raise ValueError("n_obs must be >= 0")
    out["horizon"] = out["horizon"].astype("int64")
    if spec is not None and (out["horizon"] != spec.horizon).any():
        raise ValueError(f"{spec.name}: record horizon differs from the spec's ({spec.horizon})")
    out["provenance"] = out["provenance"].astype(str)
    out["version"] = out["version"].astype(str)
    return out.sort_values(["timestamp", "instrument"]).reset_index(drop=True)


class Signal(ABC):
    """A named signal: metadata (spec) plus a way to produce records from a data bundle."""

    spec: SignalSpec

    @abstractmethod
    def compute(self, bundle) -> pd.DataFrame:
        """-> frame with columns instrument, timestamp, value, and optionally confidence, n_obs."""

    @property
    def version(self) -> str:
        return self.spec.version

    @property
    def provenance(self) -> str:
        return ";".join(self.spec.inputs) or "unspecified"

    def records(self, bundle) -> pd.DataFrame:
        raw = self.compute(bundle)
        df = raw.copy()
        if "confidence" not in df:
            df["confidence"] = np.where(df["value"].notna(), 1.0, 0.0)
        if "n_obs" not in df:
            df["n_obs"] = 0
        df["horizon"] = self.spec.horizon
        df["provenance"] = self.provenance
        df["version"] = self.version
        df = df[df["value"].notna() & np.isfinite(df["value"].astype(float))]
        return validate_records(df, self.spec)


@dataclass
class FunctionSignal(Signal):
    """A signal defined by a function bundle -> frame (the registry wraps registered functions in this)."""
    spec: SignalSpec
    fn: Callable = field(repr=False, default=None)

    def compute(self, bundle) -> pd.DataFrame:
        return self.fn(bundle)

    @property
    def version(self) -> str:
        return f"{self.spec.version}+{code_fingerprint(self.fn)}"


@dataclass
class ExternalSignal(Signal):
    """Records supplied by someone else (a vendor file, another team's model) in the record format.

    The file must hold at least instrument, timestamp and value; it is validated like any computed signal, so an
    alternative-data, options-implied or analyst-revision signal enters the platform without code changes."""
    spec: SignalSpec
    path: str | Path = ""
    frame: pd.DataFrame | None = field(default=None, repr=False)

    def compute(self, bundle=None) -> pd.DataFrame:
        if self.frame is not None:
            return self.frame
        p = Path(self.path)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, parse_dates=["timestamp"])

    @property
    def provenance(self) -> str:
        return f"external:{Path(self.path).name if self.path else 'frame'}"
