"""Shared setup of the real-data study: the pre-registered panel, walk-forward settings and method list.

Everything an experiment script needs to rebuild the exact inputs is here, so the pre-registration
(experiments/preregistration.md) and the code cannot drift apart: tests/test_study.py checks the values below
against the frozen configuration file experiments/configs/main.yaml.
"""
from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alphacomb import paths
from alphacomb.evaluation import costs
from alphacomb.models.base import Moments
from alphacomb.panel import Panel
from alphacomb.registry import fingerprint_arrays
from alphacomb.signals import registry
from alphacomb.signals.library import LIBRARY
from alphacomb.signals.transforms import Normalization
from alphacomb.validation.engine import WFConfig


def load_config(name: str = "main") -> dict:
    return yaml.safe_load((paths.CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def build_panel(cfg: dict) -> Panel:
    frame = pd.read_parquet(paths.PROCESSED / "panel.parquet")
    frame = frame[(frame["date"] >= cfg["panel_start"]) & (frame["date"] <= cfg["panel_end"])]
    names = cfg["signals"]
    specs = [registry.get(n).spec for n in names]
    info = pd.read_parquet(paths.PROCESSED / "month_info.parquet")
    norm = Normalization(**cfg["normalization"])
    p = Panel.from_frame(frame, specs, norm, date_col="date", id_col="instrument", ret_col="ret_next",
                         extras=("mcap", "dvol", "sigma", "ff48"), month_info=info, min_signals=1)
    base = costs.era_cost(pd.DatetimeIndex(np.repeat(p.dates.to_numpy(), np.diff(p.offsets))))
    p.extras["cost"] = costs.stock_cost(base, p.extras["dvol"].astype(float), p.offsets)
    return p


def wf_config(cfg: dict, **overrides) -> WFConfig:
    w = dict(cfg["walk_forward"])
    w.update(overrides)
    return WFConfig(**w)


def importance_groups(p: Panel) -> dict:
    g = {n: [k] for k, n in enumerate(p.names)}
    for f, idx in p.family_index().items():
        g[f"family:{f}"] = idx
    return g


def data_fingerprint(p: Panel) -> str:
    return fingerprint_arrays(p.X, np.nan_to_num(p.ret), p.offsets, p.ids)


def save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load(path: Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def moments_for(p: Panel, cache: Path | None = None) -> Moments:
    if cache is not None and cache.exists():
        m = load(cache)
        if m.G.shape[0] == p.T and m.G.shape[1] == p.K:
            return m
    m = Moments.from_panel(p)
    if cache is not None:
        save(m, cache)
    return m


__all__ = ["load_config", "build_panel", "wf_config", "importance_groups", "data_fingerprint", "save", "load",
           "moments_for", "LIBRARY", "asdict"]
