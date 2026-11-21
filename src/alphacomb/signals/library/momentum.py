from __future__ import annotations

import numpy as np
import pandas as pd

from alphacomb.signals.base import SignalSpec
from alphacomb.signals.library._common import CRSP_DAILY, CRSP_MONTHLY, cum_log_return
from alphacomb.signals.registry import register


@register(SignalSpec(
    name="MOM", family="momentum", direction=1, published=1993, min_history=12,
    description="Compounded total return over months t-11..t-1 (skips the most recent month)",
    reference="Jegadeesh & Titman (1993), Journal of Finance",
    inputs=(CRSP_MONTHLY,), lag="returns through month t-1; needs 8 of 11 months"))
def mom(b):
    v, n = cum_log_return(b.wide("tr"), 1, 11, 8)
    return b.on_universe(v, confidence=n / 11.0, n_obs=n)


@register(SignalSpec(
    name="IMOM", family="momentum", direction=1, published=1999, min_history=6,
    description="Six-month compounded value-weighted return of the stock's Fama-French 48 industry (S&P 500 members, "
                "weights = previous month-end market caps), months t-5..t",
    reference="Moskowitz & Grinblatt (1999), Journal of Finance",
    inputs=(CRSP_MONTHLY, "french.Siccodes48"), lag="returns through month t; industry needs >= 3 members"))
def imom(b):
    m = b.monthly[["permno", "month", "tr", "mcap", "ff48", "member", "alive"]]
    elig = m[m["member"] & m["alive"] & (m["mcap"] > 0)][["permno", "month", "mcap", "ff48"]]
    nxt = elig.assign(month=elig["month"] + pd.offsets.MonthEnd(1))          # weights formed at t-1 earn month t
    j = nxt.merge(m[["permno", "month", "tr"]], on=["permno", "month"], how="left")
    j = j[j["tr"].notna() & j["ff48"].notna()]
    j["w"] = j["mcap"]
    g = j.assign(wr=j["w"] * j["tr"]).groupby(["month", "ff48"])
    ind = (g["wr"].sum() / g["w"].sum()).rename("r").reset_index()
    cnt = elig.groupby(["month", "ff48"]).size().rename("n").reset_index()
    W = ind.pivot(index="month", columns="ff48", values="r").reindex(b.months)
    L = np.log1p(W)
    cum = np.expm1(L.rolling(6, min_periods=5).sum())
    u = b.universe
    look = cum.stack().rename("v").reset_index().rename(columns={"level_0": "month"})
    look.columns = ["month", "ff48", "v"]
    x = u[["permno", "month", "ff48"]].merge(look, on=["month", "ff48"], how="left")
    x = x.merge(cnt, on=["month", "ff48"], how="left")
    x["v"] = x["v"].where(x["n"] >= 3)
    return pd.DataFrame({"instrument": x["permno"].to_numpy(), "timestamp": x["month"].to_numpy(),
                         "value": x["v"].to_numpy(), "confidence": np.clip(x["n"].to_numpy() / 10.0, 0, 1),
                         "n_obs": x["n"].fillna(0).to_numpy()})


@register(SignalSpec(
    name="HIGH52", family="momentum", direction=1, published=2004, min_history=6,
    description="Month-end split-adjusted price divided by its highest daily price over the trailing 12 months",
    reference="George & Hwang (2004), Journal of Finance",
    inputs=(CRSP_DAILY,), lag="daily prices through month t; needs 6 months of daily data"))
def high52(b):
    return b.on_universe(b.wide("high52", "daily"), confidence=1.0, n_obs=252)
