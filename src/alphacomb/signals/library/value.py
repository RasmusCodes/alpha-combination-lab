from __future__ import annotations

import numpy as np
import pandas as pd

from alphacomb.signals.base import SignalSpec
from alphacomb.signals.library._common import COMPUSTAT, CRSP_MONTHLY, FUND_LAG, fund_confidence
from alphacomb.signals.registry import register


def _frame(x: pd.DataFrame, v) -> pd.DataFrame:
    return pd.DataFrame({"instrument": x["permno"].to_numpy(), "timestamp": x["month"].to_numpy(),
                         "value": np.asarray(v, float), "confidence": fund_confidence(x["fund_age"]),
                         "n_obs": 1})


@register(SignalSpec(
    name="BM", family="value", direction=1, published=1985, min_history=0,
    description="log(book equity / market equity); Fama-French book equity, market equity of all the issuer's share "
                "classes at month t (current price, as in Asness & Frazzini 2013); positive book equity only",
    reference="Rosenberg, Reid & Lanstein (1985), Journal of Portfolio Management; Fama & French (1992), Journal of "
              "Finance", inputs=(COMPUSTAT, CRSP_MONTHLY), lag=FUND_LAG))
def bm(b):
    x = b.fundamentals()
    return _frame(x, np.log((x["be"] / x["me"]).where((x["be"] > 0) & (x["me"] > 0))))


@register(SignalSpec(
    name="EP", family="value", direction=1, published=1977, min_history=0,
    description="Earnings before extraordinary items (ib) / market equity at month t",
    reference="Basu (1977), Journal of Finance", inputs=(COMPUSTAT, CRSP_MONTHLY), lag=FUND_LAG))
def ep(b):
    x = b.fundamentals()
    return _frame(x, (x["earn"] / x["me"]).where(x["me"] > 0))


@register(SignalSpec(
    name="CFP", family="value", direction=1, published=1994, min_history=0,
    description="Cash flow (ib + depreciation dp) / market equity at month t",
    reference="Lakonishok, Shleifer & Vishny (1994), Journal of Finance", inputs=(COMPUSTAT, CRSP_MONTHLY),
    lag=FUND_LAG))
def cfp(b):
    x = b.fundamentals()
    return _frame(x, (x["cf"] / x["me"]).where(x["me"] > 0))
