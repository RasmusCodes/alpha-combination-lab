from __future__ import annotations

import numpy as np

from alphacomb.signals.base import SignalSpec
from alphacomb.signals.library._common import COMPUSTAT, CRSP_MONTHLY, FUND_LAG
from alphacomb.signals.library.value import _frame
from alphacomb.signals.registry import register


@register(SignalSpec(
    name="AG", family="investment", direction=-1, published=2008, min_history=0,
    description="Total asset growth over the fiscal year (at / at_prev - 1)",
    reference="Cooper, Gulen & Schill (2008), Journal of Finance", inputs=(COMPUSTAT,),
    lag=FUND_LAG + "; needs the prior year"))
def ag(b):
    x = b.fundamentals()
    return _frame(x, x["ag"])


@register(SignalSpec(
    name="NSI", family="investment", direction=-1, published=2008, min_history=12,
    description="Net share issuance: log change in split-adjusted shares outstanding (shrout x cfacshr) over the "
                "past 12 months",
    reference="Pontiff & Woodgate (2008), Journal of Finance", inputs=(CRSP_MONTHLY,),
    lag="shares outstanding at months t and t-12"))
def nsi(b):
    s = b.wide("shrout") * b.wide("cfacshr")
    s = s.where(s > 0)
    v = np.log(s / s.shift(12))
    return b.on_universe(v, confidence=v.notna().astype(float), n_obs=2)
