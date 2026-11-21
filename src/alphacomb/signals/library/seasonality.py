from __future__ import annotations

import pandas as pd

from alphacomb.signals.base import SignalSpec
from alphacomb.signals.library._common import CRSP_MONTHLY
from alphacomb.signals.registry import register


@register(SignalSpec(
    name="SEAS", family="seasonality", direction=1, published=2008, min_history=60,
    description="Average return in the calendar month being predicted over the previous five years (months t-11, "
                "t-23, t-35, t-47, t-59)",
    reference="Heston & Sadka (2008), Journal of Financial Economics", inputs=(CRSP_MONTHLY,),
    lag="returns through month t-11; needs 3 of 5 years"))
def seas(b):
    R = b.wide("tr")
    lags = [R.shift(k) for k in (11, 23, 35, 47, 59)]
    stack = pd.concat(lags, keys=range(5))
    s = stack.groupby(level=1).sum(min_count=1)
    n = stack.notna().groupby(level=1).sum()
    v = (s / n).where(n >= 3).reindex(R.index)
    return b.on_universe(v, confidence=(n / 5.0).reindex(R.index), n_obs=n.reindex(R.index))
