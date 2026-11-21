from __future__ import annotations

from alphacomb.signals.base import SignalSpec
from alphacomb.signals.library._common import CRSP_MONTHLY, cum_log_return
from alphacomb.signals.registry import register


@register(SignalSpec(
    name="STREV", family="reversal", direction=-1, published=1990, min_history=1,
    description="Total return in month t (losers outperform next month)",
    reference="Jegadeesh (1990), Journal of Finance; Lehmann (1990), Quarterly Journal of Economics",
    inputs=(CRSP_MONTHLY,), lag="return of month t, known at its close"))
def strev(b):
    R = b.wide("tr")
    return b.on_universe(R, confidence=R.notna().astype(float), n_obs=1)


@register(SignalSpec(
    name="LTREV", family="reversal", direction=-1, published=1985, min_history=60,
    description="Compounded total return over months t-59..t-12 (long-term losers outperform)",
    reference="De Bondt & Thaler (1985), Journal of Finance",
    inputs=(CRSP_MONTHLY,), lag="returns through month t-12; needs 36 of 48 months"))
def ltrev(b):
    v, n = cum_log_return(b.wide("tr"), 12, 59, 36)
    return b.on_universe(v, confidence=n / 48.0, n_obs=n)
