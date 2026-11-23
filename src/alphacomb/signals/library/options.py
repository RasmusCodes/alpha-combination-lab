"""Option-implied signals from OptionMetrics (signal extension; not part of the pre-registered sixteen).
All use the 30-day volatility surface on the month's last trading day (from 1996)."""
from __future__ import annotations

from alphacomb.signals.base import SignalSpec
from alphacomb.signals.library.earnings import _records
from alphacomb.signals.registry import register

OM = "optionm.vsurfd+wrdsapps.opcrsphist"
LAG = "surface of the month's last trading day (1996 onward)"


@register(SignalSpec(
    name="IVSKEW", family="options", direction=-1, published=2010, min_history=1,
    description="30-day implied volatility of the delta -20 put minus that of the delta +50 call",
    reference="Xing, Zhang & Zhao (2010), Journal of Financial and Quantitative Analysis", inputs=(OM,), lag=LAG))
def ivskew(b):
    x = b.om_iv.assign(v=lambda d: d["put20"] - d["call50"])
    return _records(b, x, "v")


@register(SignalSpec(
    name="IVSPREAD", family="options", direction=1, published=2010, min_history=1,
    description="30-day at-the-money call implied volatility minus put implied volatility (deltas +50 / -50)",
    reference="Cremers & Weinbaum (2010), Journal of Financial and Quantitative Analysis", inputs=(OM,), lag=LAG))
def ivspread(b):
    x = b.om_iv.assign(v=lambda d: d["call50"] - d["put50"])
    return _records(b, x, "v")


@register(SignalSpec(
    name="VRP", family="options", direction=1, published=2009, min_history=1,
    description="Realized volatility of daily returns in month t (annualized, CRSP) minus the 30-day at-the-money "
                "implied volatility (mean of the delta +50 call and -50 put)",
    reference="Bali & Hovakimian (2009), Management Science", inputs=(OM, "crsp.dsf"), lag=LAG))
def vrp(b):
    iv = b.om_iv.assign(iv=lambda d: (d["call50"] + d["put50"]) / 2)[["permno", "month", "iv"]]
    rv = b.daily[["permno", "month", "rv1"]]
    x = iv.merge(rv, on=["permno", "month"], how="left").assign(v=lambda d: d["rv1"] - d["iv"])
    return _records(b, x, "v")
