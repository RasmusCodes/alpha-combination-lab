from __future__ import annotations

from alphacomb.signals.base import SignalSpec
from alphacomb.signals.library._common import COMPUSTAT, FUND_LAG
from alphacomb.signals.library.value import _frame
from alphacomb.signals.registry import register


@register(SignalSpec(
    name="GPA", family="quality", direction=1, published=2013, min_history=0,
    description="Gross profits (revt - cogs) / total assets",
    reference="Novy-Marx (2013), Journal of Financial Economics", inputs=(COMPUSTAT,), lag=FUND_LAG))
def gpa(b):
    x = b.fundamentals()
    return _frame(x, x["gp"])


@register(SignalSpec(
    name="ROA", family="quality", direction=1, published=1996, min_history=0,
    description="Earnings before extraordinary items / total assets",
    reference="Haugen & Baker (1996), Journal of Financial Economics", inputs=(COMPUSTAT,), lag=FUND_LAG))
def roa(b):
    x = b.fundamentals()
    return _frame(x, x["roa"])


@register(SignalSpec(
    name="ACC", family="quality", direction=-1, published=1996, min_history=0,
    description="Balance-sheet operating accruals: change in non-cash current assets minus change in current "
                "liabilities excluding short-term debt, minus depreciation, over average total assets",
    reference="Sloan (1996), The Accounting Review", inputs=(COMPUSTAT,), lag=FUND_LAG + "; needs the prior year"))
def acc(b):
    x = b.fundamentals()
    return _frame(x, x["acc"])
