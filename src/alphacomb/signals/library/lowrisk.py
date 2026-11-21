from __future__ import annotations

from alphacomb.signals.base import SignalSpec
from alphacomb.signals.library._common import CRSP_DAILY, FRENCH
from alphacomb.signals.registry import register


@register(SignalSpec(
    name="IVOL", family="lowrisk", direction=-1, published=2006, min_history=1,
    description="Idiosyncratic volatility: standard deviation of daily residuals from a Fama-French three-factor "
                "regression within month t",
    reference="Ang, Hodrick, Xing & Zhang (2006), Journal of Finance", inputs=(CRSP_DAILY, FRENCH),
    lag="daily returns of month t; needs 15 days"))
def ivol(b):
    days = b.wide("ivol_days", "daily")
    return b.on_universe(b.wide("ivol", "daily"), confidence=(days / 21.0).clip(upper=1), n_obs=days)


@register(SignalSpec(
    name="BETA", family="lowrisk", direction=-1, published=1972, min_history=6,
    description="Market beta from daily excess returns on the market factor over the trailing 12 months",
    reference="Black, Jensen & Scholes (1972); Frazzini & Pedersen (2014), Journal of Financial Economics",
    inputs=(CRSP_DAILY, FRENCH), lag="daily returns through month t; needs 120 days"))
def beta(b):
    days = b.wide("beta_days", "daily")
    return b.on_universe(b.wide("beta", "daily"), confidence=(days / 252.0).clip(upper=1), n_obs=days)
