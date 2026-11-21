"""Synthetic signal panels with a KNOWN data-generating process.

Signals z (K of them, in families) are Gaussian with a chosen correlation structure (rho_within inside a family,
rho_between across families) and month-to-month persistence (AR(1) coefficient phi). The true expected return is a
function f of the signals; the realized return is

    r_{i,t+1} = 0.08 * [ ic * f~_{i,t} + sqrt(1 - ic^2) * u_{i,t+1} ],   f~ = f standardized within the month,

so ic is the information coefficient of the oracle forecast. The noise u has a latent-factor part (shared by stocks
with similar loadings, so rows within a month are dependent, as in real data) and an idiosyncratic part.

DGPs
  linear_equal    every signal equally useful: equal weighting is optimal
  linear_unequal  useful families of declining strength; the last family is useless
  sparse          three signals carry everything
  wrong_sign      one signal's literature sign is wrong (its true weight is negative)
  nonlinear       part linear, plus an interaction, a V-shape and a threshold that no linear model can represent
  regime          an observable two-state regime flips the first family's sign (momentum crashes, stylized)
  decay           two families stop working halfway through (post-publication decay, stylized)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphacomb.panel import Panel
from alphacomb.signals.base import SignalSpec
from alphacomb.signals.transforms import Normalization

FAMILY_LABELS = ("momentum", "value", "quality", "lowrisk", "reversal", "investment", "seasonality", "other")
DGPS = ("linear_equal", "linear_unequal", "sparse", "wrong_sign", "nonlinear", "regime", "decay")


@dataclass
class SyntheticConfig:
    n_assets: int = 300
    n_months: int = 360
    family_sizes: tuple = (3, 3, 3, 3)
    rho_within: float = 0.6
    rho_between: float = 0.1
    persistence: float | tuple = 0.8
    dgp: str = "linear_equal"
    ic: float = 0.05
    factor_share: float = 0.3
    n_factors: int = 3
    regime_stay: float = 0.97
    missing_rate: float = 0.0
    start: str = "1990-01-31"
    seed: int = 0
    weights: tuple | None = None          # explicit true weights (overrides the DGP's linear part)

    @property
    def K(self) -> int:
        return int(sum(self.family_sizes))


@dataclass
class Synthetic:
    panel: Panel
    frame: pd.DataFrame
    true_weights: np.ndarray              # (T, K) linear part of the truth by month
    oracle: np.ndarray                    # row-level true expected-return score
    config: SyntheticConfig = field(repr=False, default=None)


def correlation_matrix(sizes, rho_within: float, rho_between: float) -> np.ndarray:
    K = int(sum(sizes))
    fam = np.repeat(np.arange(len(sizes)), sizes)
    C = np.where(fam[:, None] == fam[None, :], rho_within, rho_between).astype(float)
    np.fill_diagonal(C, 1.0)
    w = np.linalg.eigvalsh(C)
    if w.min() <= 1e-8:
        raise ValueError(f"correlation structure is not positive definite (min eigenvalue {w.min():.3g})")
    return C


def _linear_weights(cfg: SyntheticConfig, fam: np.ndarray) -> np.ndarray:
    K = cfg.K
    if cfg.weights is not None:
        return np.asarray(cfg.weights, float)
    if cfg.dgp in ("linear_equal", "regime", "decay", "nonlinear"):
        return np.ones(K)
    if cfg.dgp == "linear_unequal":
        strength = np.linspace(1.0, 0.0, len(cfg.family_sizes))
        within = np.concatenate([np.linspace(1.0, 0.4, s) for s in cfg.family_sizes])
        return strength[fam] * within
    if cfg.dgp == "sparse":
        w = np.zeros(K)
        starts = np.concatenate([[0], np.cumsum(cfg.family_sizes)[:-1]])
        w[starts[: min(3, len(starts))]] = 1.0
        return w
    if cfg.dgp == "wrong_sign":
        w = np.ones(K)
        w[1] = -1.0
        return w
    raise ValueError(f"unknown dgp {cfg.dgp!r}; known: {DGPS}")


def make(cfg: SyntheticConfig, norm: Normalization | None = None) -> Synthetic:
    rng = np.random.default_rng(cfg.seed)
    N, T, K = cfg.n_assets, cfg.n_months, cfg.K
    sizes = cfg.family_sizes
    fam = np.repeat(np.arange(len(sizes)), sizes)
    C = correlation_matrix(sizes, cfg.rho_within, cfg.rho_between)
    L = np.linalg.cholesky(C)
    phi = np.asarray(cfg.persistence, float)
    phi = np.full(K, float(phi)) if phi.ndim == 0 else np.repeat(phi, sizes) if len(phi) == len(sizes) else phi
    z = np.empty((T, N, K))
    z[0] = rng.standard_normal((N, K)) @ L.T
    for t in range(1, T):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * (rng.standard_normal((N, K)) @ L.T)
    # observable regime (a noise regime for DGPs that do not use it)
    g = np.zeros(T, dtype=int)
    for t in range(1, T):
        g[t] = g[t - 1] if rng.random() < cfg.regime_stay else 1 - g[t - 1]
    w_lin = _linear_weights(cfg, fam)
    W = np.tile(w_lin, (T, 1))
    if cfg.dgp == "regime":
        W[g == 1, :] = np.where(fam == 0, -1.5, 1.0)
    if cfg.dgp == "decay":
        W[T // 2:, :] = np.where(fam < 2, 0.0, 1.0)
    f = np.einsum("tnk,tk->tn", z, W)
    if cfg.dgp == "nonlinear":
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        a, b, c = starts[0], starts[1 % len(starts)], starts[2 % len(starts)]
        d = starts[3 % len(starts)]
        lin = f / np.sqrt(np.var(f, axis=1, keepdims=True))
        inter = z[..., a] * z[..., b]
        vee = np.abs(z[..., c]) - np.sqrt(2 / np.pi)
        thr = np.maximum(z[..., d], 0.0) - 1 / np.sqrt(2 * np.pi)
        f = 0.5 * lin + inter + vee + thr
    f = (f - f.mean(1, keepdims=True)) / f.std(1, keepdims=True)
    lam = rng.standard_normal((N, cfg.n_factors))
    F = rng.standard_normal((T, cfg.n_factors))
    common = (F @ lam.T) / np.sqrt(cfg.n_factors)
    u = np.sqrt(cfg.factor_share) * common + np.sqrt(1 - cfg.factor_share) * rng.standard_normal((T, N))
    ret = 0.08 * (cfg.ic * f + np.sqrt(1 - cfg.ic ** 2) * u)
    dates = pd.date_range(cfg.start, periods=T, freq="ME")
    names = [f"S{k + 1:02d}" for k in range(K)]
    rows = pd.DataFrame({"date": np.repeat(dates, N), "instrument": np.tile(np.arange(N), T)})
    zz = z.reshape(T * N, K).copy()
    if cfg.missing_rate > 0:
        zz[rng.random(zz.shape) < cfg.missing_rate] = np.nan
    for k, nm in enumerate(names):
        rows[nm] = zz[:, k]
    rows["ret_next"] = ret.reshape(-1)
    rows["oracle"] = f.reshape(-1)
    specs = [SignalSpec(name=nm, family=FAMILY_LABELS[fam[k] % len(FAMILY_LABELS)],
                        description=f"synthetic signal {k + 1}", direction=1, published=None,
                        inputs=("synthetic",)) for k, nm in enumerate(names)]
    info = pd.DataFrame({"regime": np.where(g == 1, "r1", "r0")}, index=dates)
    panel = Panel.from_frame(rows, specs, norm or Normalization(), extras=("oracle",), month_info=info)
    return Synthetic(panel=panel, frame=rows, true_weights=W, oracle=panel.extras["oracle"], config=cfg)
