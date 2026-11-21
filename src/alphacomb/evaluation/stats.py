"""Statistical inference for forecasts and strategy returns.

  nw_tstat / nw_var_mean     Newey-West (Bartlett) HAC inference for a mean; default lag floor(4 (T/100)^(2/9))
  stationary bootstrap       Politis & Romano (1994): resample blocks of geometric length (mean `block` months)
  sharpe_diff_test           Ledoit & Wolf (2008): HAC delta-method standard error of a Sharpe-ratio difference and
                             a studentized bootstrap p-value
  holm / bh                  Holm (1979) step-down FWER and Benjamini-Hochberg (1995) FDR adjustments
  romano_wolf                Romano & Wolf (2005) bootstrap step-down FWER that uses the tests' dependence
  spa_test                   Hansen (2005) superior predictive ability: does the best of m methods beat the benchmark
                             once the search over m is accounted for (consistent, lower and upper p-values)
  psr / dsr                  Probabilistic and Deflated Sharpe ratio (Bailey & López de Prado 2012, 2014)
  pbo_cscv                   Probability of backtest overfitting by combinatorially symmetric cross-validation
                             (Bailey, Borwein, López de Prado & Zhu 2017)
  effective_trials           number of effectively independent trials from the correlation of their returns
"""
from __future__ import annotations

from itertools import combinations
from math import e as EULER_E

import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------------------- HAC
def nw_lag(T: int) -> int:
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def nw_var_mean(x: np.ndarray, lags: int | None = None) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 3:
        return np.nan
    u = x - x.mean()
    L = nw_lag(T) if lags is None else lags
    s = u @ u / T
    for lag in range(1, L + 1):
        s += 2 * (1 - lag / (L + 1)) * (u[lag:] @ u[:-lag]) / T
    return float(max(s, 0.0) / T)


def nw_tstat(x: np.ndarray, lags: int | None = None) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    v = nw_var_mean(x, lags)
    return float(x.mean() / np.sqrt(v)) if v and v > 0 else np.nan


def hac_cov(U: np.ndarray, lags: int) -> np.ndarray:
    T = len(U)
    S = U.T @ U / T
    for lag in range(1, lags + 1):
        G = U[lag:].T @ U[:-lag] / T
        S += (1 - lag / (lags + 1)) * (G + G.T)
    return S


# ----------------------------------------------------------------------------- basic performance
def sharpe(x: np.ndarray, periods: int = 12) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * np.sqrt(periods))


def max_drawdown(r: np.ndarray) -> float:
    """Largest peak-to-trough fall of the cumulative (compounded) return path, as a negative fraction."""
    r = np.nan_to_num(np.asarray(r, dtype=float))
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float((wealth / peak - 1).min()) if len(r) else np.nan


# ----------------------------------------------------------------------------- bootstrap
def stationary_indices(T: int, block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / max(block, 1.0)
    idx = np.empty(T, dtype=np.int64)
    idx[0] = rng.integers(T)
    jumps = rng.random(T) < p
    starts = rng.integers(T, size=T)
    for t in range(1, T):
        idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % T
    return idx


def bootstrap(stat_fn, data: np.ndarray, B: int = 2000, block: float = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = len(data)
    return np.array([stat_fn(data[stationary_indices(T, block, rng)]) for _ in range(B)])


def bootstrap_ci(stat_fn, data: np.ndarray, B: int = 2000, block: float = 6, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    draws = bootstrap(stat_fn, data, B, block, seed)
    draws = draws[np.isfinite(draws)]
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


# ----------------------------------------------------------------------------- Sharpe difference (Ledoit-Wolf)
def _sr_diff_and_se(r1, r2, lags):
    T = len(r1)
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    v1, v2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    if v1 <= 0 or v2 <= 0:
        return np.nan, np.nan
    d = mu1 / np.sqrt(v1) - mu2 / np.sqrt(v2)
    grad = np.array([g1 / v1 ** 1.5, -g2 / v2 ** 1.5, -mu1 / (2 * v1 ** 1.5), mu2 / (2 * v2 ** 1.5)])
    Y = np.column_stack([r1 - mu1, r2 - mu2, r1 ** 2 - g1, r2 ** 2 - g2])
    psi = hac_cov(Y, lags)
    return d, float(np.sqrt(max(grad @ psi @ grad, 1e-300) / T))


def sharpe_diff_test(r1: np.ndarray, r2: np.ndarray, B: int = 2000, block: float = 6, seed: int = 0,
                     periods: int = 12) -> dict:
    """Two-sided test of SR(r1) = SR(r2) on paired returns (Ledoit & Wolf 2008, studentized stationary bootstrap)."""
    ok = np.isfinite(r1) & np.isfinite(r2)
    r1, r2 = np.asarray(r1, float)[ok], np.asarray(r2, float)[ok]
    T = len(r1)
    L = nw_lag(T)
    d, se = _sr_diff_and_se(r1, r2, L)
    rng = np.random.default_rng(seed)
    tb, db = np.empty(B), np.empty(B)
    for b in range(B):
        i = stationary_indices(T, block, rng)
        dd, ss = _sr_diff_and_se(r1[i], r2[i], L)
        db[b], tb[b] = dd, (dd - d) / ss
    t = d / se
    tb = tb[np.isfinite(tb)]
    k = np.sqrt(periods)
    return {"diff": float(d * k), "se": float(se * k), "t": float(t),
            "p_boot": float(np.mean(np.abs(tb) >= abs(t))), "p_normal": float(2 * sps.norm.sf(abs(t))),
            "ci_lo": float(np.quantile(db, 0.025) * k), "ci_hi": float(np.quantile(db, 0.975) * k)}


def joint_bootstrap_t(D: np.ndarray, B: int = 2000, block: float = 6, seed: int = 0, stat: str = "mean",
                      bench: np.ndarray | None = None, R: np.ndarray | None = None):
    """Observed and bootstrap-centred t statistics for m columns resampled with the SAME blocks (keeps dependence).

    stat='mean': D (T x m) are paired differences (e.g. IC of method - IC of benchmark), t = mean / NW se.
    stat='sharpe': R (T x m) strategy returns and bench (T,), t = (SR_j - SR_bench) / LW se.
    Returns (t_obs (m,), t_boot (B x m))."""
    rng = np.random.default_rng(seed)
    if stat == "mean":
        T, m = D.shape
        L = nw_lag(T)
        se = np.array([np.sqrt(nw_var_mean(D[:, j], L)) for j in range(m)])
        t_obs = D.mean(0) / se
        tb = np.empty((B, m))
        mu = D.mean(0)
        for b in range(B):
            i = stationary_indices(T, block, rng)
            Db = D[i]
            seb = np.array([np.sqrt(nw_var_mean(Db[:, j], L)) for j in range(m)])
            tb[b] = (Db.mean(0) - mu) / seb
        return t_obs, tb
    if stat == "sharpe":
        T, m = R.shape
        L = nw_lag(T)
        obs = [_sr_diff_and_se(R[:, j], bench, L) for j in range(m)]
        d = np.array([o[0] for o in obs])
        t_obs = d / np.array([o[1] for o in obs])
        tb = np.empty((B, m))
        for b in range(B):
            i = stationary_indices(T, block, rng)
            for j in range(m):
                dd, ss = _sr_diff_and_se(R[i, j], bench[i], L)
                tb[b, j] = (dd - d[j]) / ss
        return t_obs, tb
    raise ValueError(stat)


# ----------------------------------------------------------------------------- multiple testing
def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for k, i in enumerate(order):
        rank = m - k
        run = min(run, p[i] * m / rank)
        adj[i] = run
    return adj


def romano_wolf(t_obs: np.ndarray, t_boot: np.ndarray) -> np.ndarray:
    """Step-down adjusted two-sided p-values from centred bootstrap t statistics (B x m)."""
    a = np.abs(t_obs)
    order = np.argsort(-a)
    adj = np.empty(len(a))
    prev = 0.0
    for j, k in enumerate(order):
        rest = order[j:]
        mx = np.nanmax(np.abs(t_boot[:, rest]), axis=1)
        pk = max(float(np.mean(mx >= a[k])), prev)
        adj[k] = pk
        prev = pk
    return adj


def spa_test(D: np.ndarray, B: int = 2000, block: float = 6, seed: int = 0) -> dict:
    """Hansen (2005) SPA. D (T x m): performance of each method minus the benchmark (positive = better).
    H0: no method is better than the benchmark. Returns consistent / lower / upper p-values and the statistic."""
    D = np.asarray(D, float)
    T, m = D.shape
    dbar = D.mean(0)
    L = nw_lag(T)
    omega = np.array([np.sqrt(T * nw_var_mean(D[:, j], L)) for j in range(m)])
    omega = np.where(omega > 0, omega, np.nan)
    stat = max(np.nanmax(np.sqrt(T) * dbar / omega), 0.0)
    # Bootstrap null: Z* = dbar* - g(dbar). g_u(x) = x recentres every model at zero (White's reality check, upper
    # p-value); g_l(x) = max(x, 0) leaves poor models at their negative means (lower p-value); g_c keeps a model at
    # its mean only when it is significantly poor (sqrt(T) dbar / omega < -sqrt(2 log log T)), Hansen's consistent p.
    thr = -np.sqrt(2 * np.log(np.log(T)))
    g_c = dbar * (np.sqrt(T) * dbar / omega >= thr)
    g_l = np.maximum(dbar, 0.0)
    g_u = dbar
    rng = np.random.default_rng(seed)
    cnt = np.zeros(3)
    for _ in range(B):
        i = stationary_indices(T, block, rng)
        db = D[i].mean(0)
        for k, g in enumerate((g_c, g_l, g_u)):
            tb = max(np.nanmax(np.sqrt(T) * (db - g) / omega), 0.0)
            cnt[k] += tb >= stat
    p = cnt / B
    best = int(np.nanargmax(np.sqrt(T) * dbar / omega))
    return {"stat": float(stat), "p_consistent": float(p[0]), "p_lower": float(p[1]), "p_upper": float(p[2]),
            "best_index": best}


# ----------------------------------------------------------------------------- Sharpe-ratio deflation
def psr(sr: float, T: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio: P(true SR > sr_star). sr per period (not annualized); kurt is raw (normal = 3)."""
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if den <= 0:
        return np.nan
    return float(sps.norm.cdf((sr - sr_star) * np.sqrt(T - 1) / np.sqrt(den)))


def expected_max_sr(n_trials: float, var_sr: float) -> float:
    """Expected maximum Sharpe ratio among n_trials independent trials with true SR = 0 and cross-trial variance
    var_sr (Bailey & López de Prado 2014, False Strategy Theorem)."""
    if n_trials <= 1:
        return 0.0
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * EULER_E))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def dsr(returns: np.ndarray, n_trials: float, var_sr: float) -> dict:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    sk = float(sps.skew(r))
    ku = float(sps.kurtosis(r, fisher=False))
    sr0 = expected_max_sr(n_trials, var_sr)
    return {"sr": float(sr), "sr0": sr0, "dsr": psr(sr, T, sk, ku, sr0), "psr0": psr(sr, T, sk, ku, 0.0),
            "skew": sk, "kurt": ku, "T": T, "n_trials": float(n_trials)}


def effective_trials(R: np.ndarray) -> dict:
    """Effective number of independent trials from the correlation matrix of trial returns (T x N): the eigenvalue
    participation ratio (sum lambda)^2 / sum lambda^2, and N / (1 + (N - 1) mean pairwise correlation)."""
    R = R[:, np.isfinite(R).all(0)]
    C = np.corrcoef(R.T)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    N = C.shape[0]
    rho = (C.sum() - N) / (N * (N - 1)) if N > 1 else 0.0
    return {"n": N, "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "mean_corr": float(rho), "n_eff_corr": float(N / (1 + (N - 1) * max(rho, 0.0)))}


# ----------------------------------------------------------------------------- PBO (CSCV)
def pbo_cscv(M: np.ndarray, S: int = 16, periods: int = 12) -> dict:
    """M: T x N matrix of strategy returns (one column per configuration tried). Splits the T rows into S blocks;
    for every half/half split picks the in-sample best (Sharpe) and records its out-of-sample relative rank."""
    M = np.asarray(M, float)
    M = M[:, np.isfinite(M).all(0)]
    T, N = M.shape
    T = (T // S) * S
    M = M[:T]
    blocks = M.reshape(S, T // S, N)
    s1, s2, cnt = blocks.sum(1), (blocks ** 2).sum(1), T // S          # (S, N)
    combos = np.array(list(combinations(range(S), S // 2)))
    sel = np.zeros((len(combos), S))
    np.put_along_axis(sel, combos, 1.0, axis=1)

    def sr(mask):
        n = mask.sum(1, keepdims=True) * cnt
        m1 = (mask @ s1) / n
        m2 = (mask @ s2) / n
        return m1 / np.sqrt(np.maximum(m2 - m1 ** 2, 1e-300)) * np.sqrt(periods)

    is_sr, oos_sr = sr(sel), sr(1 - sel)
    best = np.argmax(is_sr, axis=1)
    oos_best = oos_sr[np.arange(len(combos)), best]
    rank = (oos_sr < oos_best[:, None]).sum(1) + 1                     # 1 = worst out of sample
    omega = rank / (N + 1)
    logit = np.log(omega / (1 - omega))
    is_best = is_sr[np.arange(len(combos)), best]
    slope = np.polyfit(is_best, oos_best, 1)[0] if len(combos) > 2 else np.nan
    return {"pbo": float(np.mean(logit <= 0)), "logits": logit, "is_best_sr": is_best, "oos_best_sr": oos_best,
            "degradation_slope": float(slope), "p_oos_loss": float(np.mean(oos_best < 0)), "n_strategies": N,
            "n_splits": len(combos)}
