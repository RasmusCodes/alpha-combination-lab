"""From forecasts to portfolios: weights, returns, turnover, costs and capacity.

Primary portfolio (pre-specified): RANK-WEIGHTED dollar-neutral long-short. Within each formation month the forecast
ranks are centred, w_i ~ rank_i - mean rank, scaled so the long side sums to +1 and the short side to -1. It holds
every stock (lower turnover than decile sorts) and is close to the forecast's own factor-mimicking portfolio.
Secondary: equal-weighted top-minus-bottom quintile.

Timing: weights formed at the close of month t earn the return of month t+1 (ret row of month t). Trades at t are
from the drifted month t-1 weights to the new targets; a stock that leaves the universe is sold at t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def rank_weights(pred: np.ndarray, offsets: np.ndarray, months, min_n: int = 10) -> np.ndarray:
    w = np.full(len(pred), np.nan)
    for t in months:
        a, b = offsets[t], offsets[t + 1]
        p = pred[a:b]
        ok = np.isfinite(p)
        if ok.sum() < min_n:
            continue
        u = rankdata(p[ok])
        u = u - u.mean()
        s = np.abs(u).sum()
        ww = np.zeros(b - a)
        if s > 0:
            ww[ok] = 2.0 * u / s
        w[a:b] = ww
    return w


def quantile_weights(pred: np.ndarray, offsets: np.ndarray, months, q: float = 0.2) -> np.ndarray:
    w = np.full(len(pred), np.nan)
    for t in months:
        a, b = offsets[t], offsets[t + 1]
        p = pred[a:b]
        ok = np.isfinite(p)
        n = int(ok.sum())
        if n < 10:
            continue
        k = max(int(round(q * n)), 1)
        r = np.full(b - a, np.nan)
        r[ok] = rankdata(p[ok], method="ordinal")
        ww = np.zeros(b - a)
        ww[ok & (r > n - k)] = 1.0 / k
        ww[ok & (r <= k)] = -1.0 / k
        w[a:b] = ww
    return w


@dataclass
class Trades:
    month: np.ndarray      # formation month of each trade
    row: np.ndarray        # panel row whose cost inputs apply (current row; previous row for exits)
    size: np.ndarray       # |change in weight|


def gross_returns(w: np.ndarray, ret: np.ndarray, offsets: np.ndarray, months) -> np.ndarray:
    """(T,) portfolio return earned by the month-t weights (NaN returns count as 0: suspended / unknown)."""
    out = np.full(len(offsets) - 1, np.nan)
    for t in months:
        a, b = offsets[t], offsets[t + 1]
        ww = w[a:b]
        if not np.isfinite(ww).any():
            continue
        out[t] = float(np.nansum(np.nan_to_num(ww) * np.nan_to_num(ret[a:b])))
    return out


def trades(w: np.ndarray, ret: np.ndarray, ids: np.ndarray, offsets: np.ndarray, months) -> tuple[np.ndarray, Trades]:
    """One-way turnover per month (sum |trade| / 2) and the individual trades, with drift between rebalances."""
    T = len(offsets) - 1
    turn = np.full(T, np.nan)
    tm, tr, ts = [], [], []
    prev_ids, prev_w, prev_rows = None, None, None
    months = list(months)
    for j, t in enumerate(months):
        a, b = offsets[t], offsets[t + 1]
        cur_ids, cur_w = ids[a:b], np.nan_to_num(w[a:b])
        cur_rows = np.arange(a, b)
        if prev_ids is None or months[j - 1] != t - 1:
            delta_cur = np.abs(cur_w)
            exits_rows, exits_size = np.array([], dtype=int), np.array([])
        else:
            pr = np.nan_to_num(ret[prev_rows])
            rp = float(np.sum(prev_w * pr))
            drift = prev_w * (1 + pr) / (1 + rp)
            common, ic, ip = np.intersect1d(cur_ids, prev_ids, assume_unique=True, return_indices=True)
            old = np.zeros(b - a)
            old[ic] = drift[ip]
            delta_cur = np.abs(cur_w - old)
            gone = np.ones(len(prev_ids), dtype=bool)
            gone[ip] = False
            exits_rows, exits_size = prev_rows[gone], np.abs(drift[gone])
        turn[t] = 0.5 * (delta_cur.sum() + exits_size.sum())
        tm.append(np.full(len(cur_rows) + len(exits_rows), t))
        tr.append(np.concatenate([cur_rows, exits_rows]))
        ts.append(np.concatenate([delta_cur, exits_size]))
        prev_ids, prev_w, prev_rows = cur_ids, cur_w, cur_rows
    if not tm:
        return turn, Trades(np.array([], int), np.array([], int), np.array([]))
    return turn, Trades(np.concatenate(tm), np.concatenate(tr), np.concatenate(ts))


def trade_costs(tr: Trades, cost_rate: np.ndarray, T: int, aum: float | None = None,
                sigma: np.ndarray | None = None, adv: np.ndarray | None = None, impact_y: float = 1.0) -> np.ndarray:
    """(T,) cost per month as a fraction of capital: sum |trade| x (one-way rate [+ impact at this AUM])."""
    rate = np.nan_to_num(cost_rate[tr.row], nan=np.nanmedian(cost_rate))
    c = tr.size * rate
    if aum is not None and aum > 0:
        s = np.nan_to_num(sigma[tr.row], nan=np.nanmedian(sigma))
        v = adv[tr.row]
        v = np.where(np.isfinite(v) & (v > 0), v, np.nanmedian(adv))
        c = c + tr.size * impact_y * s * np.sqrt(tr.size * aum / v)
    out = np.bincount(tr.month, weights=c, minlength=T).astype(float)
    return out


def evaluate(pred: np.ndarray, panel, months, cost_rate: np.ndarray | None = None, scheme: str = "rank",
             q: float = 0.2) -> dict:
    """Gross and net monthly returns and turnover for a forecast vector (months = OOS formation months)."""
    w = rank_weights(pred, panel.offsets, months) if scheme == "rank" else quantile_weights(pred, panel.offsets,
                                                                                            months, q)
    g = gross_returns(w, panel.ret, panel.offsets, months)
    turn, tr = trades(w, panel.ret, panel.ids, panel.offsets, months)
    cost = trade_costs(tr, cost_rate, panel.T) if cost_rate is not None else np.zeros(panel.T)
    m = np.asarray(list(months))
    net = g.copy()
    net[m] = g[m] - cost[m]
    return {"weights": w, "gross": g, "net": net, "turnover": turn, "cost": cost, "trades": tr}
