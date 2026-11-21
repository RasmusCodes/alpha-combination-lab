"""From experiment outputs to the report's numbers. Pure functions; scripts/analyze_study.py writes the files.

Timing convention: a forecast formed at the close of month t earns the return of month t+1. Series below are
indexed by FORMATION month in the panel; return series handed to factor regressions are re-indexed to the month
in which the return is earned.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphacomb import study
from alphacomb.evaluation import factors as fac
from alphacomb.evaluation import metrics, portfolio, stability, stats
from alphacomb.panel import Panel

PER = 12


# ----------------------------------------------------------------------------- loading
def load_dir(d: Path, names=None) -> dict:
    d = Path(d)
    out = {}
    for f in sorted(d.glob("*.pkl")):
        if names is None or f.stem in names:
            out[f.stem] = study.load(f)
    return out


def single_signal_preds(p: Panel) -> dict[str, np.ndarray]:
    return {f"SIG:{n}": p.X[:, k].copy() for k, n in enumerate(p.names)}


# ----------------------------------------------------------------------------- per-method evaluation
class Evaluated:
    """Everything derived from one forecast vector over a set of OOS months."""

    def __init__(self, name: str, pred: np.ndarray, p: Panel, months: np.ndarray, weights: np.ndarray | None = None,
                 cost_mult: float = 1.0):
        self.name, self.pred, self.months = name, pred, np.asarray(months)
        self.ic = metrics.monthly_ic(pred, p.ret, p.offsets, months=self.months)
        self.pic = metrics.monthly_ic(pred, p.ret, p.offsets, months=self.months, rank=False)
        ev = portfolio.evaluate(pred, p, self.months, p.extras["cost"] * cost_mult)
        self.w, self.gross, self.turnover, self.cost, self.trades = (ev["weights"], ev["gross"], ev["turnover"],
                                                                      ev["cost"], ev["trades"])
        self.net = ev["net"]
        self.net_high = self.gross - 2 * self.cost
        q = portfolio.evaluate(pred, p, self.months, p.extras["cost"] * cost_mult, scheme="quantile", q=0.2)
        self.q_gross, self.q_net = q["gross"], q["net"]
        self.q_turnover = q["turnover"]
        self.weights = weights

    def s(self, arr, months=None) -> np.ndarray:
        m = self.months if months is None else months
        return arr[m]


def calibrated_r2(pred: np.ndarray, p: Panel, months: np.ndarray, W: np.ndarray | None, mom, scale_free: bool) -> float:
    """Pooled OOS R^2 of the rank-gauss target. Methods fit to the target are used as they predict; scale-free
    composites (equal weights, IC weights, mean-variance, multiplicative weights) get one coefficient per month,
    the in-sample slope of the target on the composite over all months whose outcome was known (ex ante)."""
    yhat = np.full(p.n, np.nan)
    if not scale_free or W is None:
        idx = np.concatenate([np.arange(p.offsets[t], p.offsets[t + 1]) for t in months])
        yhat[idx] = pred[idx]
    else:
        Gc, cc = np.cumsum(mom.G, axis=0), np.cumsum(mom.c, axis=0)
        for t in months:
            w = W[t]
            if not np.isfinite(w).all() or t < 2:
                continue
            G, c = Gc[t - 2], cc[t - 2]
            den = w @ G @ w
            s = (w @ c) / den if den > 0 else 0.0
            r = p.rows(t, t + 1)
            yhat[r] = s * pred[r]
    idx = np.concatenate([np.arange(p.offsets[t], p.offsets[t + 1]) for t in months])
    return metrics.r2_oos(p.y[idx], yhat[idx])


def summary_row(e: Evaluated, p: Panel, ff: pd.DataFrame | None = None) -> dict:
    m = e.months
    ics = metrics.ic_summary(e.ic[m])
    g, n, nh = e.gross[m], e.net[m], e.net_high[m]
    row = {"rank_ic": ics["mean"], "rank_ic_t": ics["t"], "ic_ir": ics["ir"] * np.sqrt(PER), "hit": ics["hit"],
           "ic": float(np.nanmean(e.pic[m])),
           "gross_ret": float(np.nanmean(g) * PER), "vol": float(np.nanstd(g, ddof=1) * np.sqrt(PER)),
           "sharpe_gross": stats.sharpe(g), "sharpe_net": stats.sharpe(n), "sharpe_net_high": stats.sharpe(nh),
           "net_ret": float(np.nanmean(n) * PER), "max_dd_net": stats.max_drawdown(n),
           "turnover": float(np.nanmean(e.turnover[m[1:]]) * PER), "cost": float(np.nanmean(e.cost[m]) * PER),
           "breakeven_mult": float(np.nanmean(g) / np.nanmean(e.cost[m])) if np.nanmean(e.cost[m]) > 0 else np.nan,
           "q_sharpe_gross": stats.sharpe(e.q_gross[m]), "q_sharpe_net": stats.sharpe(e.q_net[m]),
           "q_turnover": float(np.nanmean(e.q_turnover[m[1:]]) * PER)}
    if ff is not None:
        y = pd.Series(n, index=p.dates[m] + pd.offsets.MonthEnd(1))
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "ST_Rev"]
        r = fac.regress(y, ff[cols])
        row.update({"alpha": r.get("alpha"), "alpha_t": r.get("alpha_t"), "r2_factors": r.get("r2"),
                    **{f"b_{c}": r.get(f"b_{c}") for c in cols}, **{f"t_{c}": r.get(f"t_{c}") for c in cols}})
    return row


def sharpe_ci(x: np.ndarray, B: int, block: float, seed: int = 0):
    return stats.bootstrap_ci(lambda a: stats.sharpe(a), x[np.isfinite(x)], B, block, seed=seed)


# ----------------------------------------------------------------------------- hypothesis tests
def tests_vs(evals: dict[str, Evaluated], bench: str, B: int, block: float, seed: int = 0,
             metric_net: str = "net") -> pd.DataFrame:
    """H1 (net Sharpe) and H2 (rank IC) for every method against the benchmark, with Holm, BH and Romano-Wolf."""
    b = evals[bench]
    m = b.months
    names = [k for k in evals if k != bench]
    rows = []
    R = np.column_stack([getattr(evals[k], metric_net)[m] for k in names])
    rb = getattr(b, metric_net)[m]
    D = np.column_stack([evals[k].ic[m] - b.ic[m] for k in names])
    D = np.where(np.isfinite(D), D, 0.0)
    for j, k in enumerate(names):
        sd = stats.sharpe_diff_test(R[:, j], rb, B=B, block=block, seed=seed + j)
        tic = stats.nw_tstat(D[:, j])
        rows.append({"method": k, "sharpe_diff": sd["diff"], "sharpe_ci_lo": sd["ci_lo"], "sharpe_ci_hi": sd["ci_hi"],
                     "sharpe_t": sd["t"], "p_sharpe": sd["p_boot"], "ic_diff": float(D[:, j].mean()),
                     "ic_t": tic, "p_ic": float(2 * stats.sps.norm.sf(abs(tic))),
                     "corr_with_bench": float(np.corrcoef(R[:, j], rb)[0, 1])})
    out = pd.DataFrame(rows).set_index("method")
    for col in ("p_sharpe", "p_ic"):
        out[col + "_holm"] = stats.holm(out[col].to_numpy())
        out[col + "_bh"] = stats.bh(out[col].to_numpy())
    t_obs, tb = stats.joint_bootstrap_t(None, B=min(B, 2000), block=block, seed=seed, stat="sharpe", bench=rb, R=R)
    out["p_sharpe_rw"] = stats.romano_wolf(t_obs, tb)
    t_obs, tb = stats.joint_bootstrap_t(D, B=min(B, 2000), block=block, seed=seed + 1, stat="mean")
    out["p_ic_rw"] = stats.romano_wolf(t_obs, tb)
    return out


def spa(evals: dict[str, Evaluated], bench: str, names: list[str], B: int, block: float, seed: int = 0) -> dict:
    b = evals[bench]
    m = b.months
    Dr = np.column_stack([evals[k].net[m] - b.net[m] for k in names])
    Di = np.column_stack([np.nan_to_num(evals[k].ic[m] - b.ic[m]) for k in names])
    s1 = stats.spa_test(Dr, B=B, block=block, seed=seed)
    s2 = stats.spa_test(Di, B=B, block=block, seed=seed + 1)
    s1["best"], s2["best"] = names[s1["best_index"]], names[s2["best_index"]]
    return {"net_return": s1, "rank_ic": s2}


# ----------------------------------------------------------------------------- slices
def subperiods(evals: dict[str, Evaluated], p: Panel, periods) -> pd.DataFrame:
    rows = []
    for a, z in periods:
        for k, e in evals.items():
            m = e.months[(p.dates[e.months] >= pd.Timestamp(a)) & (p.dates[e.months] <= pd.Timestamp(z))]
            rows.append({"period": f"{a[:4]}-{z[:4]}", "method": k, "rank_ic": float(np.nanmean(e.ic[m])),
                         "sharpe_net": stats.sharpe(e.net[m]), "sharpe_gross": stats.sharpe(e.gross[m]),
                         "months": len(m)})
    return pd.DataFrame(rows)


def regimes(evals: dict[str, Evaluated], p: Panel, bench: str = "EW") -> pd.DataFrame:
    mi = p.month_info
    rows = []
    b = evals[bench]
    nber_next = np.r_[mi["nber"].to_numpy()[1:], False]
    slices = {f"volatility={v}": mi["volatility"].to_numpy() == v for v in ("low", "high")}
    slices |= {f"state={v}": mi["state"].to_numpy() == v for v in ("up", "down")}
    slices |= {"recession (NBER, next month)": nber_next, "expansion (NBER, next month)": ~nber_next}
    for lab, mask in slices.items():
        for k, e in evals.items():
            m = e.months[mask[e.months]]
            d = e.ic[m] - b.ic[m]
            rows.append({"slice": lab, "method": k, "months": len(m), "rank_ic": float(np.nanmean(e.ic[m])),
                         "ic_minus_bench": float(np.nanmean(d)), "ic_diff_t": stats.nw_tstat(d),
                         "net_ret": float(np.nanmean(e.net[m]) * PER), "sharpe_net": stats.sharpe(e.net[m])})
    return pd.DataFrame(rows)


def stability_table(results: dict, evals: dict[str, Evaluated]) -> pd.DataFrame:
    rows = []
    for k, res in results.items():
        e = evals[k]
        ws = stability.weight_stability(res.weights, e.months)
        ss = stability.selection_stability([s for s in res.selections if s.get("tuned")])
        rows.append({"method": k, "weight_corr_refits": ws.get("mean_corr"), "weight_l1_change": ws.get(
            "mean_l1_change"), "sign_flip_rate": ws.get("sign_flip_rate"),
            "monthly_weight_turnover": stability.monthly_weight_turnover(res.weights, e.months),
            "tunings": ss.get("n_refits"), "selection_change_rate": ss.get("change_rate"),
            "distinct_selections": ss.get("n_distinct"), "modal_selection": ss.get("modal"),
            "modal_share": ss.get("modal_share")})
    return pd.DataFrame(rows).set_index("method")


def importance_table(results: dict) -> pd.DataFrame:
    rows = {}
    for k, res in results.items():
        if not res.importance:
            continue
        rows[k] = {g: float(np.nanmean(v)) for g, v in res.importance.items()}
    return pd.DataFrame(rows).T


def weights_by_year(res, p: Panel) -> pd.DataFrame:
    W = res.weights[res.months]
    s = np.abs(W).sum(1, keepdims=True)
    W = W / np.where(s > 0, s, 1)
    df = pd.DataFrame(W, index=p.dates[res.months], columns=p.names)
    return df.groupby(df.index.year).mean()


def capacity(e: Evaluated, p: Panel, aum_grid, impact_y: float = 1.0) -> dict:
    m = e.months
    out = {}
    for aum in aum_grid:
        c = portfolio.trade_costs(e.trades, p.extras["cost"], p.T, aum=aum, sigma=p.extras["sigma"].astype(float),
                                  adv=p.extras["dvol"].astype(float), impact_y=impact_y)
        net = e.gross[m] - c[m]
        out[aum] = {"sharpe_net": stats.sharpe(net), "net_ret": float(np.nanmean(net) * PER)}
    return out


def breakeven_aum(cap: dict) -> float:
    """Smallest AUM on the grid (log-interpolated) at which the net mean return reaches zero."""
    a = np.array(sorted(cap))
    r = np.array([cap[x]["net_ret"] for x in a])
    if r[0] <= 0:
        return float(a[0])
    for i in range(1, len(a)):
        if r[i] <= 0:
            la, lb = np.log10(a[i - 1]), np.log10(a[i])
            return float(10 ** (la + (lb - la) * r[i - 1] / (r[i - 1] - r[i])))
    return float("inf")


# ----------------------------------------------------------------------------- E6, E4, E1
def complexity_frame(d: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(d).glob("*.pkl")):
        meth, tag = f.stem.split("__", 1)
        s = study.load(f)
        m = s["months"]
        rows.append({"method": meth, "config": tag, "rank_ic": float(np.nanmean(s["rank_ic"][m])),
                     "sharpe_gross": stats.sharpe(s["gross"][m]), "sharpe_net": stats.sharpe(s["net"][m]),
                     "turnover": float(np.nanmean(s["turnover"][m[1:]]) * PER), "file": f.name})
    return pd.DataFrame(rows)


def complexity_returns(d: Path) -> pd.DataFrame:
    cols = {}
    for f in sorted(Path(d).glob("*.pkl")):
        s = study.load(f)
        cols[f.stem] = pd.Series(s["net"][s["months"]], index=s["months"])
    return pd.DataFrame(cols)


def leakage_frame(main_res: dict, leak_dirs: dict[str, Path], p: Panel, horizon_months: int = 60) -> pd.DataFrame:
    """Per tuning event: the selected configuration's inner-validation rank IC vs the realized rank IC over the next
    `horizon_months` OOS months, for each inner-CV scheme."""
    rows = []
    sources = {"forward": {k: ("full", v) for k, v in main_res.items()}}
    for scheme, d in leak_dirs.items():
        sources[scheme] = {f.stem: ("compact", study.load(f)) for f in sorted(Path(d).glob("*.pkl"))}
    for scheme, items in sources.items():
        for meth, (kind, obj) in items.items():
            if meth not in ("RIDGE", "ENET", "GBM"):
                continue
            sels = obj.selections if kind == "full" else obj["selections"]
            if kind == "full":
                ic = metrics.monthly_ic(obj.pred, p.ret, p.offsets, months=obj.months)
                months = obj.months
            else:
                ic, months = obj["rank_ic"], obj["months"]
            for s in sels:
                if not s.get("tuned") or not s.get("cv"):
                    continue
                chosen = next(r for r in s["cv"] if r["config"] == s["params"])
                t0 = p.t_of(s["refit"])
                fut = months[(months >= t0) & (months < t0 + horizon_months)]
                rows.append({"scheme": scheme, "method": meth, "refit": s["refit"], "params": str(s["params"]),
                             "inner_ic": chosen["mean"], "realized_ic": float(np.nanmean(ic[fut])),
                             "optimism": chosen["mean"] - float(np.nanmean(ic[fut]))})
    return pd.DataFrame(rows)


def synthetic_frame(d: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(d).glob("*.pkl")):
        s = study.load(f)
        o = float(np.nanmean(s["oracle_rank_ic"]))
        for k, v in s["methods"].items():
            rows.append({"dgp": s["dgp"], "ic": s["ic"], "seed": s["seed"], "method": k,
                         "rank_ic": float(np.nanmean(v["rank_ic"])), "oracle_ic": o,
                         "ratio": float(np.nanmean(v["rank_ic"])) / o if o else np.nan, "sharpe": v["sharpe"],
                         "weight_corr": v["weight_corr"]})
    return pd.DataFrame(rows)
