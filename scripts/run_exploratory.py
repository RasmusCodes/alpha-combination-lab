"""Exploratory analyses decided AFTER the confirmatory results were seen (registry kind 'exploratory').

X1  market-hedged returns: each method's net long-short return minus an ex-ante market hedge (beta from the
    method's previous 36 months of returns on the market factor). Asks whether the ranking of methods survives once
    differences in market exposure are removed (equal weights carries a beta of about -0.3 from its low-risk signals).
X2  what drives MVO: minimum-variance mix (covariance only), MVO with an unshrunk sample covariance, and
    inverse-volatility weights, all run through the same walk-forward protocol.

    python scripts/run_exploratory.py
writes reports/tables/exploratory_*.csv and reports/exploratory.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_study as rs                                            # noqa: E402

from alphacomb import analysis as A                               # noqa: E402
from alphacomb import models, paths, study                        # noqa: E402
from alphacomb.evaluation import stats                            # noqa: E402
from alphacomb.models.exploratory import EXPLORATORY              # noqa: E402
from alphacomb.registry import Registry                           # noqa: E402
from alphacomb.reporting import md_table                          # noqa: E402
from alphacomb.validation import engine                           # noqa: E402

T_DIR = paths.REPORTS / "tables"


def hedged(net: np.ndarray, mkt: np.ndarray, months: np.ndarray, window: int = 36) -> np.ndarray:
    """net, mkt indexed by formation month (mkt = market excess return earned the following month)."""
    out = np.full(len(net), np.nan)
    for j, t in enumerate(months):
        if j < window:
            continue
        past = months[j - window:j]               # their returns were earned by the end of month t
        y, x = net[past], mkt[past]
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < 24:
            continue
        b = np.cov(x[ok], y[ok], ddof=1)[0, 1] / np.var(x[ok], ddof=1)
        out[t] = net[t] - b * mkt[t]
    return out


def main() -> None:
    cfg, p, mom = rs.real_inputs()
    reg = Registry()
    fp = study.data_fingerprint(p)
    order = [m for m in models.ORDER if (paths.RESULTS / "main" / f"{m}.pkl").exists()]
    months = study.load(paths.RESULTS / "main" / "EW.pkl").months
    ff = pd.read_parquet(paths.PROCESSED / "factors_monthly.parquet")
    earn = p.dates + pd.offsets.MonthEnd(1)
    mkt = ff["Mkt-RF"].reindex(earn).to_numpy()
    md = ["# Exploratory analyses (decided after the confirmatory results; not pre-registered)", ""]

    # ------------------------------------------------------------------ X2 first: run the variants
    xdir = paths.RESULTS / "exploratory"
    wf = study.wf_config(cfg, importance=False)
    for name, cls in EXPLORATORY.items():
        f = xdir / f"{name}.pkl"
        if f.exists():
            continue
        run_id = reg.start("real/exploratory/mvo_decomposition", {"method": name, "protocol": cfg["walk_forward"],
                                                                  "data": fp}, ["X2"], "exploratory", fp)
        res = engine.run(cls(), p, mom, wf)
        study.save(res, f)
        reg.finish(run_id, results=rs.summarize(res, p))

    ev = {}
    for m in order + list(EXPLORATORY):
        d = paths.RESULTS / ("main" if m in order else "exploratory") / f"{m}.pkl"
        r = study.load(d)
        ev[m] = A.Evaluated(m, r.pred, p, r.months, r.weights, keep_trades=False)
    rows = {m: {"rank_ic": float(np.nanmean(e.ic[months])), "sharpe_gross": stats.sharpe(e.gross[months]),
                "sharpe_net": stats.sharpe(e.net[months]), "turnover": float(np.nanmean(e.turnover[months[1:]]) * 12)}
            for m, e in ev.items()}
    x2 = pd.DataFrame(rows).T.loc[["EW", "MVO"] + list(EXPLORATORY)]
    for m in list(EXPLORATORY) + ["MVO"]:
        t = stats.sharpe_diff_test(ev[m].net[months], ev["EW"].net[months], B=2000, block=6, seed=5)
        x2.loc[m, "net_minus_EW"], x2.loc[m, "p"] = t["diff"], t["p_boot"]
    x2.to_csv(T_DIR / "exploratory_mvo_decomposition.csv")
    md += ["## X2: what drives MVO", "", md_table(x2.astype(float), {"rank_ic": 4, "turnover": 1, "p": 3}), ""]

    # ------------------------------------------------------------------ X1: market-hedged
    reg_id = reg.start("real/exploratory/market_hedged", {"window": 36, "methods": order, "data": fp}, ["X1"],
                       "exploratory", fp)
    H = {m: hedged(ev[m].net, mkt, months) for m in order + ["MINVAR"]}
    hm = months[np.isfinite(H["EW"][months])]
    x1 = {}
    for m, h in H.items():
        row = {"sharpe_net": stats.sharpe(ev[m].net[hm]), "sharpe_hedged": stats.sharpe(h[hm]),
               "beta_full": float(np.polyfit(mkt[hm], ev[m].net[hm], 1)[0])}
        if m != "EW":
            t = stats.sharpe_diff_test(h[hm], H["EW"][hm], B=2000, block=6, seed=7)
            row.update(hedged_minus_EW=t["diff"], p=t["p_boot"])
        x1[m] = row
    x1 = pd.DataFrame(x1).T
    x1["p_holm"] = np.nan
    mask = x1.index.isin([m for m in order if m != "EW"])
    x1.loc[mask, "p_holm"] = stats.holm(x1.loc[mask, "p"].to_numpy())
    x1.to_csv(T_DIR / "exploratory_market_hedged.csv")
    reg.finish(reg_id, results={"months": int(len(hm)), "ew_hedged_sharpe": float(x1.loc["EW", "sharpe_hedged"])})
    md += ["## X1: market-hedged net returns (ex-ante 36-month beta)", "",
           f"Months {p.dates[hm[0]].date()} to {p.dates[hm[-1]].date()} ({len(hm)}; the first 36 out-of-sample months "
           "estimate the first hedge ratio).", "", md_table(x1.astype(float), {"p": 3, "p_holm": 3, "beta_full": 2}), ""]
    # ------------------------------------------------------------------ X3: SPA on volatility-scaled returns
    # The pre-registered SPA compares MEAN net returns, so a method that earns the same mean with half the risk
    # gets no credit there. Scaling every series to the same ex-ante volatility (trailing 36 months, 10% a year)
    # makes the SPA's mean comparison a Sharpe-like comparison.
    reg_id = reg.start("real/exploratory/spa_vol_scaled", {"window": 36, "target_vol": 0.10, "data": fp}, ["X3"],
                       "exploratory", fp)

    def scaled(net):
        out = np.full(len(net), np.nan)
        for j, t in enumerate(months):
            if j < 36:
                continue
            sd = np.nanstd(net[months[j - 36:j]], ddof=1) * np.sqrt(12)
            out[t] = net[t] * 0.10 / sd if sd > 0 else np.nan
        return out

    Sx = {m: scaled(ev[m].net) for m in order}
    sm = months[np.isfinite(Sx["EW"][months])]
    complex_m = [m for m in order if m not in cfg["simple_methods"]]
    D = np.column_stack([Sx[m][sm] - Sx["EW"][sm] for m in complex_m])
    spa = stats.spa_test(D, B=5000, block=6, seed=13)
    spa["best"] = complex_m[spa["best_index"]]
    reg.finish(reg_id, results={k: v for k, v in spa.items() if k != "best_index"})
    md += ["## X3: SPA on volatility-scaled net returns (Sharpe-consistent)", "",
           f"Complex methods vs equal weights, {len(sm)} months: best {spa['best']}, consistent p = "
           f"{spa['p_consistent']:.3f} (lower {spa['p_lower']:.3f}, upper {spa['p_upper']:.3f}).", ""]
    (paths.REPORTS / "exploratory.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
