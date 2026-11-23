"""Build the signal extension (no returns used): processed I/B/E/S and OptionMetrics inputs, the seven new signals
as validated records, the extended modelling panel, and reports/data_integrity_extension.md.

    python scripts/build_extension.py [--skip-daily]

--skip-daily reuses data/processed/daily_features.parquet if it already has the rv1 column (realized volatility
of month t, needed by VRP); otherwise the daily features are rebuilt (existing columns unchanged).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from alphacomb import paths
from alphacomb.data import daily, ibes, optionm
from alphacomb.data.bundle import CRSPBundle
from alphacomb.reporting import md_table
from alphacomb.signals import diagnostics, registry
from alphacomb.signals.library import EXT_EARNINGS, EXT_OPTIONS, LIBRARY

RAW = paths.DATA / "raw" / "wrds_ext"
P = paths.PROCESSED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-daily", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    # ------------------------------------------------------------------ processed inputs
    link = pd.read_parquet(RAW / "ibes_link.parquet")
    stat = pd.read_parquet(RAW / "ibes_statsumu.parquet")
    actu = pd.read_parquet(RAW / "ibes_actu.parquet")
    fy1 = ibes.monthly_fy1(stat, link)
    fy1.to_parquet(P / "ibes_fy1.parquet", index=False)
    sur = ibes.quarterly_surprises(stat, actu, link)
    sur.to_parquet(P / "ibes_surprises.parquet", index=False)
    olink = pd.read_parquet(RAW / "om_link.parquet")
    surf = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob("om_vsurfd_*.parquet"))], ignore_index=True)
    iv = optionm.monthly_iv(surf, olink)
    iv.to_parquet(P / "om_iv.parquet", index=False)
    std = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob("om_stdopd_*.parquet"))], ignore_index=True)
    std = std.pivot_table(index=["secid", "date"], columns="cp_flag", values="impl_volatility", aggfunc="first")
    cross = surf[(surf["delta"].abs() == 50)].pivot_table(index=["secid", "date"], columns="cp_flag",
                                                          values="impl_volatility", aggfunc="first")
    j = std.join(cross, lsuffix="_std", rsuffix="_surf", how="inner").dropna()
    check = {"pairs": int(len(j)),
             "corr_call": float(np.corrcoef(j["C_std"], j["C_surf"])[0, 1]) if len(j) > 2 else None,
             "corr_put": float(np.corrcoef(j["P_std"], j["P_surf"])[0, 1]) if len(j) > 2 else None,
             "median_abs_diff_call": float((j["C_std"] - j["C_surf"]).abs().median()) if len(j) else None}
    print("built inputs:", len(fy1), "FY1 months;", len(sur), "surprises;", len(iv), "IV months; check", check,
          flush=True)
    # ------------------------------------------------------------------ daily features with rv1
    have = pd.read_parquet(P / "daily_features.parquet", columns=None).columns
    if not (a.skip_daily and "rv1" in have):
        feats = daily.build()
        feats.to_parquet(P / "daily_features.parquet", index=False)
        print("daily features rebuilt with rv1", flush=True)
    # ------------------------------------------------------------------ signals and the extended panel
    b = CRSPBundle()
    panel = pd.read_parquet(P / "panel.parquet")
    meta = []
    for name in EXT_EARNINGS + EXT_OPTIONS:
        s = registry.get(name)
        rec = s.records(b)
        rec.to_parquet(P / "signals" / f"{name}.parquet", index=False)
        v = rec[["instrument", "timestamp", "value"]].rename(columns={"timestamp": "date", "value": name})
        panel = panel.merge(v, on=["instrument", "date"], how="left")
        meta.append({**s.spec.as_dict(), "version": s.version, "records": len(rec),
                     "first": str(rec["timestamp"].min().date()) if len(rec) else ""})
        print(f"{name}: {len(rec):,} records", flush=True)
    panel.to_parquet(P / "panel_ext.parquet", index=False)
    # ------------------------------------------------------------------ integrity report (no returns)
    new = EXT_EARNINGS + EXT_OPTIONS
    allsig = LIBRARY + new
    d = panel.copy()
    d["year"] = d["date"].dt.year
    cov = d[d["year"] >= 1976].groupby((d["year"] // 5) * 5)[new].apply(lambda g: g.notna().mean()).round(3)
    cov.index = [f"{y}-{y + 4}" for y in cov.index]
    sign = {n: registry.get(n).spec.direction for n in allsig}
    out = {}
    for lab, start, names in (("1985 onward, 20 signals", "1985-01-01", LIBRARY + EXT_EARNINGS),
                              ("1996 onward, 23 signals", "1996-01-01", allsig)):
        C = diagnostics.mean_rank_correlation(d[d["date"] >= start], names)
        s = np.array([sign[n] for n in names])
        Co = C * np.outer(s, s)
        out[lab] = Co
        Co.round(3).to_csv(paths.REPORTS / "tables" / f"signal_correlation_ext_{start[:4]}.csv")
    eff = {lab: diagnostics.effective_number(C.to_numpy()) for lab, C in out.items()}
    eff16 = diagnostics.effective_number(out["1996 onward, 23 signals"].loc[LIBRARY, LIBRARY].to_numpy())
    C23 = out["1996 onward, 23 signals"]
    lines = ["# Data integrity: signal extension (no performance statistics)", "",
             f"Built {time.strftime('%Y-%m-%d')} in {time.time() - t0:.0f} s from I/B/E/S (unadjusted summary and "
             "actuals) and OptionMetrics (30-day surface at month ends), linked to CRSP through the WRDS link tables. "
             "Written before any return-based statistic of the new signals was computed.", "",
             "## Coverage of the new signals (share of S&P 500 universe rows with a value)", "", md_table(cov, 2), "",
             "## Correlation of the new signals with the sixteen (oriented, average within-month rank correlation, "
             "1996 onward)", "", md_table(C23.loc[new, LIBRARY], 2), "",
             "## Among the new signals", "", md_table(C23.loc[new, new], 2), "",
             "## Effective number of independent signals", "",
             f"- Sixteen original signals, 1996 onward: {eff16['participation_ratio']:.2f}",
             f"- Twenty (with earnings), 1985 onward: {eff['1985 onward, 20 signals']['participation_ratio']:.2f}",
             f"- Twenty-three (with options), 1996 onward: {eff['1996 onward, 23 signals']['participation_ratio']:.2f}",
             "", "## Cross-checks", "",
             f"- OptionMetrics standardized at-the-money 30-day IV vs the surface at delta 50: {check['pairs']:,} "
             f"secid-months, correlation {check['corr_call']:.3f} (calls) / {check['corr_put']:.3f} (puts), median "
             f"absolute difference {check['median_abs_diff_call']:.4f} (calls).",
             f"- I/B/E/S FY1 consensus rows linked to S&P 500 PERMNOs: {len(fy1):,}; quarterly surprises: "
             f"{len(sur):,}; OptionMetrics PERMNO-months: {len(iv):,}.", "",
             "## Specs", "", md_table(pd.DataFrame(meta)[["name", "family", "direction", "published", "records",
                                                          "first"]].set_index("name"), 0)]
    (paths.REPORTS / "data_integrity_extension.md").write_text("\n".join(lines), encoding="utf-8")
    (P / "extension_effective_number.json").write_text(json.dumps({"eff16_1996": eff16, **eff}, indent=1))
    print(f"panel_ext: {len(panel):,} rows; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
