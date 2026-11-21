"""Compute every library signal as validated records, assemble the modelling panel, and write the data-integrity
report. Uses NO returns beyond attaching the next-month return column: no performance statistic is computed here,
so the pre-registration can be written after this step without having seen any result.

    python scripts/build_signals.py
outputs  data/processed/signals/<NAME>.parquet   records (instrument, timestamp, horizon, value, confidence, n_obs,
                                                  provenance, version)
         data/processed/panel.parquet           one row per formation month x stock: raw signals, ret_next, extras
         reports/data_integrity.md + reports/tables/{signal_coverage,signal_correlation,signal_specs}.csv
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from alphacomb import paths
from alphacomb.data.bundle import CRSPBundle
from alphacomb.reporting import md_table
from alphacomb.signals import diagnostics, registry
from alphacomb.signals.library import LIBRARY


def main() -> None:
    t0 = time.time()
    b = CRSPBundle()
    sig_dir = paths.PROCESSED / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    u = b.universe.rename(columns={"permno": "instrument", "month": "date"}).copy()
    meta = []
    for name in LIBRARY:
        s = registry.get(name)
        rec = s.records(b)
        rec.to_parquet(sig_dir / f"{name}.parquet", index=False)
        v = rec[["instrument", "timestamp", "value"]].rename(columns={"timestamp": "date", "value": name})
        u = u.merge(v, on=["instrument", "date"], how="left")
        meta.append({**s.spec.as_dict(), "version": s.version, "records": len(rec),
                     "first": str(rec["timestamp"].min().date()), "mean_confidence": float(rec["confidence"].mean())})
        print(f"{name}: {len(rec):,} records, version {s.version}", flush=True)
    m = b.monthly[["permno", "month", "tr"]].copy()
    m["month"] = m["month"] - pd.offsets.MonthEnd(1)
    u = u.merge(m.rename(columns={"permno": "instrument", "month": "date", "tr": "ret_next"}),
                on=["instrument", "date"], how="left")
    d = b.daily[["permno", "month", "dvol", "sigma"]].rename(columns={"permno": "instrument", "month": "date"})
    u = u.merge(d, on=["instrument", "date"], how="left")
    u = u.sort_values(["date", "instrument"]).reset_index(drop=True)
    u.to_parquet(paths.PROCESSED / "panel.parquet", index=False)

    tables = paths.REPORTS / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    cov = diagnostics.coverage(u, LIBRARY)
    cov.to_csv(tables / "signal_coverage.csv")
    sample = u[u["date"] >= "1967-01-01"]
    C = diagnostics.mean_rank_correlation(sample, LIBRARY)
    # orient by the literature sign so a positive correlation means "agree on which stocks to buy"
    sign = np.array([registry.get(n).spec.direction for n in LIBRARY])
    Co = C * np.outer(sign, sign)
    Co.round(3).to_csv(tables / "signal_correlation.csv")
    eff = diagnostics.effective_number(Co.to_numpy())
    pd.DataFrame(meta).to_csv(tables / "signal_specs.csv", index=False)
    per = u.groupby("date").size()
    miss_next = float(u.loc[u["date"] < "2024-12-31", "ret_next"].isna().mean())
    lines = ["# Data integrity (no performance statistics)", "",
             f"Built {time.strftime('%Y-%m-%d')} in {time.time() - t0:.0f} s from CRSP/Compustat (S&P 500 point-in-time "
             "members). This report is written before any return-based result is computed.", "",
             "## Universe", "",
             f"- Formation months: {per.index.min().date()} to {per.index.max().date()} ({len(per)} months)",
             f"- Stocks per month: min {per.min()}, median {per.median():.0f}, max {per.max()}",
             f"- Rows: {len(u):,}; next-month return missing for {miss_next:.4%} of rows before the last month "
             "(suspensions / data gaps; treated as zero in portfolios, excluded from ICs)", "",
             "## Coverage by decade (share of universe rows with a value)", "",
             md_table(cov, 2), "",
             "## Average within-month rank correlation (oriented by expected sign), 1967 onward", "",
             md_table(Co, 2), "",
             f"Effective number of independent signals: participation ratio {eff['participation_ratio']:.2f}, "
             f"exchangeable formula {eff['exchangeable']:.2f} (mean oriented correlation {eff['mean_corr']:.3f}); "
             f"first eigenvalue share {eff['first_eigen_share']:.1%}.", "",
             "## Specs", "", md_table(pd.DataFrame(meta)[["name", "family", "direction", "published", "records",
                                                          "first", "mean_confidence"]].set_index("name"),
                                      {"mean_confidence": 3})]
    (paths.REPORTS / "data_integrity.md").write_text("\n".join(lines), encoding="utf-8")
    (paths.PROCESSED / "signal_effective_number.json").write_text(json.dumps(eff, indent=1))
    print(f"panel: {len(u):,} rows; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
