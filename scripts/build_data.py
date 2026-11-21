"""Build the processed files the signals read (data/processed/, git-ignored: CRSP/Compustat are licensed).

    python scripts/build_data.py [--pull] [--raw DIR] [--stage all|monthly|funda|factors|daily|regimes]

--pull downloads the raw tables from WRDS first (needs an account). Without it the raw files are read from --raw,
ALPHACOMB_CRSP_RAW or data/raw/crsp.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alphacomb import paths
from alphacomb.data import compustat, crsp, daily, french
from alphacomb.evaluation.regimes import market_regimes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    raw = Path(a.raw) if a.raw else paths.RAW_CRSP
    out = paths.PROCESSED
    out.mkdir(parents=True, exist_ok=True)
    diag_path = out / "build_diagnostics.json"
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    t0 = time.time()
    if a.pull:
        crsp.pull(raw)
    st = a.stage
    if st in ("all", "monthly"):
        m = crsp.build_monthly(raw)
        m.to_parquet(out / "monthly.parquet", index=False)
        u = m[m["member"] & m["alive"] & (m["mcap"] > 0)]
        per = u.groupby("month").size()
        diag["monthly"] = {"rows": len(m), "permnos": int(m["permno"].nunique()),
                           "delisting_rows_booked": int(m["delist_booked"].sum()),
                           "delisting_rows_appended": int(m["appended"].sum()),
                           "universe_months": int(len(per)), "universe_min": int(per.min()),
                           "universe_median": float(per.median()), "universe_max": int(per.max()),
                           "first_month": str(m["month"].min().date()), "last_month": str(m["month"].max().date())}
        print("monthly", diag["monthly"])
    if st in ("all", "funda"):
        fl = crsp.link_funda(raw)
        ann = compustat.annual_inputs(fl)
        ann.to_parquet(out / "funda_annual.parquet", index=False)
        diag["funda"] = {"linked_rows": len(fl), "permnos": int(fl["permno"].nunique()),
                         "coverage": ann[["be", "gp", "roa", "acc", "ag", "earn", "cf"]].notna().mean().round(3)
                         .to_dict()}
        print("funda", diag["funda"])
    if st in ("all", "factors", "daily", "regimes"):
        fd = french.daily_factors()
        fm = french.monthly_factors()
        fd.to_parquet(out / "factors_daily.parquet")
        fm.to_parquet(out / "factors_monthly.parquet")
    if st in ("all", "daily"):
        feats = daily.build(raw, fd)
        feats.to_parquet(out / "daily_features.parquet", index=False)
        diag["daily"] = {"rows": len(feats), "coverage": feats[["ivol", "beta", "sigma", "dvol", "high52"]]
                         .notna().mean().round(3).to_dict()}
        print("daily", diag["daily"])
    if st in ("all", "regimes"):
        months = pd.date_range("1962-01-31", "2024-12-31", freq="ME")
        mkt = (fd["Mkt-RF"] + fd["RF"]).dropna()
        mi = market_regimes(mkt, months)
        mi.to_parquet(out / "month_info.parquet")
        diag["regimes"] = mi["regime"].value_counts(dropna=False).astype(int).to_dict()
        print("regimes", diag["regimes"])
    diag["seconds_last_build"] = round(time.time() - t0, 1)
    diag_path.write_text(json.dumps(diag, indent=1, default=str))


if __name__ == "__main__":
    main()
