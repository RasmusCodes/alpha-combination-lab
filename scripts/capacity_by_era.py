"""Capacity by era (descriptive). The full-sample capacity table is dominated by the thin markets of the 1980s and
1990s (median S&P 500 stock: about $3M a day traded in the 1980s, about $210M in the 2020s), so this repeats the
square-root-impact calculation over 1980-1999 and 2010-2024 separately.

    python scripts/capacity_by_era.py      -> reports/tables/capacity_by_era.csv
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

ERAS = {"1980-1999": ("1980-01-31", "1999-12-31"), "2010-2024": ("2010-01-31", "2024-11-30")}


def main() -> None:
    cfg, p, _ = rs.real_inputs()
    grid = [float(x) for x in cfg["costs"]["aum_grid"]]
    rows = []
    for m in [m for m in models.ORDER if (paths.RESULTS / "main" / f"{m}.pkl").exists()]:
        r = study.load(paths.RESULTS / "main" / f"{m}.pkl")
        for era, (a, z) in ERAS.items():
            months = r.months[(p.dates[r.months] >= pd.Timestamp(a)) & (p.dates[r.months] <= pd.Timestamp(z))]
            e = A.Evaluated(m, r.pred, p, months)
            cap = A.capacity(e, p, grid, float(cfg["costs"]["impact_y"]))
            row = {"method": m, "era": era, **{f"${x / 1e9:g}B": v["sharpe_net"] for x, v in cap.items()},
                   "breakeven_aum_$B": A.breakeven_aum(cap) / 1e9}
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(paths.REPORTS / "tables" / "capacity_by_era.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
