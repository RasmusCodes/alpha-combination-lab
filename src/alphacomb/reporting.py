"""Small reporting helpers (Markdown tables without extra dependencies)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fmt(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def md_table(df: pd.DataFrame, nd: int | dict = 2, index: bool = True) -> str:
    cols = list(df.columns)
    head = ([df.index.name or ""] if index else []) + [str(c) for c in cols]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] if index else []
        for c in cols:
            d = nd.get(c, 2) if isinstance(nd, dict) else nd
            cells.append(fmt(row[c], d))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
