"""Factor exposures of a long-short return series: OLS on factor returns with Newey-West standard errors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphacomb.evaluation.stats import hac_cov, nw_lag


def regress(y: pd.Series, F: pd.DataFrame, periods: int = 12) -> dict:
    """y and F indexed by month; returns annualized alpha, its t-stat, betas with t-stats, R^2 and n."""
    d = pd.concat([y.rename("y"), F], axis=1, join="inner").dropna()
    if len(d) < F.shape[1] + 10:
        return {"n": len(d)}
    Y = d["y"].to_numpy()
    X = np.column_stack([np.ones(len(d)), d[F.columns].to_numpy()])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ b
    T = len(Y)
    XtX_inv = np.linalg.inv(X.T @ X / T)
    S = hac_cov(X * e[:, None], nw_lag(T))
    V = XtX_inv @ S @ XtX_inv / T
    se = np.sqrt(np.diag(V))
    out = {"alpha": float(b[0] * periods), "alpha_t": float(b[0] / se[0]),
           "r2": float(1 - e.var() / Y.var()), "n": int(T)}
    for k, c in enumerate(F.columns, start=1):
        out[f"b_{c}"] = float(b[k])
        out[f"t_{c}"] = float(b[k] / se[k])
    return out
