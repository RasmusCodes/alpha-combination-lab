"""Write demo/fixtures/parity.json: a small synthetic panel plus the Python library's fitted weights and
walk-forward forecasts, so demo/tests.html can check that the browser engine reproduces them."""
from __future__ import annotations

import json

import numpy as np

from alphacomb import models, paths
from alphacomb.data import synthetic
from alphacomb.models.base import Moments, TrainData
from alphacomb.validation import engine

CASES = {"EW": {}, "HEW": {}, "ICW": {}, "OLS": {}, "RIDGE": {"lam": 0.3}, "LASSO": {"frac": 0.1},
         "ENET": {"frac": 0.1, "alpha": 0.5}, "PCR": {"k": 3}, "MVO": {}, "BMA": {}, "BSHRINK": {},
         "ONLINE_EG": {"eta": 0.5}, "ONLINE_RLS": {"forget": 0.98}, "DMA": {}}
WF = {"RIDGE": {"lam": 0.3}, "OLS": {}, "DMA": {}, "ONLINE_RLS": {"forget": 0.98}, "REGIME": {"kappa": 0.1},
      "BSHRINK": {}, "ONLINE_EG": {"eta": 0.5}}


def main() -> None:
    cfg = synthetic.SyntheticConfig(n_assets=60, n_months=100, family_sizes=(3, 2, 1), dgp="linear_unequal",
                                    ic=0.15, seed=5)
    s = synthetic.make(cfg)
    p = s.panel
    mom = Moments.from_panel(p)
    train = np.arange(0, 80)
    d = TrainData(p, mom, train)
    out = {"N": cfg.n_assets, "T": cfg.n_months, "K": p.K, "names": p.names, "families": p.families,
           "X": np.round(p.X, 12).ravel().tolist(), "y": np.round(p.y, 12).tolist(),
           "ret": np.round(p.ret, 12).tolist(),
           "regime": [0 if r == "r0" else 1 for r in p.month_info["regime"]],
           "train": train.tolist(), "deff": d.deff(), "fit": {}, "wf": {}}
    for name, params in CASES.items():
        m = models.make(name, **params).fit(TrainData(p, mom, train))
        out["fit"][name] = {"params": params, "beta": np.asarray(m.weights(), float).tolist()}
    reg = models.make("REGIME", kappa=0.1).fit(TrainData(p, mom, train))
    out["fit"]["REGIME"] = {"params": {"kappa": 0.1}, "byRegime": {
        "0": reg.by_regime["r0"].tolist(), "1": reg.by_regime["r1"].tolist()}, "beta": reg.pool.tolist()}
    wf = engine.WFConfig(first_test=str(p.dates[84].date()), refit_every=12, tune_every=60, embargo=1,
                         min_train=60, inner="none")
    for name, params in WF.items():
        r = engine.run(models.make(name, **params), p, mom, wf, fixed=params)
        rows = slice(int(p.offsets[84]), int(p.offsets[99]))       # months 84..98 (the browser skips the last)
        out["wf"][name] = {"params": params, "pred": np.round(r.pred[rows], 12).tolist()}
    f = paths.DEMO / "fixtures" / "parity.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {f} ({f.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
