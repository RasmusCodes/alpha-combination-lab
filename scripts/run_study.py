"""Run the registered experiments (experiments/preregistration.md). Every run is logged in the registry before it
starts and again when it ends; outputs are checkpointed per job, so an interrupted experiment resumes where it
stopped (finished jobs are skipped unless --force).

    python scripts/run_study.py synthetic   [--workers 3]   E1 positive controls
    python scripts/run_study.py main        [--workers 3]   E2 all methods, frozen protocol
    python scripts/run_study.py robust                      E3 rolling windows, 1-SE rule
    python scripts/run_study.py leakage                     E4 inner-CV schemes
    python scripts/run_study.py realtime                    E5 publication-date signal availability
    python scripts/run_study.py complexity                  E6 every grid configuration, fixed
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from alphacomb import models, paths, study                       # noqa: E402
from alphacomb.evaluation import metrics, portfolio             # noqa: E402
from alphacomb.registry import Registry                         # noqa: E402
from alphacomb.signals import registry as sigreg                # noqa: E402
from alphacomb.validation import engine                         # noqa: E402

CACHE = paths.RESULTS / "cache"


# ----------------------------------------------------------------------------- shared inputs
_INPUTS: dict = {}


def real_inputs(config: str = "main"):
    """A configuration's panel and moments (built once, then loaded from results/cache; once per worker process).
    'main' is the frozen confirmatory study; other configurations (experiments/configs/<name>.yaml) reuse its
    protocol with their own signal set and windows."""
    if config not in _INPUTS:
        cfg = study.load_config(config)
        pk = CACHE / f"panel_{config}.pkl"
        if pk.exists():
            p = study.load(pk)
        else:
            p = study.build_panel(cfg)
            study.save(p, pk)
        mom = study.moments_for(p, CACHE / f"moments_{config}.pkl")
        _INPUTS[config] = (cfg, p, mom)
    return _INPUTS[config]


def summarize(res, p, months=None) -> dict:
    months = res.months if months is None else months
    ic = metrics.monthly_ic(res.pred, p.ret, p.offsets, months=months)
    ev = portfolio.evaluate(res.pred, p, months, p.extras.get("cost"))
    from alphacomb.evaluation.stats import sharpe
    return {"rank_ic": float(np.nanmean(ic[months])), "sharpe_gross": sharpe(ev["gross"][months]),
            "sharpe_net": sharpe(ev["net"][months]), "turnover": float(np.nanmean(ev["turnover"][months])),
            "seconds": round(res.seconds, 1)}


def series(res, p) -> dict:
    """Compact per-month outputs (enough for PBO, complexity curves and the leakage comparison)."""
    m = res.months
    ic = metrics.monthly_ic(res.pred, p.ret, p.offsets, months=m)
    ev = portfolio.evaluate(res.pred, p, m, p.extras.get("cost"))
    return {"months": m, "rank_ic": ic, "gross": ev["gross"], "net": ev["net"], "turnover": ev["turnover"],
            "selections": res.selections, "weights": res.weights, "seconds": res.seconds}


# ----------------------------------------------------------------------------- job bodies (run in workers)
def job_real(spec: dict) -> dict:
    cfg, p, mom = real_inputs(spec.get("config", "main"))
    name = spec["method"]
    proto = models.make(name, **cfg["method_params"].get(name, {}))
    wf = study.wf_config(cfg, **spec.get("wf", {}))
    if wf.importance:
        wf.importance_groups = study.importance_groups(p)
    available = None
    if spec.get("realtime"):
        pub = {n: sigreg.get(n).spec.published for n in p.names}
        available = lambda d: [n for n in p.names if pub[n] is not None and pub[n] <= d.year]  # noqa: E731
    res = engine.run(proto, p, mom, wf, fixed=spec.get("fixed"), available=available)
    out = Path(spec["out"])
    if spec.get("compact"):
        study.save(series(res, p), out)
    else:
        study.save(res, out)
    return summarize(res, p)


def job_gbm_staged(spec: dict) -> dict:
    """E6 for GBM: one walk-forward per depth, predictions at every number of rounds from the same fits."""
    cfg, p, mom = real_inputs()
    depth, rounds = spec["depth"], spec["rounds"]
    store: dict[int, np.ndarray] = {r: np.full(p.n, np.nan) for r in rounds}

    class Staged(models.METHODS["GBM"]):
        def predict(self, X, ctx=None):
            if ctx is None or X.shape[0] != (p.offsets[ctx.t + 1] - p.offsets[ctx.t]):
                return super().predict(X, ctx)
            st = self.predict_staged(X, rounds)
            for r_, v in st.items():
                store[r_][p.rows(ctx.t, ctx.t + 1)] = v
            return st[max(rounds)]

    proto = Staged(**cfg["method_params"]["GBM"])
    wf = study.wf_config(cfg, importance=False)
    res = engine.run(proto, p, mom, wf, fixed={"depth": depth, "rounds": max(rounds)})
    out = {}
    for r_ in rounds:
        res.pred = store[r_]
        study.save(series(res, p), Path(spec["out"].format(rounds=r_)))
        out[r_] = summarize(res, p)
    return out


def job_synthetic(spec: dict) -> dict:
    from alphacomb.data import synthetic
    from alphacomb.models.base import Moments
    c = study.load_config("synthetic")
    s = synthetic.make(synthetic.SyntheticConfig(dgp=spec["dgp"], ic=spec["ic"], seed=spec["seed"],
                                                 **{k: tuple(v) if isinstance(v, list) else v
                                                    for k, v in c["panel"].items()}))
    p = s.panel
    mom = Moments.from_panel(p)
    w = dict(c["walk_forward"])
    w["first_test"] = str(p.dates[w["first_test"]].date())
    wf = engine.WFConfig(**w)
    months = None
    out = {"dgp": spec["dgp"], "ic": spec["ic"], "seed": spec["seed"], "methods": {}}
    for name in c["methods"]:
        res = engine.run(models.make(name, **c["method_params"].get(name, {})), p, mom, wf)
        months = res.months
        ic = metrics.monthly_ic(res.pred, p.ret, p.offsets, months=months)[months]
        ev = portfolio.evaluate(res.pred, p, months, None)
        from alphacomb.evaluation.stats import sharpe
        W = res.weights[months]
        truth = s.true_weights[months]
        wcorr = np.nan
        if np.isfinite(W).all():
            wcorr = float(np.nanmean([np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else np.nan
                                      for a, b in zip(W, truth)]))
        out["methods"][name] = {"rank_ic": ic, "sharpe": sharpe(ev["gross"][months]), "weight_corr": wcorr,
                                "seconds": res.seconds,
                                "params": [sel["params"] for sel in res.selections if sel.get("tuned")]}
    oracle = metrics.monthly_ic(s.oracle, p.ret, p.offsets, months=months)[months]
    out["oracle_rank_ic"] = oracle
    study.save(out, Path(spec["out"]))
    return {"oracle": float(np.nanmean(oracle)),
            **{k: float(np.nanmean(v["rank_ic"])) for k, v in out["methods"].items()}}


# ----------------------------------------------------------------------------- job lists
def jobs_for(exp: str, cfg: dict, methods: list[str] | None, config: str = "main") -> list[dict]:
    ms = methods or cfg.get("methods", [])
    R = paths.RESULTS
    if exp == "main":
        if config != "main":
            return [{"fn": "real", "method": m, "config": config, "out": str(R / config / f"{m}.pkl"),
                     "experiment": f"real/{config}", "kind": "confirmatory", "hyp": cfg.get("hypotheses", [])}
                    for m in ms]
        return [{"fn": "real", "method": m, "out": str(R / "main" / f"{m}.pkl"), "experiment": "real/main",
                 "kind": "confirmatory", "hyp": ["H1", "H2", "H3", "H4", "H7"]} for m in ms]
    if exp == "robust":
        tuned = [m for m in ms if models.METHODS[m].grid]
        return ([{"fn": "real", "method": m, "wf": {"window": 120, "importance": False},
                  "out": str(R / "robust" / "rolling120" / f"{m}.pkl"), "experiment": "real/robust/rolling120",
                  "kind": "confirmatory", "hyp": ["E3"]} for m in ms] +
                [{"fn": "real", "method": m, "wf": {"rule": "1se", "importance": False},
                  "out": str(R / "robust" / "rule1se" / f"{m}.pkl"), "experiment": "real/robust/rule1se",
                  "kind": "confirmatory", "hyp": ["E3"]} for m in tuned])
    if exp == "leakage":
        return [{"fn": "real", "method": m, "wf": {"inner": inner, "importance": False}, "compact": True,
                 "out": str(R / "leakage" / inner / f"{m}.pkl"), "experiment": f"real/leakage/{inner}",
                 "kind": "confirmatory", "hyp": ["H9"]}
                for inner in ("purged_kfold", "naive_kfold") for m in (methods or ["RIDGE", "ENET", "GBM"])]
    if exp == "realtime":
        return [{"fn": "real", "method": m, "realtime": True, "wf": {"importance": False},
                 "out": str(R / "realtime" / f"{m}.pkl"), "experiment": "real/realtime", "kind": "confirmatory",
                 "hyp": ["H8"]} for m in ms]
    if exp == "complexity":
        out = []
        for m in ms:
            cls = models.METHODS[m]
            if not cls.grid or m == "GBM":
                continue
            for i, g in enumerate(cls.grid):
                tag = "_".join(f"{k}{v}" for k, v in g.items())
                out.append({"fn": "real", "method": m, "fixed": g, "wf": {"importance": False}, "compact": True,
                            "out": str(R / "complexity" / f"{m}__{tag}.pkl"), "experiment": "real/complexity",
                            "kind": "confirmatory", "hyp": ["H5", "H6"], "grid_index": i})
        if "GBM" in ms:
            rounds = sorted({g["rounds"] for g in models.METHODS["GBM"].grid})
            for d in sorted({g["depth"] for g in models.METHODS["GBM"].grid}):
                out.append({"fn": "gbm_staged", "method": "GBM", "depth": d, "rounds": rounds,
                            "out": str(R / "complexity" / f"GBM__depth{d}_rounds{{rounds}}.pkl"),
                            "experiment": "real/complexity", "kind": "confirmatory", "hyp": ["H5", "H6"]})
        return out
    if exp == "synthetic":
        c = study.load_config("synthetic")
        return [{"fn": "synthetic", "dgp": d, "ic": ic, "seed": s,
                 "out": str(R / "synthetic" / f"{d}_ic{ic}_seed{s}.pkl"), "experiment": "synthetic/positive_controls",
                 "kind": "development", "hyp": ["E1"]}
                for d in c["dgps"] for ic in c["ics"] for s in c["seeds"]]
    raise SystemExit(f"unknown experiment {exp}")


FNS = {"real": job_real, "gbm_staged": job_gbm_staged, "synthetic": job_synthetic}


def _run(spec):
    try:
        return spec, FNS[spec["fn"]](spec), None
    except Exception:  # noqa: BLE001
        return spec, None, traceback.format_exc()


def done(spec) -> bool:
    if spec["fn"] == "gbm_staged":
        return all(Path(spec["out"].format(rounds=r)).exists() for r in spec["rounds"])
    return Path(spec["out"]).exists()


def run_config(spec: dict, cfg: dict | None, fp: str | None) -> dict:
    keys = ("method", "wf", "fixed", "realtime", "dgp", "ic", "seed", "depth", "rounds")
    c = {k: spec[k] for k in keys if k in spec}
    if cfg is not None and spec["fn"] != "synthetic":
        c["protocol"] = {k: cfg[k] for k in ("panel_start", "panel_end", "signals", "normalization", "walk_forward")}
        if spec.get("method") in cfg["method_params"]:
            c["method_params"] = cfg["method_params"][spec["method"]]
    c["data"] = fp
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--kind", default=None, help="override registry kind (e.g. exploratory)")
    ap.add_argument("--config", default="main", help="configuration for 'main' runs (experiments/configs/<name>.yaml)")
    a = ap.parse_args()
    cfg = fp = None
    if a.experiment != "synthetic":
        cfg, p, _ = real_inputs(a.config)
        fp = study.data_fingerprint(p)
    if a.config != "main" and a.experiment != "main":
        raise SystemExit("--config applies to 'main' runs only")
    jobs = jobs_for(a.experiment, cfg or {}, a.methods, a.config)
    heavy = {"RF": 0, "GBM": 1}                       # longest jobs first so the pool drains evenly
    jobs.sort(key=lambda j: heavy.get(j.get("method"), 2))
    todo = [j for j in jobs if a.force or not done(j)]
    print(f"{a.experiment}: {len(jobs)} jobs, {len(todo)} to run", flush=True)
    reg = Registry()
    ids = {}
    for j in todo:
        ids[id(j)] = reg.start(j["experiment"], run_config(j, cfg, fp), j["hyp"], a.kind or j["kind"],
                               data_fingerprint=fp)
    t0 = time.time()
    broken = False
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_run, j): j for j in todo}
        for f in as_completed(futs):
            j = futs[f]
            try:
                spec, summary, err = f.result()
            except BrokenProcessPool as e:      # a worker was killed (on this shared machine: out of memory)
                summary, err, broken = None, f"worker pool broken, job not finished: {e!r}", True
            if err:
                reg.finish(ids[id(j)], status="failed", notes=err[-2000:])
                print(f"FAILED {j.get('method', j.get('dgp'))}: {err[-600:]}", flush=True)
            else:
                reg.finish(ids[id(j)], results=summary)
                tag = j.get("method") or f"{j['dgp']} ic={j['ic']} s={j['seed']}"
                print(f"done {tag} {j.get('fixed', '')} [{time.time() - t0:.0f}s] {summary}", flush=True)
    print(f"{a.experiment} finished in {time.time() - t0:.0f}s{' (pool broken: rerun to resume)' if broken else ''}",
          flush=True)
    return 3 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
