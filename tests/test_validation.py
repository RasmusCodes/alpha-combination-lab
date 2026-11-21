import numpy as np

from alphacomb import models
from alphacomb.data import synthetic
from alphacomb.models.base import Moments
from alphacomb.validation import engine, splits


def test_train_window_purge_and_embargo():
    w = splits.train_window(100, first=0, horizon=1, embargo=1)
    assert w.max() == 98 and w.min() == 0
    w = splits.train_window(100, first=0, horizon=3, embargo=2, window=24)
    assert w.max() == 95 and len(w) == 24


def test_forward_folds_never_look_ahead():
    train = np.arange(0, 200)
    folds = splits.forward_folds(train, n_folds=3, block=36, horizon=1, embargo=1)
    assert len(folds) == 3
    for f in folds:
        assert f.train.max() <= f.val.min() - 2
    assert folds[-1].val.max() == 199


def test_purged_kfold_embargo():
    train = np.arange(0, 100)
    for f in splits.purged_kfold(train, n_folds=5, horizon=1, embargo=12):
        assert len(np.intersect1d(f.train, f.val)) == 0
        after = f.train[f.train > f.val.max()]
        if len(after):
            assert after.min() > f.val.max() + 12
    folds = splits.purged_kfold(train, n_folds=5, horizon=3, embargo=0)
    for f in folds:
        before = f.train[f.train < f.val.min()]
        if len(before):
            assert before.max() <= f.val.min() - 3      # labels of the last training month end before validation


def test_cpcv_counts():
    folds = splits.cpcv(np.arange(120), n_groups=6, k_test=2)
    assert len(folds) == 15


def test_walk_forward_schedule():
    import pandas as pd
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    sched = splits.walk_forward_refits(dates, "2002-01-31", every=12)
    assert [r for r, _ in sched] == [24, 36, 48]
    assert sum(len(m) for _, m in sched) == 36


def _panel():
    s = synthetic.make(synthetic.SyntheticConfig(n_assets=100, n_months=132, dgp="linear_unequal", ic=0.12,
                                                 seed=11))
    return s


def test_engine_runs_and_selects():
    s = _panel()
    mom = Moments.from_panel(s.panel)
    cfg = engine.WFConfig(first_test=str(s.panel.dates[84].date()), min_train=60, inner_block=12, inner_folds=2)
    for name in ("EW", "RIDGE", "ONLINE_EG", "DMA"):
        res = engine.run(models.make(name), s.panel, mom, cfg)
        assert len(res.months) == 132 - 84
        assert np.isfinite(res.pred[s.panel.rows(84, 132)]).all()
        ic = res.ic(s.panel)[res.months]
        assert np.nanmean(ic) > 0
        if name == "RIDGE":
            assert all("lam" in sel["params"] for sel in res.selections)


def test_no_lookahead_perturbing_the_future():
    """Forecasts for months <= T0 must not change when anything dated after T0 changes (signals of later months,
    and outcomes of formation months >= T0, which are only known at T0 + 1 or later)."""
    s = _panel()
    p1 = s.panel
    T0 = 100
    cfg = engine.WFConfig(first_test=str(p1.dates[84].date()), min_train=60, inner_block=12, inner_folds=2)
    p2 = engine_copy = __import__("copy").deepcopy(p1)
    rows_future_x = slice(int(p2.offsets[T0 + 1]), None)
    rows_future_y = slice(int(p2.offsets[T0]), None)
    rng = np.random.default_rng(99)
    p2.X[rows_future_x] = rng.standard_normal(p2.X[rows_future_x].shape)
    p2.ret[rows_future_y] = rng.standard_normal(p2.ret[rows_future_y].shape) * 0.1
    p2.y[rows_future_y] = rng.standard_normal(p2.y[rows_future_y].shape)
    m1, m2 = Moments.from_panel(p1), Moments.from_panel(engine_copy)
    keep = slice(int(p1.offsets[84]), int(p1.offsets[T0 + 1]))
    for name in ("EW", "OLS", "RIDGE", "LASSO", "BMA", "BSHRINK", "MVO", "ICW", "DMA", "ONLINE_EG", "ONLINE_RLS",
                 "REGIME", "PCR"):
        a = engine.run(models.make(name), p1, m1, cfg).pred[keep]
        b = engine.run(models.make(name), p2, m2, cfg).pred[keep]
        assert np.allclose(a, b, equal_nan=True), name
    for name, kw in (("GBM", {"n_jobs": 1}), ("RF", {"n_jobs": 1, "trees": 30})):
        a = engine.run(models.make(name, **kw), p1, m1, cfg, fixed={"depth": 2, "rounds": 25} if name == "GBM"
                       else {"depth": 3}).pred[keep]
        b = engine.run(models.make(name, **kw), p2, m2, cfg, fixed={"depth": 2, "rounds": 25} if name == "GBM"
                       else {"depth": 3}).pred[keep]
        assert np.allclose(a, b), name
