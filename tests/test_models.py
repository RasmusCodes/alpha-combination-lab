import numpy as np
import pytest
from sklearn.linear_model import ElasticNet as SkENet
from sklearn.linear_model import Lasso as SkLasso

from alphacomb import models
from alphacomb.models.base import TrainData
from alphacomb.models.linear import enet_moments, ridge_moments


def _moments(rng, n=400, K=6):
    X = rng.standard_normal((n, K))
    X[:, 1] += 0.7 * X[:, 0]
    y = X @ np.array([0.5, -0.2, 0.0, 0.3, 0.0, 0.1]) + rng.standard_normal(n)
    return X, y, X.T @ X, X.T @ y, n


@pytest.mark.parametrize("alpha,lam", [(1.0, 0.05), (0.5, 0.1), (0.2, 0.02)])
def test_enet_moments_matches_sklearn(rng, alpha, lam):
    X, y, G, c, n = _moments(rng)
    b = enet_moments(G, c, n, lam, alpha)
    sk = (SkLasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=100000) if alpha == 1.0 else
          SkENet(alpha=lam, l1_ratio=alpha, fit_intercept=False, tol=1e-12, max_iter=100000)).fit(X, y)
    assert np.allclose(b, sk.coef_, atol=1e-6)


def test_ridge_and_ols_moments(rng):
    X, y, G, c, n = _moments(rng)
    assert np.allclose(ridge_moments(G, c, n, 0.0), np.linalg.lstsq(X, y, rcond=None)[0])
    lam = 0.3
    assert np.allclose(ridge_moments(G, c, n, lam), np.linalg.solve(X.T @ X / n + lam * np.eye(6), X.T @ y / n))


def test_pcr_full_rank_equals_ols(small_linear):
    s, mom = small_linear
    d = TrainData(s.panel, mom, np.arange(20, 120))
    b_ols = models.make("OLS").fit(d).beta
    b_pcr = models.make("PCR", k=s.panel.K).fit(d).beta
    assert np.allclose(b_ols, b_pcr, atol=1e-8)


def test_simple_weights(small_linear):
    s, mom = small_linear
    d = TrainData(s.panel, mom, np.arange(20, 120))
    ew = models.make("EW").fit(d).beta
    assert np.allclose(ew, 1 / s.panel.K)
    hew = models.make("HEW").fit(d).beta
    assert np.isclose(hew.sum(), 1.0)
    fam = s.panel.family_index()
    for idx in fam.values():                         # each family gets the same total weight
        assert np.isclose(hew[idx].sum(), 1 / len(fam))
    icw = models.make("ICW").fit(d).beta
    assert np.isclose(icw.sum(), 1.0) and (icw >= 0).all()


def test_regressions_recover_true_direction():
    """With enough data every regression recovers the true weights' direction (estimation noise ~ 1/sqrt(n))."""
    from alphacomb.data import synthetic
    s = synthetic.make(synthetic.SyntheticConfig(n_assets=300, n_months=220, dgp="linear_unequal", ic=0.2, seed=3))
    mom = models.Moments.from_panel(s.panel)
    d = TrainData(s.panel, mom, np.arange(0, 215))
    truth = s.true_weights[0]
    for name in ("OLS", "RIDGE", "BMA", "BSHRINK", "ENET"):
        b = models.make(name).fit(d).beta
        assert np.corrcoef(b, truth)[0, 1] > 0.85, name


def test_bma_inclusion_and_deff(small_linear):
    s, mom = small_linear
    d = TrainData(s.panel, mom, np.arange(0, 140))
    m = models.make("BMA").fit(d)
    useless = s.true_weights[0] == 0
    assert m.inclusion[~useless].mean() > m.inclusion[useless].mean()
    assert d.deff() >= 1.0
    assert 0 < m.expected_size <= s.panel.K


def test_bshrink_falls_back_to_equal_weights_without_signal(rng):
    from alphacomb.data import synthetic
    s = synthetic.make(synthetic.SyntheticConfig(n_assets=80, n_months=120, ic=0.0, seed=5))
    mom = models.Moments.from_panel(s.panel)
    b = models.make("BSHRINK").fit(TrainData(s.panel, mom, np.arange(0, 110))).beta
    bn = b / np.abs(b).sum()
    assert np.abs(bn - 1 / s.panel.K).max() < 0.05 or np.std(bn) < 0.05


def test_online_methods(small_linear):
    s, mom = small_linear
    months = np.arange(0, 100)
    d = TrainData(s.panel, mom, months)
    rls = models.make("ONLINE_RLS", forget=1.0, ridge=0.0).fit(d)
    ols = models.make("OLS").fit(d)
    assert np.allclose(rls.beta, ols.beta, atol=1e-8)
    hedge = models.make("ONLINE_EG", eta=0.0).fit(d)
    assert np.allclose(hedge.beta, 1 / s.panel.K)
    before = rls.beta.copy()
    rls.update(100, s.panel, mom)
    assert not np.allclose(before, rls.beta)
    dma = models.make("DMA").fit(d)
    w = dma.weights()
    assert np.isfinite(w).all() and np.corrcoef(w, s.true_weights[0])[0, 1] > 0.5
    probs = dma.family_probabilities(s.panel)
    assert set(probs) == set(s.panel.family_index())


def test_trees_fit_and_predict(small_linear):
    s, mom = small_linear
    d = TrainData(s.panel, mom, np.arange(0, 100))
    X = s.panel.X[s.panel.rows(120, 121)]
    for name, kw in (("RF", {"trees": 20, "depth": 3}), ("GBM", {"rounds": 30, "depth": 2})):
        m = models.make(name, n_jobs=1, **kw).fit(d)
        p = m.predict(X)
        assert p.shape == (len(X),) and np.isfinite(p).all()
    g = models.make("GBM", n_jobs=1, rounds=40, depth=2).fit(d)
    st = g.predict_staged(X, [10, 40])
    assert np.allclose(st[40], g.predict(X))


def test_regime_ensemble_uses_context(small_linear):
    s, mom = small_linear
    d = TrainData(s.panel, mom, np.arange(0, 140))
    m = models.make("REGIME", kappa=0.01).fit(d)
    assert set(m.by_regime) <= {"r0", "r1"} and len(m.by_regime) >= 1
    from alphacomb.models.base import MonthContext
    X = s.panel.X[s.panel.rows(141, 142)]
    r = next(iter(m.by_regime))
    p = m.predict(X, MonthContext(141, s.panel.dates[141], {"regime": r}))
    assert np.allclose(p, X @ m.by_regime[r])
