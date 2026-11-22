import numpy as np
import pytest

from alphacomb.evaluation import portfolio, stats


def test_stationary_indices_match_reference_loop():
    def loop(T, block, rng):
        p = 1.0 / block
        idx = np.empty(T, dtype=np.int64)
        idx[0] = rng.integers(T)
        jumps = rng.random(T) < p
        starts = rng.integers(T, size=T)
        for t in range(1, T):
            idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % T
        return idx
    for seed in range(5):
        a = stats.stationary_indices(97, 6, np.random.default_rng(seed))
        b = loop(97, 6, np.random.default_rng(seed))
        assert np.array_equal(a, b)


def test_holm_and_bh_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(stats.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(stats.bh(p), [0.02, 0.04, 0.04, 0.02])


def test_nw_tstat_iid_close_to_classic(rng):
    x = rng.standard_normal(2000) + 0.1
    t_classic = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(stats.nw_tstat(x) - t_classic) / t_classic < 0.1


def test_psr_and_dsr_behave():
    assert stats.psr(0.2, 240, 0, 3, 0) > stats.psr(0.1, 240, 0, 3, 0) > 0.5
    assert stats.expected_max_sr(1, 0.01) == 0
    assert stats.expected_max_sr(100, 0.01) > stats.expected_max_sr(10, 0.01) > 0


def test_expected_max_sr_matches_simulation(rng):
    n, T, reps = 50, 240, 400
    sims = []
    for _ in range(reps):
        R = rng.standard_normal((T, n))
        sr = R.mean(0) / R.std(0, ddof=1)
        sims.append(sr.max())
    v = 1.0 / T
    assert abs(np.mean(sims) - stats.expected_max_sr(n, v)) < 0.02


def test_pbo_noise_vs_real_edge(rng):
    T, N = 480, 30
    noise = rng.standard_normal((T, N)) * 0.02
    out = stats.pbo_cscv(noise, S=10)
    assert 0.25 < out["pbo"] < 0.8
    edge = noise.copy()
    edge[:, 0] += 0.02            # one strategy with Sharpe ~ 3.5 annualized
    assert stats.pbo_cscv(edge, S=10)["pbo"] < 0.05


def test_sharpe_diff_and_spa(rng):
    T = 360
    r1 = rng.standard_normal(T) * 0.03 + 0.004
    r2 = 0.8 * r1 + 0.6 * rng.standard_normal(T) * 0.03
    res = stats.sharpe_diff_test(r1, r2, B=300, seed=1)
    assert np.isfinite(res["t"]) and 0 <= res["p_boot"] <= 1 and res["ci_lo"] < res["ci_hi"]
    D = rng.standard_normal((T, 5)) * 0.01
    spa = stats.spa_test(D, B=300, seed=2)
    assert spa["p_consistent"] > 0.05
    D[:, 0] -= 0.004                  # a clearly poor model must not raise the consistent p-value much
    s0 = stats.spa_test(D, B=300, seed=2)
    assert s0["p_lower"] <= s0["p_consistent"] <= s0["p_upper"]
    D[:, 2] += 0.01
    s1 = stats.spa_test(D, B=300, seed=2)
    assert s1["p_consistent"] < 0.05 and s1["best_index"] == 2


def test_romano_wolf_monotone(rng):
    t_obs = np.array([3.0, 0.5, 2.0])
    tb = rng.standard_normal((2000, 3))
    adj = stats.romano_wolf(t_obs, tb)
    assert adj[0] <= adj[2] <= adj[1]
    assert adj[0] < 0.05


def test_portfolio_weights_turnover_and_drift():
    pred = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    ret = np.zeros(8)
    ids = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    off = np.array([0, 4, 8])
    w = portfolio.rank_weights(pred, off, [0, 1], min_n=2)
    assert np.isclose(w[:4][w[:4] > 0].sum(), 1) and np.isclose(w[:4][w[:4] < 0].sum(), -1)
    turn, tr = portfolio.trades(w, ret, ids, off, [0, 1])
    assert np.isclose(turn[0], 1.0) and np.isclose(turn[1], 0.0)
    ret2 = np.array([0.1, 0.0, 0.0, 0.1, 0, 0, 0, 0])
    turn2, _ = portfolio.trades(w, ret2, ids, off, [0, 1])
    assert turn2[1] > 0                         # drift forces a rebalance
    pred3 = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 5.0])
    ids3 = np.array([1, 2, 3, 4, 1, 2, 3, 9])     # stock 4 leaves, 9 enters
    w3 = portfolio.rank_weights(pred3, off, [0, 1], min_n=2)
    turn3, tr3 = portfolio.trades(w3, ret, ids3, off, [0, 1])
    assert np.isclose(turn3[1], 0.5 * (abs(w3[3]) + abs(w3[7])))
    cost = portfolio.trade_costs(tr3, np.full(8, 0.001), 2)
    assert np.isclose(cost[1], 2 * turn3[1] * 0.001)


def test_registry_redacts_local_paths(tmp_path):
    from pathlib import Path

    from alphacomb import paths
    from alphacomb.registry import Registry, redact
    tb = f'File "{paths.ROOT}\\scripts\\x.py", line 1; File "{Path.home()}\\lib\\y.py"'
    out = redact(tb)
    assert str(paths.ROOT) not in out and str(Path.home()) not in out and "<repo>" in out
    reg = Registry(tmp_path / "r.jsonl")
    rid = reg.start("x", {"a": 1}, kind="development")
    reg.finish(rid, status="failed", notes=tb)
    assert str(Path.home()) not in (tmp_path / "r.jsonl").read_text()


def test_registry_counts_distinct_trials(tmp_path):
    from alphacomb.registry import Registry
    reg = Registry(tmp_path / "r.jsonl")
    a = reg.start("real/main", {"m": "EW"})
    reg.finish(a, results={"sr": 1})
    b = reg.start("real/main", {"m": "EW"})          # same config again: one trial
    reg.finish(b)
    c = reg.start("real/main", {"m": "GBM"})
    reg.finish(c, status="failed")
    d = reg.start("real/robust", {"m": "GBM"}, kind="exploratory")
    reg.finish(d)
    e = reg.start("synthetic/x", {"m": "GBM"}, kind="development")
    reg.finish(e)
    assert reg.trials("real") == 2
    assert reg.trials("real/main") == 1
    assert len(reg.runs()) == 5
    with pytest.raises(ValueError):
        reg.start("x", {}, kind="bogus")
