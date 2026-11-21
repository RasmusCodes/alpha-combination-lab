import numpy as np
import pandas as pd
import pytest

from alphacomb.signals import registry, transforms
from alphacomb.signals.base import ExternalSignal, SignalSpec, validate_records


def _records(n=5):
    return pd.DataFrame({"instrument": np.arange(n), "timestamp": pd.Timestamp("2020-01-31"),
                         "horizon": 1, "value": np.linspace(-1, 1, n), "confidence": 1.0, "n_obs": 12,
                         "provenance": "test", "version": "1.0"})


def test_validate_records_accepts_and_rejects():
    ok = validate_records(_records())
    assert list(ok.columns)[:4] == ["instrument", "timestamp", "horizon", "value"]
    bad = _records()
    bad.loc[0, "timestamp"] = pd.Timestamp("2020-01-15")
    with pytest.raises(ValueError, match="month ends"):
        validate_records(bad)
    dup = pd.concat([_records(), _records()])
    with pytest.raises(ValueError, match="duplicate"):
        validate_records(dup)
    conf = _records()
    conf["confidence"] = 1.5
    with pytest.raises(ValueError, match="confidence"):
        validate_records(conf)
    with pytest.raises(ValueError, match="lack columns"):
        validate_records(_records().drop(columns=["provenance"]))


def test_spec_checks():
    with pytest.raises(ValueError):
        SignalSpec(name="X", family="nonsense", description="")
    with pytest.raises(ValueError):
        SignalSpec(name="X", family="value", description="", direction=2)


def test_register_and_external_signal():
    spec = SignalSpec(name="TEST_EXT_SIGNAL", family="alternative", description="vendor file", direction=1,
                      inputs=("vendor",), published=2024)
    frame = _records()[["instrument", "timestamp", "value"]]
    sig = ExternalSignal(spec=spec, frame=frame)
    registry.add(sig, replace=True)
    rec = registry.get("TEST_EXT_SIGNAL").records(None)
    assert len(rec) == 5 and set(rec["provenance"]) == {"external:frame"}
    assert (rec["horizon"] == 1).all()


def test_register_function_signal_versions_with_code():
    spec = SignalSpec(name="TEST_FN_SIGNAL", family="other", description="", inputs=("x",))

    @registry.register(spec)
    def fn(bundle):
        return _records()[["instrument", "timestamp", "value"]]

    s = registry.get("TEST_FN_SIGNAL")
    assert s.version.startswith("1.0+") and len(s.version) > 5
    assert len(s.records(None)) == 5


def test_transforms_stay_within_month():
    x = np.array([1.0, 2.0, 3.0, np.nan, 10.0, 20.0, 30.0, 40.0])
    off = np.array([0, 4, 8])
    z = transforms.zscore(x, off, min_n=2)
    assert np.isnan(z[3])
    assert abs(np.nanmean(z[:4])) < 1e-12 and abs(z[4:].mean()) < 1e-12
    x2 = x.copy()
    x2[4:] = x2[4:] * 1000            # changing month 2 must not move month 1
    assert np.allclose(transforms.zscore(x2, off, 2)[:3], z[:3])
    g = transforms.rank_gauss(x, off, min_n=2)
    assert np.allclose(g[4:], transforms.rank_gauss(x2, off, 2)[4:])     # ranks are scale free
    w = transforms.winsorize(np.r_[np.arange(99.0), 1e6], np.array([0, 100]), q=0.01)
    assert w.max() < 1e6
    d = transforms.demean_within(np.array([1.0, 3.0, 10.0, 14.0]), np.array([0, 4]), np.array([0, 0, 1, 1]))
    assert np.allclose(d, [-1, 1, -2, 2])


def test_normalization_orients_and_fills():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    off = np.array([0, 6])
    norm = transforms.Normalization(min_n=3)
    v = norm.apply(x, off, direction=-1)
    assert v[2] == 0.0
    assert v[0] > v[5]


def test_library_specs_complete():
    specs = registry.specs()
    lib = [s for s in specs if s.inputs and s.inputs[0].startswith(("crsp", "comp"))]
    assert len(lib) >= 16
    for s in lib:
        assert s.reference and s.published and s.lag and s.description
