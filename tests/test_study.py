"""The frozen configurations must match the code they configure (no drift between pre-registration and runs)."""
import dataclasses

from alphacomb import models, paths, study
from alphacomb.data import synthetic
from alphacomb.signals import registry
from alphacomb.signals.library import LIBRARY
from alphacomb.validation.engine import WFConfig


def test_main_config_consistent():
    cfg = study.load_config("main")
    assert cfg["signals"] == LIBRARY
    assert set(cfg["methods"]) == set(models.METHODS) and cfg["benchmark"] == "EW"
    assert set(cfg["simple_methods"]) <= set(cfg["methods"])
    fields = {f.name for f in dataclasses.fields(WFConfig)}
    assert set(cfg["walk_forward"]) <= fields
    wf = study.wf_config(cfg)
    assert wf.tune_every % wf.refit_every == 0
    for name, params in cfg["method_params"].items():
        models.make(name, **params)                 # every parameter is accepted
    for n in cfg["signals"]:
        s = registry.get(n).spec
        assert s.published is not None and s.direction in (-1, 1)


def test_preregistration_names_every_method_and_signal():
    text = (paths.ROOT / "experiments" / "preregistration.md").read_text(encoding="utf-8")
    cfg = study.load_config("main")
    for m in cfg["methods"]:
        assert m in text, m
    for n in cfg["signals"]:
        assert n in text, n


def test_synthetic_config_consistent():
    c = study.load_config("synthetic")
    assert set(c["dgps"]) <= set(synthetic.DGPS)
    assert set(c["methods"]) == set(models.METHODS)
