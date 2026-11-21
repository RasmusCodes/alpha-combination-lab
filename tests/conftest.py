import numpy as np
import pytest

from alphacomb.data import synthetic
from alphacomb.models.base import Moments


@pytest.fixture(scope="session")
def small_linear():
    cfg = synthetic.SyntheticConfig(n_assets=120, n_months=150, dgp="linear_unequal", ic=0.15, seed=1)
    s = synthetic.make(cfg)
    return s, Moments.from_panel(s.panel)


@pytest.fixture(scope="session")
def small_equal():
    cfg = synthetic.SyntheticConfig(n_assets=120, n_months=150, dgp="linear_equal", ic=0.10, seed=2,
                                    missing_rate=0.05)
    s = synthetic.make(cfg)
    return s, Moments.from_panel(s.panel)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
