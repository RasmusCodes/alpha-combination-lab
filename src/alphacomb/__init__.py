"""Alpha Combination and Meta-Modeling Lab.

Combining imperfect, correlated return signals into one forecast, evaluated strictly out of sample.

Layers
  signals/     the standardized signal interface, cross-sectional transforms, the signal library (plugins)
  data/        CRSP/Compustat point-in-time panel, Fama-French factors, synthetic panels with a known truth
  models/      combination methods behind one interface (equal weights ... gradient boosting, DMA, online)
  validation/  walk-forward engine, purged/embargoed splits, nested model selection
  evaluation/  predictive and economic metrics, costs, capacity, statistics (DSR, PBO, SPA, bootstrap)
  registry     the append-only experiment registry every run is recorded in
"""
__version__ = "0.1.0"
