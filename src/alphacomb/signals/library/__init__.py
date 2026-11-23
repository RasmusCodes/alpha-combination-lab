"""Built-in signal library: sixteen published US-equity return predictors in seven families, computed point in
time from CRSP and Compustat. Importing this package registers them.

Family        Signals
momentum      MOM (12-1), IMOM (industry), HIGH52 (52-week high)
reversal      STREV (1-month), LTREV (5-year)
value         BM, EP, CFP
quality       GPA, ROA, ACC
investment    AG (asset growth), NSI (net share issuance)
lowrisk       IVOL, BETA
seasonality   SEAS (same calendar month, prior five years)

Directions are the published signs, fixed before any data were seen. `published` is the year of the reference used
for the real-time-availability analysis (a signal may enter a combination only after it was published).
"""
from alphacomb.signals.library import (investment, lowrisk, momentum, quality, reversal,  # noqa: F401
                                       seasonality, value)
from alphacomb.signals.library import earnings, options  # noqa: F401,E402  (extension: I/B/E/S, OptionMetrics)
from alphacomb.signals.library._common import EXT_EARNINGS, EXT_OPTIONS, LIBRARY  # noqa: F401
