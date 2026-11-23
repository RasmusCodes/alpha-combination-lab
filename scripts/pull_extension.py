"""Pull the I/B/E/S and OptionMetrics tables for the signal extension (approved 2026-11-22) into data/raw/wrds_ext/.
Resumable: tables already in the manifest are skipped. Needs a WRDS login (libpq password file).

    python scripts/pull_extension.py [--only ibes|optionm]
"""
from __future__ import annotations

import argparse

from alphacomb import paths
from alphacomb.data import ibes, optionm, wrds

OUT = paths.DATA / "raw" / "wrds_ext"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ibes", "optionm"])
    a = ap.parse_args()
    jobs = []
    if a.only in (None, "ibes"):
        jobs += ibes.jobs()
    if a.only in (None, "optionm"):
        jobs += optionm.jobs()
    wrds.pull(jobs, OUT, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
