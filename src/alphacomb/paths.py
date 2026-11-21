"""Repository paths. Licensed data never lives in the repository: raw CRSP/Compustat files are read from
ALPHACOMB_CRSP_RAW (default data/raw/crsp) and everything derived from them is written under data/ (git-ignored)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_CRSP = Path(os.environ.get("ALPHACOMB_CRSP_RAW", DATA / "raw" / "crsp"))
RAW_FRENCH = Path(os.environ.get("ALPHACOMB_FRENCH_RAW", DATA / "raw" / "french"))
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"            # run outputs (git-ignored; aggregated tables go to reports/)
REPORTS = ROOT / "reports"
REGISTRY = ROOT / "registry" / "experiments.jsonl"
CONFIGS = ROOT / "experiments" / "configs"
DEMO = ROOT / "demo"
