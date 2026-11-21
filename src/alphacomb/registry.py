"""Experiment registry: an append-only JSON-lines log of every hypothesis and configuration that was run.

Each run writes a 'started' event before any result exists and a 'completed' (or 'failed') event after, with the
config, its hash, the git commit and dirty flag, a fingerprint of the input data, and a summary of results. Events are
never edited or deleted. The number of distinct configurations tested within an experiment family is what the
Deflated Sharpe Ratio deflates by (trials()), so repeated experimentation is counted, not hidden.

kind: 'confirmatory' (pre-registered), 'exploratory' (decided after seeing results; labelled as such in reports),
      'development' (code checks on synthetic data; not trials on real data).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from alphacomb import paths


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(_jsonable(cfg), sort_keys=True).encode()).hexdigest()[:12]


def git_state(root: Path = paths.ROOT) -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return {"commit": sha or None, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def fingerprint_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


class Registry:
    def __init__(self, path: Path | str = paths.REGISTRY):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")

    def start(self, experiment: str, config: dict, hypotheses=(), kind: str = "confirmatory",
              data_fingerprint: str | None = None, counts_as_trial: bool = True, notes: str = "") -> str:
        if kind not in ("confirmatory", "exploratory", "development"):
            raise ValueError(kind)
        run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self._append({"event": "started", "run_id": run_id, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "experiment": experiment, "hypotheses": list(hypotheses), "kind": kind, "config": config,
                      "config_hash": config_hash(config), "git": git_state(), "data": data_fingerprint,
                      "counts_as_trial": counts_as_trial, "notes": notes})
        return run_id

    def finish(self, run_id: str, status: str = "completed", results: dict | None = None, notes: str = "") -> None:
        self._append({"event": status, "run_id": run_id, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "results": results or {}, "notes": notes})

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def runs(self) -> list[dict]:
        """One record per run: the started event merged with its final event (status 'running' if none)."""
        by = {}
        for e in self.events():
            if e["event"] == "started":
                by[e["run_id"]] = dict(e, status="running")
            elif e["run_id"] in by:
                by[e["run_id"]].update(status=e["event"], results=e.get("results", {}), end=e["time"],
                                       end_notes=e.get("notes", ""))
        return list(by.values())

    def trials(self, experiment_prefix: str = "", kinds=("confirmatory", "exploratory")) -> int:
        """Distinct configurations completed within the experiment family (each config hash counted once)."""
        hashes = {r["config_hash"] for r in self.runs()
                  if r["experiment"].startswith(experiment_prefix) and r["kind"] in kinds
                  and r.get("counts_as_trial", True) and r["status"] == "completed"}
        return len(hashes)
