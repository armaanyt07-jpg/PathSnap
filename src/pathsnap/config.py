"""Configuration and path resolution.

Data root precedence:
  1. $PATHSNAP_DATA_DIR
  2. <repo>/data            (if it contains train/)
  3. <repo>/../materials/student/Student folder   (local convenience)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "base.json"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as fh:
        return json.load(fh)


def data_dir() -> Path:
    env = os.environ.get("PATHSNAP_DATA_DIR")
    if env:
        return Path(env).expanduser()
    for candidate in (ROOT, ROOT / "data", ROOT.parent / "materials" / "student" / "Student folder"):
        if (candidate / "train").exists():
            return candidate
    return ROOT / "data"


def results_dir() -> Path:
    p = ROOT / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def artifacts_dir() -> Path:
    p = ROOT / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir() -> Path:
    p = ROOT / "data" / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p
