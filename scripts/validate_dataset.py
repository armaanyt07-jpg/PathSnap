#!/usr/bin/env python3
"""Validate the dataset layout before starting the API."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.config import data_dir
from pathsnap.data_validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    root = Path(args.data_dir).expanduser() if args.data_dir else data_dir()
    try:
        summary = validate_dataset(root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Dataset OK: {summary['root']} | train={summary['train_trips']} trips | "
        f"student_test={summary['student_test_trips']} trips"
    )


if __name__ == "__main__":
    main()
