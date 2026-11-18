"""Cross-platform structural validation for the train and student-test splits."""
from __future__ import annotations

from pathlib import Path


TRIP_SUFFIXES = (".track", ".nodes", ".arcs", ".route")


def validate_dataset(root: str | Path, expected_student_trips: int = 10) -> dict:
    root = Path(root).expanduser()
    train_dir, test_dir = root / "train", root / "student_test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise ValueError(f"Dataset root must contain train/ and student_test/: {root}")

    train_trips = sorted(p for p in train_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    test_trips = sorted(p for p in test_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not train_trips:
        raise ValueError(f"No training trips found under {train_dir}")
    if len(test_trips) != expected_student_trips:
        raise ValueError(
            f"Expected {expected_student_trips} reserved student_test trips; found {len(test_trips)} under {test_dir}"
        )

    missing = []
    for trip_dir in [*train_trips, *test_trips]:
        for suffix in TRIP_SUFFIXES:
            if not (trip_dir / f"{trip_dir.name}{suffix}").is_file():
                missing.append(f"{trip_dir.name}/{trip_dir.name}{suffix}")
    if missing:
        sample = ", ".join(missing[:8])
        remainder = len(missing) - min(len(missing), 8)
        more = f" (+{remainder} more)" if remainder else ""
        raise ValueError(f"Dataset is missing required trip files: {sample}{more}")

    return {"root": str(root.resolve()), "train_trips": len(train_trips), "student_test_trips": len(test_trips)}
