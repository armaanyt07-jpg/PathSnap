from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pathsnap.data_validation import TRIP_SUFFIXES, validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def test_repository_dataset_has_all_reserved_trips_and_required_files(self):
        root = Path(__file__).resolve().parents[1]
        summary = validate_dataset(root)
        self.assertEqual(summary["student_test_trips"], 10)
        self.assertGreater(summary["train_trips"], 0)

    def test_rejects_missing_trip_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train" / "train_1").mkdir(parents=True)
            (root / "student_test").mkdir()
            for index in range(10):
                trip = root / "student_test" / f"{index:08d}"
                trip.mkdir()
                for suffix in TRIP_SUFFIXES:
                    (trip / f"{trip.name}{suffix}").touch()
            with self.assertRaisesRegex(ValueError, "train_1"):
                validate_dataset(root)

    def test_rejects_wrong_reserved_trip_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train" / "train_1"
            train.mkdir(parents=True)
            (root / "student_test").mkdir()
            for suffix in TRIP_SUFFIXES:
                (train / f"train_1{suffix}").touch()
            with self.assertRaisesRegex(ValueError, "Expected 10"):
                validate_dataset(root)


if __name__ == "__main__":
    unittest.main()
