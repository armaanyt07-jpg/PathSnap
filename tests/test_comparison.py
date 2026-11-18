from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pathsnap.comparison import read_experiments


class ExperimentReaderTests(unittest.TestCase):
    def test_reads_real_numeric_rows_and_leaves_blank_metrics_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiments.csv"
            path.write_text(
                "utc,model,cv_top1,test_top1,inference_ms\n"
                "2026-11-18T10:00:00Z,example,0.7,0.6,12.5\n"
                "2026-11-18T11:00:00Z,second,,0.8,\n",
                encoding="utf-8",
            )

            rows = read_experiments(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["cv_top1"], 0.7)
        self.assertEqual(rows[0]["inference_ms"], 12.5)
        self.assertIsNone(rows[1]["cv_top1"])
        self.assertIsNone(rows[1]["inference_ms"])

    def test_missing_or_unrecognized_file_has_no_comparison_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            invalid = Path(directory) / "invalid.csv"
            invalid.write_text("name,value\nfoo,1\n", encoding="utf-8")
            self.assertEqual(read_experiments(missing), [])
            self.assertEqual(read_experiments(invalid), [])


if __name__ == "__main__":
    unittest.main()
