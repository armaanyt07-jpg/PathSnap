from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pathsnap.comparison import read_decoder_comparison, read_experiments, read_model_comparison


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

    def test_reads_hmm_and_baseline_metrics_from_actual_summary_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student_test_comparison.json"
            path.write_text(
                '[{"trip_id":"00000056","metrics":{"top1_accuracy":0.8},'
                '"baseline_metrics":{"top1_accuracy":0.7},'
                '"timings":{"decode_ms":21.5}}]',
                encoding="utf-8",
            )
            rows = read_decoder_comparison(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["method"], "HMM/Viterbi")
        self.assertEqual(rows[0]["top1_accuracy"], 0.8)
        self.assertEqual(rows[0]["decode_ms"], 21.5)
        self.assertEqual(rows[1]["method"], "Classifier argmax")
        self.assertEqual(rows[1]["top1_accuracy"], 0.7)
        self.assertIsNone(rows[1]["decode_ms"])

    def test_duplicate_trip_runs_remain_chartable_without_pivot_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student_test_comparison.json"
            row = '{"trip_id":"00000056","metrics":{"top1_accuracy":0.8},"baseline_metrics":{"top1_accuracy":0.7}}'
            path.write_text(f"[{row},{row}]", encoding="utf-8")
            rows = read_decoder_comparison(path)
        self.assertEqual([row["run"] for row in rows], ["1", "1", "2", "2"])

    def test_reads_member_b_family_comparison_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_comparison.csv"
            path.write_text(
                "family,cv_top1_mean,holdout_top1,holdout_top1_covered,holdout_macro_f1,"
                "holdout_log_loss,test_top1,test_log_loss,fit_seconds,inference_ms_per_1k\n"
                "hist_gbdt,0.73,0.75,0.86,0.81,0.16,0.84,0.10,4.5,1.17\n",
                encoding="utf-8",
            )
            rows = read_model_comparison(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "hist_gbdt")
        self.assertEqual(rows[0]["cv_top1"], 0.73)
        self.assertEqual(rows[0]["holdout_top1"], 0.75)
        self.assertEqual(rows[0]["test_top1"], 0.84)
        self.assertEqual(rows[0]["candidate_log_loss"], 0.10)
        self.assertEqual(rows[0]["train_seconds"], 4.5)
        self.assertEqual(rows[0]["inference_ms_per_1k"], 1.17)


if __name__ == "__main__":
    unittest.main()
