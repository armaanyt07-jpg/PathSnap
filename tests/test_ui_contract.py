from __future__ import annotations

import unittest

from pathsnap.ui_contract import public_prediction_payload


class PublicPredictionPayloadTests(unittest.TestCase):
    def test_only_public_contract_fields_are_kept(self):
        payload = {
            "trip_id": "00000056",
            "decoder": "hmm-viterbi",
            "n_observations": 2,
            "observations": [[1.0, 2.0], [3.0, 4.0]],
            "predicted_route": [4, 5],
            "predicted_line": [[1.0, 2.0], [3.0, 4.0]],
            "metrics": {"top1_accuracy": 0.75, "candidate_miss_rate": 0.1},
            "timings": {"inference_ms": 12.5, "total_ms": 13.0},
            "reference_route": [100, 101],
            "label_arc": [100, 100],
            "reference_line": [[0.0, 0.0]],
        }

        public = public_prediction_payload(payload)
        self.assertNotIn("reference_route", public)
        self.assertNotIn("label_arc", public)
        self.assertNotIn("reference_line", public)
        self.assertEqual(public["predicted_route"], [4, 5])
        self.assertEqual(public["metrics"]["top1_accuracy"], 0.75)

    def test_missing_contract_field_is_rejected(self):
        with self.assertRaises(ValueError):
            public_prediction_payload({"trip_id": "00000056"})


if __name__ == "__main__":
    unittest.main()
