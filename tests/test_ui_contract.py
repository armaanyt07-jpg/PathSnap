from __future__ import annotations

import unittest

from pathsnap.ui_contract import public_prediction_payload, validate_health_payload


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
            "baseline_metrics": {"top1_accuracy": 0.6, "reference_route": [100, 101]},
            "predicted_lines": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            "breaks": [{"obs_idx": 4, "reason": "large_time_gap", "reference_line": [[0, 0]]}],
            "decoder_config": {"beta_m": 50, "label_arc": [100, 101]},
            "timings": {"model_ms": 4.0, "decode_ms": 8.5, "inference_ms": 12.5, "total_ms": 13.0},
            "reference_route": [100, 101],
            "label_arc": [100, 100],
            "reference_line": [[0.0, 0.0]],
            "predictions": [{"label_arc": 100}],
            "metric_definitions": {"macro_f1": "wrong definition"},
        }

        public = public_prediction_payload(payload)
        self.assertNotIn("reference_route", public)
        self.assertNotIn("label_arc", public)
        self.assertNotIn("reference_line", public)
        self.assertNotIn("predictions", public)
        self.assertNotIn("metric_definitions", public)
        self.assertEqual(public["predicted_route"], [4, 5])
        self.assertEqual(public["metrics"]["top1_accuracy"], 0.75)
        self.assertEqual(public["baseline_metrics"]["top1_accuracy"], 0.6)
        self.assertNotIn("reference_route", public["baseline_metrics"])
        self.assertEqual(len(public["predicted_lines"]), 2)
        self.assertEqual(public["breaks"], [{"obs_idx": 4, "reason": "large_time_gap"}])
        self.assertEqual(public["decoder_config"], {"beta_m": 50})
        self.assertEqual(public["timings"]["decode_ms"], 8.5)

    def test_missing_contract_field_is_rejected(self):
        with self.assertRaises(ValueError):
            public_prediction_payload({"trip_id": "00000056"})

    def test_unready_health_means_connected_without_model(self):
        self.assertEqual(
            validate_health_payload({"status": "not_ready", "model_loaded": False}),
            {"status": "not_ready", "model_loaded": False},
        )
        with self.assertRaises(ValueError):
            validate_health_payload({"status": "error", "model_loaded": False})

    def test_ready_health_preserves_member_b_model_choices(self):
        payload = {
            "status": "ok",
            "model_loaded": True,
            "models": ["champion", "hist_gbdt"],
            "default_model": "champion",
            "reference_route": [1, 2],
        }
        self.assertEqual(
            validate_health_payload(payload),
            {
                "status": "ok",
                "model_loaded": True,
                "models": ["champion", "hist_gbdt"],
                "default_model": "champion",
            },
        )

    def test_rejects_nonfinite_or_out_of_range_map_coordinates(self):
        payload = {
            "trip_id": "00000056", "decoder": "hmm-viterbi", "n_observations": 1,
            "observations": [[float("nan"), 2.0]], "predicted_route": [],
            "predicted_line": [], "metrics": {}, "timings": {},
        }
        with self.assertRaisesRegex(ValueError, "WGS84"):
            public_prediction_payload(payload)

    def test_rejects_invalid_coordinates_in_segmented_routes(self):
        payload = {
            "trip_id": "00000056", "decoder": "hmm-viterbi", "n_observations": 0,
            "observations": [], "predicted_route": [], "predicted_line": [],
            "predicted_lines": [[[181, 0], [1, 1]]], "metrics": {}, "timings": {},
        }
        with self.assertRaisesRegex(ValueError, "WGS84"):
            public_prediction_payload(payload)


if __name__ == "__main__":
    unittest.main()
