from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from api.main import create_app
except ImportError as exc:
    TestClient = None
    create_app = None
    API_IMPORT_ERROR = str(exc)
else:
    API_IMPORT_ERROR = ""


@unittest.skipIf(create_app is None, f"FastAPI runtime dependencies unavailable: {API_IMPORT_ERROR}")
class APIContractTests(unittest.TestCase):
    def setUp(self):
        class StubService:
            def trip_ids(self):
                return [
                    "00000000", "00000007", "00000009", "00000040", "00000052",
                    "00000056", "00000065", "00000076", "00000077", "00000091",
                ]

            def predict(self, trip_id):
                return {
                    "trip_id": trip_id,
                    "decoder": "hmm-viterbi",
                    "n_observations": 2,
                    "observations": [[1.0, 2.0], [3.0, 4.0]],
                    "predicted_route": [4, 5],
                    "predicted_line": [[1.0, 2.0], [3.0, 4.0]],
                    "predicted_lines": [[[1.0, 2.0], [3.0, 4.0]]],
                    "baseline_metrics": {"top1_accuracy": 0.6},
                    "metrics": {"top1_accuracy": 0.75, "route_continuity": 1.0},
                    "timings": {"features_ms": 2.0, "decode_ms": 8.5, "total_ms": 11.0},
                }

        self.client = TestClient(create_app(StubService()))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_and_all_ten_trip_ids(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["model_loaded"])
        trips = self.client.get("/trips").json()["trips"]
        self.assertEqual(len(trips), 10)

    def test_prediction_matches_contract_without_reference_fields(self):
        response = self.client.post("/predict", json={"trip_id": "00000056"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        required = {
            "trip_id", "decoder", "n_observations", "observations", "predicted_route",
            "predicted_line", "metrics", "timings",
        }
        self.assertTrue(required <= payload.keys())
        self.assertEqual(payload["decoder"], "hmm-viterbi")
        self.assertIn("predicted_lines", payload)
        self.assertIn("baseline_metrics", payload)
        self.assertIn("decode_ms", payload["timings"])
        self.assertFalse({"label_arc", "reference_route", "reference_line", "route"} & payload.keys())

    def test_invalid_and_unknown_requests_have_contract_errors(self):
        self.assertEqual(self.client.post("/predict", json={"trip_id": "unknown"}).status_code, 404)
        self.assertEqual(self.client.post("/predict", json={"trip_id": "../secret"}).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={"trip_id": "00000056", "labels": [1]}).status_code, 422)

    def test_missing_model_is_reported_as_not_ready(self):
        app = create_app()
        with tempfile.TemporaryDirectory() as directory:
            missing_model = Path(directory) / "missing.joblib"
            with patch.dict(os.environ, {"PATHSNAP_MODEL": str(missing_model)}):
                with TestClient(app) as client:
                    health = client.get("/health").json()
                    self.assertEqual(health["status"], "not_ready")
                    self.assertFalse(health["model_loaded"])
                    self.assertEqual(client.post("/predict", json={"trip_id": "00000056"}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
