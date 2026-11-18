from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
    from api import main as api
except ImportError as exc:  # The source-only test environment may not have app dependencies installed.
    api = None
    np = None
    pd = None
    API_IMPORT_ERROR = str(exc)
else:
    API_IMPORT_ERROR = ""


@unittest.skipIf(api is None, f"FastAPI runtime dependencies unavailable: {API_IMPORT_ERROR}")
class APIContractTests(unittest.TestCase):
    def test_trips_endpoint_returns_contract_object(self):
        trip_ids = [f"{i:08d}" for i in range(10)]
        with patch.object(api.psio, "trip_ids", return_value=trip_ids):
            self.assertEqual(api.trips(), {"trips": trip_ids})

    def test_prediction_response_is_public_and_contract_shaped(self):
        features = pd.DataFrame(
            {
                "trip_id": ["00000056", "00000056"],
                "obs_idx": [0, 1],
                "arc_id": [1, 2],
                "label_arc": [1, 2],
                "label": [True, True],
            }
        )
        chosen = pd.DataFrame({"arc_id": [1, 2], "label_arc": [1, 2]})
        trip = {
            "track": pd.DataFrame({"lon": [-122.0, -121.9], "lat": [37.0, 37.1]}),
            "nodes": np.array([[-122.0, 37.0], [-121.9, 37.1]]),
            "arcs": np.array([[0, 1], [1, 0]]),
        }
        diag = {"candidate_miss_rate": 0.0}

        with (
            patch.object(api, "get_model", return_value=object()),
            patch.object(api.pipeline, "get_features", return_value=(features, diag)),
            patch.object(api.pipeline, "get_trip", return_value=trip),
            patch.object(api.baseline, "predict_scores", return_value=np.array([0.9, 0.8])),
            patch.object(api.baseline, "choose_one_per_observation", return_value=chosen),
            patch.object(api.pipeline, "line_from_arcs", return_value=np.array([[-122.0, 37.0], [-121.9, 37.1]])),
        ):
            payload = api.predict(api.PredictRequest(trip_id="00000056"))

        self.assertEqual(
            set(payload),
            {"trip_id", "decoder", "n_observations", "observations", "predicted_route", "predicted_line", "metrics", "timings"},
        )
        self.assertEqual(payload["trip_id"], "00000056")
        self.assertEqual(payload["predicted_route"], [1, 2])
        self.assertEqual(payload["n_observations"], 2)
        self.assertTrue({"top1_accuracy", "macro_f1", "candidate_miss_rate", "route_continuity"} <= payload["metrics"].keys())
        self.assertTrue({"features_ms", "inference_ms", "total_ms"} <= payload["timings"].keys())
        self.assertFalse({"label_arc", "reference_route", "reference_line", "route"} & payload.keys())

    def test_prediction_unknown_trip_maps_to_404(self):
        from fastapi import HTTPException

        with (
            patch.object(api, "get_model", return_value=object()),
            patch.object(api.pipeline, "get_features", side_effect=FileNotFoundError),
        ):
            with self.assertRaises(HTTPException) as raised:
                api.predict(api.PredictRequest(trip_id="unknown"))
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
