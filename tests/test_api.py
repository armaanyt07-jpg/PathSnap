import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from api.main import create_app
from api.service import predict_trip
from hmm import Config
from hmm.model import TrainedHMM

CFG = {'k_candidates': 2, 'buffer_m': 300, 'network_prefilter': {'max_segments': 100}}


def fixture():
    return {'trip_id': 'demo', 'track': pd.DataFrame({'lon': [77.0001, 77.0008, 77.0015],
        'lat': [12., 12., 12.], 't': [0., 10., 20.]}),
        'nodes': np.array([[77., 12.], [77.001, 12.], [77.002, 12.], [77., 12.002], [77.002, 12.002]]),
        'arcs': np.array([[0, 1], [1, 2], [3, 4]]), 'route': np.array([0, 1])}


def fake_scores(model, frame):
    # Synthetic plumbing scorer, not a trained classifier or experiment result.
    return np.exp(-frame['perp_dist_m'].to_numpy() / 20)


class IntegrationTests(unittest.TestCase):
    def test_trained_hmm_uses_twenty_features_and_saved_transition_scale(self):
        class EmissionModel:
            def predict_proba(self, rows):
                self.rows = rows
                p = np.exp(-rows[:, 1] / 20)
                return np.column_stack([1 - p, p])

        emitter = EmissionModel()
        model = TrainedHMM(emitter, Config(beta_m=19.5))
        result = predict_trip(fixture(), model, CFG)
        self.assertEqual(emitter.rows.shape[1], 20)
        self.assertTrue(np.isfinite(emitter.rows).all())
        self.assertEqual(result['decoder_config']['beta_m'], 19.5)
        self.assertEqual(result['predicted_route'], [0, 1])

    @patch('api.service.baseline.predict_scores', side_effect=fake_scores)
    def test_reference_does_not_change_predictions(self, mocked):
        trip = fixture()
        a = predict_trip(trip, object(), CFG)
        trip['route'] = np.array([2])
        b = predict_trip(trip, object(), CFG)
        trip['route'] = None
        c = predict_trip(trip, object(), CFG)
        self.assertEqual(a['predictions'], b['predictions'])
        self.assertEqual(a['predictions'], c['predictions'])
        self.assertNotIn('top1_accuracy', c['metrics'])
        self.assertEqual(a['predicted_route'], [0, 1])
        self.assertNotIn('label_arc', json.dumps(a))
        json.dumps(a, allow_nan=False)

    def test_http_contract(self):
        class Stub:
            def trip_ids(self):
                return ['demo']
            def predict(self, trip_id):
                with patch('api.service.baseline.predict_scores', side_effect=fake_scores):
                    return predict_trip(fixture(), object(), CFG)
        with TestClient(create_app(Stub())) as client:
            self.assertTrue(client.get('/health').json()['model_loaded'])
            self.assertEqual(client.get('/trips').json(), {'trips': ['demo']})
            response = client.post('/predict', json={'trip_id': 'demo'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['decoder'], 'hmm-viterbi')
            self.assertEqual(client.post('/predict', json={'trip_id': 'missing'}).status_code, 404)
            self.assertEqual(client.post('/predict', json={'trip_id': '../demo'}).status_code, 422)
            self.assertEqual(client.post('/predict', json={'trip_id': 'demo', 'labels': [1]}).status_code, 422)
            r = client.options('/predict', headers={'Origin': 'http://localhost:8501', 'Access-Control-Request-Method': 'POST'})
            self.assertEqual(r.headers['access-control-allow-origin'], 'http://localhost:8501')

    def test_missing_model(self):
        with patch.dict('os.environ', {'PATHSNAP_MODEL': 'no-such-model.joblib'}):
            with TestClient(create_app()) as client:
                self.assertFalse(client.get('/health').json()['model_loaded'])
                self.assertEqual(client.post('/predict', json={'trip_id': 'demo'}).status_code, 503)
