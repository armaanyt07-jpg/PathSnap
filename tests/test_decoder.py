import copy
import json
import unittest
from pathlib import Path
from hmm import decode, Config
from hmm.decoder import Network


ROOT = Path(__file__).resolve().parents[1]


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'examples/prepared/demo.json').read_text())

    def run_decode(self):
        return decode(self.data['observations'], self.data['arcs'])

    def test_sequence_overrules_greedy_and_adds_bridge(self):
        result = self.run_decode()
        self.assertEqual([o['arc_id'] for o in result['observations']], [10, 10, 12])
        self.assertEqual(result['predicted_route'], [10, 11, 12])
        self.assertEqual(result['baseline_route'], [10, 99, 12])

    def test_missing_candidates_preserves_observation(self):
        self.data['observations'][1]['candidates'] = []
        result = self.run_decode()
        self.assertIsNone(result['observations'][1]['arc_id'])
        self.assertEqual(result['predicted_route'], [10, None, 12])

    def test_disconnection_restarts(self):
        self.data['observations'][1]['candidates'] = [self.data['observations'][1]['candidates'][1]]
        result = self.run_decode()
        self.assertEqual(result['breaks'][-1]['reason'], 'disconnected_or_speed_limit')
        self.assertEqual(result['route_segments'][-1], [12])

    def test_gap(self):
        self.data['observations'][2]['timestamp'] = 1000
        self.assertEqual(self.run_decode()['breaks'][0]['reason'], 'large_time_gap')

    def test_invalid_probabilities(self):
        for p in [-1, float('nan'), float('inf'), 1.1]:
            self.data['observations'][0]['candidates'][0]['probability'] = p
            with self.assertRaises(ValueError):
                self.run_decode()

    def test_time_order(self):
        self.data['observations'][1]['timestamp'] = 0
        with self.assertRaises(ValueError):
            self.run_decode()

    def test_empty(self):
        self.assertEqual(decode([], [])['predicted_route'], [])

    def test_zero_emissions(self):
        for o in self.data['observations']:
            for c in o['candidates']:
                c['probability'] = 0
        for o in self.run_decode()['observations']:
            self.assertAlmostEqual(sum(o['probabilities']), 1)

    def test_directed_same_arc_backwards_requires_loop(self):
        net = Network(self.data['arcs'])
        distance, _ = net.travel({'arc_id': 10, 'along_arc_m': 80}, {'arc_id': 10, 'along_arc_m': 20}, 500)
        self.assertEqual(distance, float('inf'))

    def test_small_backwards_projection_can_be_treated_as_gps_jitter(self):
        net = Network(self.data['arcs'])
        left = {'arc_id': 10, 'along_arc_m': 80}
        right = {'arc_id': 10, 'along_arc_m': 77}
        self.assertEqual(net.travel(left, right, 500, backtrack_tolerance_m=4), (0.0, [10]))
        self.assertEqual(net.travel(left, right, 500, backtrack_tolerance_m=2)[0], float('inf'))

    def test_optional_restart_marks_disconnected_segments(self):
        self.data['observations'][1]['candidates'] = [self.data['observations'][1]['candidates'][1]]
        result = decode(self.data['observations'], self.data['arcs'], Config(restart_penalty=2))
        self.assertEqual(result['route_segments'], [[10], [99], [12]])
        self.assertEqual([b['reason'] for b in result['breaks']], ['low_transition_confidence'] * 2)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            Config(beta_m=0)


