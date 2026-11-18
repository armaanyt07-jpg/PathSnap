from __future__ import annotations

import unittest

import numpy as np

from pathsnap import metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_macro_f1_uses_reference_present_arc_classes(self):
        result = metrics.classification_metrics(
            np.array([1, 1, 2]), np.array([1, 3, 3])
        )

        # F1(arc 1) = 2/3, F1(arc 2) = 0. The predicted-only arc 3 is
        # not added as a third class in the macro average.
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["n"], 3)

    def test_unmapped_reference_observations_are_excluded(self):
        result = metrics.classification_metrics(
            np.array([-1, 4]), np.array([9, 4])
        )
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["top1_accuracy"], 1.0)

    def test_empty_labeled_set_returns_nan_metrics(self):
        result = metrics.classification_metrics(
            np.array([-1, -1]), np.array([2, 3])
        )
        self.assertEqual(result["n"], 0)
        self.assertTrue(np.isnan(result["macro_f1"]))

    def test_mismatched_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics.classification_metrics(np.array([1]), np.array([1, 2]))
        with self.assertRaises(ValueError):
            metrics.candidate_prf(np.array([True]), np.array([0.4, 0.8]))


class RouteMetricTests(unittest.TestCase):
    def test_continuity_checks_same_or_connected_arcs(self):
        arcs = np.array([[0, 1], [1, 2], [5, 6]])
        self.assertEqual(metrics.route_continuity(np.array([0, 1, 1, 2]), arcs), 2 / 3)

    def test_invalid_arc_ids_count_as_disconnected(self):
        arcs = np.array([[0, 1], [1, 2]])
        self.assertEqual(metrics.route_continuity(np.array([0, 7, 1]), arcs), 0.0)

    def test_candidate_miss_rate_uses_only_labeled_observations(self):
        ids = np.array([[1, 2], [3, 4], [-1, -1]])
        labels = np.array([2, 9, -1])
        self.assertEqual(metrics.candidate_miss_rate(ids, labels), 0.5)


if __name__ == "__main__":
    unittest.main()
