"""Metric tests use synthetic predictions only; no estimator.fit is called."""
import json
import unittest

import numpy as np

from diagnostics import by_impact, funding_metrics, measure_curves, strategy_metrics


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
        self.scores = np.array([[.9, .8, .1, .2], [.1, .8, .7, .2], [.9, .8, .1, .7]])

    def test_f1_uses_top_two_and_per_sample_aggregation(self):
        metrics = strategy_metrics(self.labels, self.scores)
        self.assertAlmostEqual(metrics["precision_at_2"], .5)
        self.assertAlmostEqual(metrics["recall_at_2"], 2 / 3)
        self.assertAlmostEqual(metrics["f1_samples"], 5 / 9)
        self.assertAlmostEqual(metrics["f1_micro"], .6)
        self.assertAlmostEqual(metrics["f1_macro"], (2 / 3 + .5 + 1) / 4)
        self.assertNotAlmostEqual(metrics["f1_samples"], 2 * .5 * (2 / 3) / (.5 + 2 / 3))
        self.assertEqual(metrics["per_measure"]["acquisition"]["confusion_matrix"], [[1, 1], [0, 1]])
        # Perfect class-specific ROC ranking does not imply inclusion in top 2.
        self.assertEqual(metrics["per_measure"]["flood_control"]["roc_auc"], 1)
        self.assertEqual(metrics["per_measure"]["flood_control"]["f1"], 0)

    def test_curves_use_scores_and_serialize_undefined_thresholds(self):
        curves = measure_curves(self.labels, self.scores)
        self.assertIsNone(curves["acquisition"]["roc"][0]["threshold"])
        self.assertIsNone(curves["acquisition"]["pr"][-1]["threshold"])
        self.assertAlmostEqual(curves["acquisition"]["roc_auc"], .75)
        json.dumps(curves, allow_nan=False)

    def test_missing_outcome_class_is_null_not_fake_auc(self):
        labels = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
        scores = np.array([[.9, .2, .1, .0], [.8, .1, .2, .0]])
        metrics = strategy_metrics(labels, scores)
        self.assertIsNone(metrics["macro_roc_auc"])
        self.assertIsNone(metrics["per_measure_average_precision"]["acquisition"])
        self.assertFalse(measure_curves(labels, scores)["drainage"]["defined"])
        json.dumps(metrics, allow_nan=False)

    def test_stable_ties_match_prediction_interface(self):
        labels = np.array([[1, 1, 0, 0]])
        metrics = strategy_metrics(labels, np.full((1, 4), .5))
        self.assertEqual(metrics["f1_samples"], 1)
        self.assertEqual(metrics["per_measure"]["acquisition"]["predicted_positive"], 1)
        self.assertEqual(metrics["per_measure"]["drainage"]["predicted_positive"], 1)

    def test_regression_errors_and_interval_penalty(self):
        truth = np.log1p([100, 200, 1000])
        predictions = np.log1p([[80, 110, 120], [150, 190, 250], [800, 900, 900]])
        metrics = funding_metrics(truth, predictions, .8)
        self.assertAlmostEqual(metrics["mae_nominal_usd"], 40)
        self.assertAlmostEqual(metrics["median_absolute_error_nominal_usd"], 10)
        self.assertAlmostEqual(metrics["rmse_nominal_usd"], np.sqrt(3400))
        self.assertAlmostEqual(metrics["empirical_interval_coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["above_upper_fraction"], 1 / 3)
        self.assertAlmostEqual(metrics["mean_interval_score_nominal_usd"], 1240 / 3)
        json.dumps(metrics, allow_nan=False)

    def test_empty_grade_and_constant_targets_are_not_nan(self):
        truth = np.log1p([100, 100])
        predictions = np.log1p([[80, 100, 120], [80, 100, 120]])
        metrics = funding_metrics(truth, predictions, .8)
        self.assertIsNone(metrics["r2_nominal_usd"])
        groups = by_impact(funding_metrics, truth, predictions, [1, 1], coverage=.8)
        self.assertIsNone(groups["2"]["metrics"])
        self.assertEqual(groups["2"]["rows"], 0)
        json.dumps(groups, allow_nan=False)

    def test_invalid_inputs_fail_before_producing_misleading_metrics(self):
        with self.assertRaises(ValueError):
            strategy_metrics(np.zeros((1, 4)), np.ones((1, 4)))
        with self.assertRaises(ValueError):
            strategy_metrics(self.labels, np.full((3, 4), np.nan))
        with self.assertRaises(ValueError):
            funding_metrics(np.log1p([100]), np.log1p([[120, 100, 80]]), .8)


if __name__ == "__main__":
    unittest.main()
