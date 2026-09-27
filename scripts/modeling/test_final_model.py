"""Checks for county isolation and the endpoint-only funding selection objective."""
import unittest

import numpy as np
import pandas as pd

from data import split_validation
from diagnostics import funding_metrics
from models import fit_funding, interval_padding, regression_logs


class FinalModelTests(unittest.TestCase):
    def test_county_split_keeps_all_years_together(self):
        frame = pd.DataFrame({"county_fips": np.repeat([f"{i:05d}" for i in range(20)], 3),
                              "analysis_year": np.tile([2020, 2021, 2022], 20)})
        fitting, calibration = split_validation(frame, 42, .8)
        self.assertEqual(fitting.county_fips.nunique(), 16)
        self.assertEqual(calibration.county_fips.nunique(), 4)
        self.assertFalse(set(fitting.county_fips) & set(calibration.county_fips))
        self.assertEqual(set(fitting.index) | set(calibration.index), set(frame.index))

    def test_selection_score_uses_only_interval_endpoints(self):
        truth = np.log1p([100, 500, 1000])
        a = np.log1p([[50, 100, 200], [200, 300, 700], [600, 900, 1500]])
        b = a.copy()
        b[:, 1] = b[:, 0]
        self.assertEqual(funding_metrics(truth, a, .8)["mean_interval_score_log1p"],
                         funding_metrics(truth, b, .8)["mean_interval_score_log1p"])
        self.assertNotEqual(funding_metrics(truth, a, .8)["mean_pinball_log1p"],
                            funding_metrics(truth, b, .8)["mean_pinball_log1p"])

    def test_point_candidate_gets_a_real_calibrated_interval(self):
        x_train = pd.DataFrame({"exposure": [1, 2, 3, 4, 5]})
        x_cal = pd.DataFrame({"exposure": range(10)})
        candidate, _ = fit_funding("median_baseline", x_train, np.log1p([100, 200, 300, 400, 500]),
                                   x_cal, np.log1p(np.linspace(10, 2000, 10)), seed=42, jobs=1, coverage=.8)
        prediction = regression_logs(candidate, x_cal)
        self.assertTrue(np.all(prediction[:, 0] < prediction[:, 2]))
        self.assertGreaterEqual(funding_metrics(np.log1p(np.linspace(10, 2000, 10)), prediction, .8)["empirical_interval_coverage"], .8)

    def test_padding_uses_finite_sample_order_statistic(self):
        truth = np.array([1., 2., 3., 4., 5.])
        self.assertEqual(interval_padding(truth, np.zeros((5, 3)), .8), 5.)


if __name__ == "__main__":
    unittest.main()
