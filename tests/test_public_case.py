import unittest
from pathlib import Path
import numpy as np
from macro_forecast.public_case import load_snapshot, evaluate, metrics


class PublicCaseTest(unittest.TestCase):
    def setUp(self):
        self.frame, self.metadata = load_snapshot(Path(__file__).resolve().parents[1]/'data/public/hong_kong_wdi')

    def test_offline_snapshot_has_complete_official_series(self):
        self.assertEqual(len(self.frame), 34)
        self.assertEqual(self.metadata['country'], 'HKG')
        self.assertTrue(np.isfinite(self.frame.drop(columns='year')).all().all())

    def test_unknown_outcomes_and_drivers_do_not_change_first_prediction(self):
        expected = evaluate(self.frame)
        modified = self.frame.copy()
        first_year = expected.year.iloc[0]
        # At end of t-1 even t-1 annual outcomes are treated as unavailable.
        modified.loc[modified.year >= first_year-1, ['gdp_growth_pct', 'inflation_pct', 'unemployment_pct']] += 100
        actual = evaluate(modified)
        self.assertAlmostEqual(expected.predicted_pct.iloc[0], actual.predicted_pct.iloc[0])
        self.assertTrue((expected.latest_training_target_year <= expected.year-2).all())
        self.assertTrue((expected.latest_source_year <= expected.year-2).all())

    def test_identical_holdout_for_all_models_and_signed_growth(self):
        backtest = evaluate(self.frame)
        scores = metrics(backtest)
        self.assertEqual(scores.n.nunique(), 1)
        self.assertEqual(scores.n.iloc[0], len(backtest))
        self.assertTrue((backtest.actual_pct < 0).any())
        self.assertTrue(np.isfinite(scores[['mae_pp', 'rmse_pp', 'bias_pp']]).all().all())

    def test_gaps_and_short_history_are_rejected(self):
        for frame in (self.frame.drop(index=5), self.frame.head(12)):
            with self.assertRaises(ValueError):
                evaluate(frame)
