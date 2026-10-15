import unittest
import numpy as np
import pandas as pd
from macro_forecast.synthetic import build_synthetic_dataset
from macro_forecast.model import fit_model, benchmark_backtest, lagged_backtest, forecast_scenarios
from macro_forecast.validation import DataValidationError


class BoundaryTest(unittest.TestCase):
    def setUp(self):
        self.data = build_synthetic_dataset()

    def test_constant_revenue_and_rank_deficiency_are_explained(self):
        for col in ('revenue_mn', 'inflation'):
            h = self.data.history.copy()
            h[col] = 100.
            with self.assertRaises(DataValidationError):
                fit_model(h)

    def test_calendar_gaps_and_same_month_duplicates_are_rejected(self):
        h = self.data.history.copy()
        h.loc[1, 'month'] = h.loc[0, 'month'] + pd.Timedelta(days=7)
        for bad in (h, self.data.history.drop(index=30)):
            with self.assertRaises(DataValidationError):
                benchmark_backtest(bad)

    def test_nonfinite_input_is_rejected(self):
        h = self.data.history.copy()
        h.loc[0, 'inflation'] = np.inf
        with self.assertRaises(DataValidationError):
            fit_model(h)

    def test_scenarios_must_follow_history_on_common_horizon(self):
        for frame in (self.data.scenarios.iloc[:-1], self.data.scenarios.assign(month=lambda x: x.month-pd.DateOffset(years=1))):
            with self.assertRaises(DataValidationError):
                forecast_scenarios(self.data.history, frame)

    def test_lagged_forecast_does_not_see_target_or_future_values(self):
        original = lagged_backtest(self.data.history)
        changed = self.data.history.copy()
        changed.loc[36:, ['gdp_growth', 'inflation', 'unemployment', 'consumer_confidence', 'fx_index']] *= 2
        changed.loc[36:, 'revenue_mn'] *= 3
        again = lagged_backtest(changed)
        self.assertAlmostEqual(original.predicted_revenue_mn.iloc[0], again.predicted_revenue_mn.iloc[0])
        self.assertNotEqual(original.actual_revenue_mn.iloc[0], again.actual_revenue_mn.iloc[0])
        for col in original.filter(like='_source_month'):
            self.assertTrue((original[col] < original.forecast_origin).all())

    def test_zero_lag_cannot_be_presented_as_advance_forecast(self):
        from macro_forecast.model import DEFAULT_RELEASE_LAGS
        with self.assertRaises(DataValidationError):
            lagged_backtest(self.data.history, release_lags={**DEFAULT_RELEASE_LAGS, 'gdp_growth': 0})

    def test_benchmark_matches_calendar_year(self):
        f = benchmark_backtest(self.data.history)
        history = self.data.history.set_index('month')
        for row in f.itertuples():
            self.assertEqual(row.predicted_revenue_mn, history.loc[row.month-pd.DateOffset(years=1), 'revenue_mn'])
