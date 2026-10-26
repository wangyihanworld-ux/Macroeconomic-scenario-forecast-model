import unittest
import pandas as pd
from macro_forecast.synthetic import build_synthetic_dataset
from macro_forecast.benchmark_evidence import trend_season_backtest

class BenchmarkEvidenceTests(unittest.TestCase):
    def test_future_targets_do_not_change_earlier_predictions(self):
        h=build_synthetic_dataset().history; a=trend_season_backtest(h)
        h.loc[48:,'revenue_mn']*=10; b=trend_season_backtest(h)
        pd.testing.assert_series_equal(a.predicted_revenue_mn.iloc[:13],b.predicted_revenue_mn.iloc[:13])
    def test_macro_inputs_are_unused(self):
        h=build_synthetic_dataset().history; a=trend_season_backtest(h)
        h['consumer_confidence']*=3
        pd.testing.assert_frame_equal(a,trend_season_backtest(h))
