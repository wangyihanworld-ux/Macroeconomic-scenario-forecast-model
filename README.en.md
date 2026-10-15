# Macroeconomic Scenarios and Forecast Evaluation

[中文](README.md) · English

Separate conditional scenario analysis from advance forecasting, and evaluate a transparent public-data case.

## Analytical approach and deliverables

The monthly business demo is synthetic: five macro drivers, trend and seasonal terms explain fictional revenue. It reports coefficients, residuals, VIF and a seasonal-naive benchmark.

Two evaluations are kept separate:

- **Conditional backtest:** uses realized macro drivers in the target month. Its accuracy is not advance-forecast accuracy.
- **Lagged backtest:** uses configurable assumed publication lags (GDP 3 months; inflation/unemployment 2; confidence/FX 1), with a previous-month-end origin and driver source dates. Past revenue is assumed available at that origin. These are assumptions, not verified release calendars.

The independent [Hong Kong public-data case](docs/public-data-case.md) uses World Bank WDI annual GDP growth, CPI inflation and ILO-modelled unemployment, 1991–2024. It predicts GDP growth, not company revenue. The 2009–2024 holdout RMSE is 3.87 percentage points, versus 3.95 for the historical-mean baseline and 5.19 for persistence. The small advantage over the mean should not be overstated.

Raw responses, source URLs, retrieval time, hashes and results are committed. The public case runs offline from the repository root; `--refresh` explicitly replaces the data snapshot and should be reviewed as a data-version change.

## Reproduce

Run from the repository root with Python 3.12:

```powershell
python -m pip install -e .
python -m macro_forecast.demo
python -m macro_forecast.public_case
python -m unittest discover -s tests -v
```

## Interpretation and limitations

All public observations are latest-vintage data, with revisions already incorporated. Lagged evaluation is not a historical real-time vintage backtest. The annual case assumes t-2 indicators and training outcomes are available at the end of t-1. Small samples, shocks, parameter instability and lack of prediction intervals limit inference. Correlation is not causation. Zero-variance revenue, rank-deficient designs and missing monthly periods are rejected with explicit errors.

Business examples use synthetic data and contain no private company or customer records. Only the separately labelled World Bank case in the macro repository uses public economic observations. Refer to current source and CI for this update; older release tags and screenshots describe earlier snapshots.
