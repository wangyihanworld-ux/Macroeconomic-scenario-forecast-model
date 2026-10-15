# Hong Kong annual GDP growth: public-data case

This case asks whether a small lagged regression improves on transparent baselines. It is independent of the synthetic monthly company-revenue demo.

## Source and provenance

Source: World Bank, World Development Indicators, economy `HKG` (Hong Kong SAR, China).

| Variable | Indicator | Unit |
| --- | --- | --- |
| Real GDP growth | NY.GDP.MKTP.KD.ZG | Annual percent |
| CPI inflation | FP.CPI.TOTL.ZG | Annual percent |
| Unemployment, modelled ILO estimate | SL.UEM.TOTL.ZS | Percent of labour force |

The complete common period is 1991–2024. The unemployment observation for 1990 was unavailable, so the case starts in 1991; nothing is interpolated. Raw JSON responses, request URLs, UTC retrieval time, WDI last-updated metadata and SHA-256 hashes are in [data/public/hong_kong_wdi](../data/public/hong_kong_wdi/metadata.json). Retrieved on 15 October 2026 in Asia/Shanghai (14 October UTC). WDI dataset metadata reports 8 October 2026 as its update date. That date is not an observation's original publication date.

Attribution: World Bank WDI and its underlying sources, including ILO modelled estimates for unemployment. Review the [World Bank dataset terms](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets); data rights are separate from this repository's code license.

## Forecast timing

At the end of year t-1, predict GDP growth in t using GDP growth, inflation and unemployment in t-2 plus an intercept. Training targets also stop at t-2; their predictors are lagged by two years. The first fit has 15 annual targets. The expanding holdout covers 2009–2024 (16 predictions).

Two years is an explicit conservative availability assumption, not a verified release calendar. All values come from one current WDI snapshot and include revisions. Therefore this is **latest-vintage lagged evaluation**, not a real-time backtest. A genuine real-time study needs archived publication vintages for both indicators and targets.

Persistence uses the latest allowed GDP observation (t-2). The historical-mean baseline averages the same target observations used to train the regression. All models score exactly the same years. GDP growth can be zero or negative, so use MAE/RMSE/bias in percentage points, not MAPE.

## Reproduced results

| Model | MAE (pp) | RMSE (pp) | Bias (pp) |
| --- | ---: | ---: | ---: |
| Lagged regression | 2.52 | 3.87 | +1.01 |
| Persistence | 4.22 | 5.19 | +0.18 |
| Historical mean | 2.74 | 3.95 | +1.86 |

Bias is predicted minus actual. Regression only marginally outperforms the mean on this small holdout. This is not statistical evidence of reliable superiority, causal effects or deployable forecast skill. Large shocks and structural change remain important; no confidence/prediction intervals are supplied. The model and lag were not selected by a reported hyperparameter search, but this is still an exploratory single-sample analysis.

Inspect [year-level errors](public-case-results/backtest.csv), [scores](public-case-results/metrics.csv) and [observations](public-case-results/observations.csv). The raw JSON hashes are verified before use.

```powershell
python -m pip install -e .
python -m macro_forecast.public_case --output-dir public_case_output
python -m unittest discover -s tests -v
```

Default execution is offline. Explicit `--refresh` retrieves a new snapshot; commit its metadata and regenerate results together. Refresh may fail on missing values rather than silently changing the sample. This example supports economic-analysis discussion; it is not a forecast of any private business.
