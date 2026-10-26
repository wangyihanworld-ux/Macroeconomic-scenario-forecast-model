"""Predeclared descriptive sensitivity; no test-set model selection."""
import json
from pathlib import Path
import pandas as pd
from .public_case import load_snapshot, evaluate, metrics


def main():
    frame, metadata = load_snapshot('data/public/hong_kong_wdi')
    cases = [(15, 2), (12, 2), (18, 2), (15, 3)]
    backtests = [(window, lag, evaluate(frame, window, lag)) for window, lag in cases]
    common = set.intersection(*(set(part.year) for _, _, part in backtests))
    scores, errors = [], []
    for window, lag, part in backtests:
        part = part[part.year.isin(common)].copy()
        score = metrics(part).assign(initial_years=window, lag_years=lag)
        scores.append(score); errors.append(part.assign(initial_years=window, lag_years=lag))
    out = Path('docs/robustness'); out.mkdir(parents=True, exist_ok=True)
    pd.concat(scores).to_csv(out/'common_year_metrics.csv', index=False)
    pd.concat(errors).to_csv(out/'annual_errors.csv', index=False)
    (out/'method.json').write_text(json.dumps({'cases': cases, 'common_years': sorted(common),
        'release_lags': 'assumed; latest-vintage observations, not real-time vintages',
        'selection': 'No winner selected; base model remains initial_years=15, lag_years=2',
        'data_sources': metadata['sources']}, indent=2), encoding='utf-8')
    print(pd.concat(scores).to_string(index=False))


if __name__ == '__main__':
    main()
