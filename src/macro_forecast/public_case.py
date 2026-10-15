"""Offline-reproducible Hong Kong annual GDP growth case using World Bank WDI.

This is a latest-vintage, lagged evaluation, NOT a real-time vintage backtest.
No synthetic company revenue is mixed with these official economic series.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen

import numpy as np
import pandas as pd

SERIES = {
    'gdp_growth_pct': 'NY.GDP.MKTP.KD.ZG',
    'inflation_pct': 'FP.CPI.TOTL.ZG',
    'unemployment_pct': 'SL.UEM.TOTL.ZS',
}


def download_snapshot(folder):
    """Explicit refresh only. Keep raw responses, retrieval time and checksums."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {'country': 'HKG', 'period': '1991:2024',
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'vintage': 'latest available at retrieval; historical revisions included',
                'release_dates': 'not supplied by this endpoint; lastupdated is dataset metadata, not observation publication time',
                'attribution': 'World Bank, World Development Indicators',
                'terms_url': 'https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets',
                'sources': {}}
    # Fetch every response before replacing the committed snapshot.
    payloads = {}
    for name, code in SERIES.items():
        url = f'https://api.worldbank.org/v2/country/HKG/indicator/{code}?format=json&date=1991:2024&per_page=100'
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        parsed = json.loads(payload)
        if not isinstance(parsed, list) or len(parsed) != 2 or not parsed[1]:
            raise ValueError(f'Unexpected World Bank response: {code}')
        payloads[name] = payload
        metadata['sources'][name] = dict(indicator=code, url=url, unit='percent',
                                        lastupdated=parsed[0].get('lastupdated'),
                                        sha256=hashlib.sha256(payload).hexdigest())
    for name, payload in payloads.items():
        (folder / f'{name}.json').write_bytes(payload)
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def load_snapshot(folder):
    folder = Path(folder)
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    values = {}
    for name, code in SERIES.items():
        payload = (folder / f'{name}.json').read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata['sources'][name]['sha256']:
            raise ValueError(f'Snapshot checksum mismatch: {name}')
        rows = json.loads(payload)[1]
        if any(r['indicator']['id'] != code or r['countryiso3code'] != 'HKG' for r in rows):
            raise ValueError('Unexpected country or indicator')
        series = pd.Series({int(r['date']): r['value'] for r in rows}, name=name, dtype=float)
        if len(series) != len(rows):
            raise ValueError('Duplicate annual observations')
        values[name] = series
    frame = pd.DataFrame(values).sort_index().rename_axis('year').reset_index()
    if frame.year.tolist() != list(range(1991, 2025)):
        raise ValueError('Snapshot must cover every year from 1991 through 2024')
    # Do not interpolate or silently drop unavailable official observations.
    if not np.isfinite(frame[list(SERIES)].to_numpy()).all():
        raise ValueError('Snapshot has missing or nonfinite observations')
    return frame, metadata


def evaluate(frame, initial_years=15, lag_years=2):
    """Predict target t at end of t-1, conservatively using t-2 indicators.

    At origin t-1, training outcomes are limited to t-2 as well. The two-year
    lag is a stated availability assumption, not verified historical releases.
    """
    if not isinstance(lag_years, int) or isinstance(lag_years, bool) or lag_years < 2:
        raise ValueError('Use an integer lag of at least two years')
    if not isinstance(initial_years, int) or isinstance(initial_years, bool) or initial_years < 6:
        raise ValueError('At least six initial training years required')
    data = frame.sort_values('year').reset_index(drop=True)
    years = data.year.to_numpy()
    if not np.array_equal(years, np.arange(years.min(), years.max()+1)):
        raise ValueError('Annual data must be unique and continuous')
    values = data[list(SERIES)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Annual values must be finite')
    rows = []
    for end in range(initial_years+2*lag_years-1, len(data)):
        # Training targets run from lag_years through end-lag_years inclusive.
        target_indices = np.arange(lag_years, end-lag_years+1)
        x = np.column_stack([np.ones(len(target_indices)), values[target_indices-lag_years]])
        if np.linalg.matrix_rank(x) != x.shape[1]:
            raise ValueError('Annual design matrix is rank deficient')
        y = values[target_indices, 0]
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        pred = float(np.r_[1., values[end-lag_years]] @ beta)
        rows.append(dict(year=int(years[end]), forecast_origin=f'{years[end]-1}-12-31',
                         latest_source_year=int(years[end-lag_years]),
                         latest_training_target_year=int(years[target_indices[-1]]),
                         actual_pct=float(values[end, 0]), predicted_pct=pred,
                         persistence_pct=float(values[end-lag_years, 0]),
                         historical_mean_pct=float(y.mean()),
                         error_pp=pred-float(values[end, 0])))
    if not rows:
        raise ValueError('Not enough annual history for a holdout')
    return pd.DataFrame(rows)


def metrics(backtest):
    rows = []
    for label in ('predicted_pct', 'persistence_pct', 'historical_mean_pct'):
        error = backtest[label] - backtest.actual_pct
        rows.append(dict(model=label, n=len(error), mae_pp=float(error.abs().mean()),
                         rmse_pp=float(np.sqrt((error**2).mean())), bias_pp=float(error.mean())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/public/hong_kong_wdi')
    parser.add_argument('--output-dir', default='public_case_output')
    parser.add_argument('--refresh', action='store_true', help='Explicitly download a new data vintage')
    args = parser.parse_args()
    if args.refresh:
        download_snapshot(args.data_dir)
    frame, metadata = load_snapshot(args.data_dir)
    backtest = evaluate(frame)
    scores = metrics(backtest)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/'observations.csv', index=False)
    backtest.to_csv(out/'backtest.csv', index=False)
    scores.to_csv(out/'metrics.csv', index=False)
    (out/'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(scores.to_string(index=False))
    print('Units: percentage points. Latest-vintage lagged evaluation; not real-time forecasting evidence.')


if __name__ == '__main__':
    main()

