from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticDataset:
    history: pd.DataFrame
    scenarios: pd.DataFrame


def build_synthetic_dataset(seed: int = 42) -> SyntheticDataset:
    """Create reproducible fictional monthly macro and operating data."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2021-01-01", periods=60, freq="MS")
    t = np.arange(60)
    gdp_growth = 0.032 + 0.012 * np.sin(t / 6) + rng.normal(0, 0.004, 60)
    inflation = 0.022 + 0.006 * np.cos(t / 8) + rng.normal(0, 0.0025, 60)
    unemployment = 0.052 - 0.18 * (gdp_growth - 0.032) + rng.normal(0, 0.002, 60)
    consumer_confidence = 100 + 120 * (gdp_growth - 0.032) - 80 * (inflation - 0.022) + rng.normal(0, 1.2, 60)
    fx_index = 100 + np.cumsum(rng.normal(0.05, 0.35, 60))
    season = 7 * np.sin(2 * np.pi * t / 12)
    revenue = (108 + 6.5 * t / 12 + 620 * gdp_growth - 190 * inflation
               - 260 * unemployment + 0.48 * consumer_confidence
               - 0.18 * fx_index + season + rng.normal(0, 2.2, 60))
    history = pd.DataFrame({
        "month": months, "gdp_growth": gdp_growth, "inflation": inflation,
        "unemployment": unemployment, "consumer_confidence": consumer_confidence,
        "fx_index": fx_index, "revenue_mn": revenue,
    })
    future = pd.date_range("2026-01-01", periods=12, freq="MS")
    rows = []
    definitions = {
        "压力": (0.010, 0.038, 0.068, 91.0, 106.0),
        "基准": (0.030, 0.024, 0.052, 101.0, 103.0),
        "乐观": (0.047, 0.017, 0.044, 109.0, 99.0),
    }
    for scenario, values in definitions.items():
        for i, month in enumerate(future):
            gdp, inf, unemp, conf, fx = values
            rows.append({"scenario": scenario, "month": month,
                         "gdp_growth": gdp + 0.002 * np.sin(i / 2),
                         "inflation": inf, "unemployment": unemp,
                         "consumer_confidence": conf + 0.3 * i,
                         "fx_index": fx + 0.15 * i})
    return SyntheticDataset(history, pd.DataFrame(rows))

