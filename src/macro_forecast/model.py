from dataclasses import dataclass
import numpy as np
import pandas as pd
from .validation import FEATURES, validate_history, validate_scenarios, DataValidationError


@dataclass(frozen=True)
class ModelResult:
    coefficients: pd.DataFrame
    fitted: pd.DataFrame
    r_squared: float
    adjusted_r_squared: float


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    month_no = frame["month"].dt.month.to_numpy()
    trend = ((frame["month"].dt.year - 2021) * 12 + month_no - 1).to_numpy()
    season_sin = np.sin(2 * np.pi * month_no / 12)
    season_cos = np.cos(2 * np.pi * month_no / 12)
    return np.column_stack([np.ones(len(frame)), frame[FEATURES].to_numpy(float), trend, season_sin, season_cos])


NAMES = ["截距", "GDP增长率", "通胀率", "失业率", "消费者信心", "汇率指数", "时间趋势", "季节正弦", "季节余弦"]


def fit_model(history: pd.DataFrame) -> ModelResult:
    data = validate_history(history)
    if len(data) <= len(NAMES):
        raise DataValidationError("历史期数不足以估计模型")
    x, y = _matrix(data), data["revenue_mn"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    prediction = x @ beta
    residual = y - prediction
    sse = float(residual @ residual)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst
    adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - x.shape[1])
    fitted = data[["month", "revenue_mn"]].copy()
    fitted["fitted_revenue_mn"] = prediction
    fitted["residual_mn"] = residual
    coefficients = pd.DataFrame({"driver": NAMES, "coefficient": beta})
    return ModelResult(coefficients, fitted, r2, adj)


def rolling_backtest(history: pd.DataFrame, initial_months: int = 36) -> pd.DataFrame:
    data = validate_history(history)
    if initial_months <= len(NAMES) or initial_months >= len(data):
        raise DataValidationError("滚动回测窗口不合理")
    rows = []
    for end in range(initial_months, len(data)):
        trained = fit_model(data.iloc[:end])
        beta = trained.coefficients["coefficient"].to_numpy()
        actual = data.iloc[[end]]
        predicted = float((_matrix(actual) @ beta)[0])
        value = float(actual["revenue_mn"].iloc[0])
        rows.append({"month": actual["month"].iloc[0], "actual_revenue_mn": value,
                     "predicted_revenue_mn": predicted, "error_mn": value - predicted,
                     "absolute_percentage_error": abs(value - predicted) / value})
    return pd.DataFrame(rows)


def benchmark_backtest(history: pd.DataFrame, initial_months: int = 36) -> pd.DataFrame:
    """Evaluate a transparent seasonal-naive benchmark using the same holdout months."""
    data = validate_history(history)
    if initial_months < 12 or initial_months >= len(data):
        raise DataValidationError("基准回测窗口不合理")
    rows = []
    for index in range(initial_months, len(data)):
        actual = float(data.iloc[index]["revenue_mn"])
        predicted = float(data.iloc[index - 12]["revenue_mn"])
        error = actual - predicted
        rows.append({"month": data.iloc[index]["month"], "actual_revenue_mn": actual,
                     "predicted_revenue_mn": predicted, "error_mn": error,
                     "absolute_percentage_error": abs(error) / actual})
    return pd.DataFrame(rows)


def error_metrics(backtest: pd.DataFrame) -> dict[str, float]:
    errors = backtest["error_mn"].to_numpy(float)
    return {"mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mape": float(backtest["absolute_percentage_error"].mean())}


def variance_inflation_factors(history: pd.DataFrame) -> pd.DataFrame:
    """Report VIF for economic drivers so multicollinearity is visible to reviewers."""
    data = validate_history(history)
    values = data[FEATURES].to_numpy(float)
    rows = []
    for index, feature in enumerate(FEATURES):
        y = values[:, index]
        other = np.delete(values, index, axis=1)
        x = np.column_stack([np.ones(len(other)), other])
        prediction = x @ np.linalg.lstsq(x, y, rcond=None)[0]
        sst = float(((y - y.mean()) ** 2).sum())
        r_squared = 1 - float(((y - prediction) ** 2).sum()) / sst if sst else 1.0
        vif = float("inf") if r_squared >= 1 - 1e-12 else 1 / (1 - r_squared)
        rows.append({"driver": NAMES[index + 1], "vif": vif})
    return pd.DataFrame(rows)


def forecast_scenarios(history: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    model = fit_model(history)
    future = validate_scenarios(scenarios)
    beta = model.coefficients["coefficient"].to_numpy()
    future["forecast_revenue_mn"] = _matrix(future) @ beta
    order = pd.Categorical(future["scenario"], categories=["压力", "基准", "乐观"], ordered=True)
    future = future.assign(_order=order).sort_values(["_order", "month"]).drop(columns="_order").reset_index(drop=True)
    if (future["forecast_revenue_mn"] <= 0).any():
        raise DataValidationError("情景预测产生非正收入，请检查假设")
    return future


def summarize(history: pd.DataFrame, scenarios: pd.DataFrame) -> dict:
    model = fit_model(history)
    backtest = rolling_backtest(history)
    forecast = forecast_scenarios(history, scenarios)
    totals = forecast.groupby("scenario", as_index=False)["forecast_revenue_mn"].sum()
    totals["scenario"] = pd.Categorical(totals["scenario"], categories=["压力", "基准", "乐观"], ordered=True)
    totals = totals.sort_values("scenario").reset_index(drop=True)
    base = float(totals.loc[totals["scenario"] == "基准", "forecast_revenue_mn"].iloc[0])
    totals["variance_vs_base_mn"] = totals["forecast_revenue_mn"] - base
    benchmark = benchmark_backtest(history)
    metrics = error_metrics(backtest)
    benchmark_metrics = error_metrics(benchmark)
    return {"model": model, "backtest": backtest, "benchmark_backtest": benchmark,
            "forecast": forecast, "scenario_totals": totals, "mape": metrics["mape"],
            "metrics": metrics, "benchmark_metrics": benchmark_metrics,
            "vif": variance_inflation_factors(history)}
