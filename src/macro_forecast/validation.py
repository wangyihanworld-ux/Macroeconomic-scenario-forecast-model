import pandas as pd
import numpy as np

FEATURES = ["gdp_growth", "inflation", "unemployment", "consumer_confidence", "fx_index"]


class DataValidationError(ValueError):
    pass


def validate_history(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["month", *FEATURES, "revenue_mn"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataValidationError(f"缺少必要字段: {missing}")
    if frame.empty:
        raise DataValidationError("历史数据不能为空")
    out = frame.copy()
    if out.empty:
        raise DataValidationError("数据不能为空")
    out["month"] = pd.to_datetime(out["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    for col in [*FEATURES, "revenue_mn"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[required].isna().any().any():
        raise DataValidationError("日期或数值字段存在空值/无效值")
    if out["month"].duplicated().any():
        raise DataValidationError("月份不能重复")
    if not np.isfinite(out[[*FEATURES, "revenue_mn"]].to_numpy(float)).all():
        raise DataValidationError("数值必须为有限数字")
    ordered = pd.DatetimeIndex(out["month"].sort_values())
    if not ordered.equals(pd.date_range(ordered.min(), ordered.max(), freq="MS")):
        raise DataValidationError("历史月份必须连续，不能缺月")
    if (out["revenue_mn"] <= 0).any():
        raise DataValidationError("收入必须为正数")
    return out.sort_values("month").reset_index(drop=True)


def validate_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["scenario", "month", *FEATURES]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataValidationError(f"情景缺少字段: {missing}")
    out = frame.copy()
    if out.empty:
        raise DataValidationError("数据不能为空")
    out["month"] = pd.to_datetime(out["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    for col in FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[required].isna().any().any():
        raise DataValidationError("情景存在空值/无效值")
    out["scenario"] = out["scenario"].astype("string").str.strip()
    if out["scenario"].eq("").any():
        raise DataValidationError("情景名称不能为空")
    if not np.isfinite(out[FEATURES].to_numpy(float)).all():
        raise DataValidationError("情景数值必须为有限数字")
    if out.duplicated(["scenario", "month"]).any():
        raise DataValidationError("情景与月份组合不能重复")
    return out.sort_values(["scenario", "month"]).reset_index(drop=True)

