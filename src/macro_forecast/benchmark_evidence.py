"""A trend/season baseline requiring no target-month macroeconomic inputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .validation import validate_history,DataValidationError
from .model import benchmark_backtest,rolling_backtest,lagged_backtest,error_metrics
from .synthetic import build_synthetic_dataset

def trend_season_backtest(history,initial_months=36):
    data=validate_history(history)
    if isinstance(initial_months,bool) or not isinstance(initial_months,int) or not 12<=initial_months<len(data):
        raise DataValidationError('Invalid initial window')
    months=data.month.dt.month.to_numpy()
    # Calendar-only features are known at every forecast origin.
    matrix=np.column_stack([np.ones(len(data)),np.arange(len(data)),np.sin(2*np.pi*months/12),np.cos(2*np.pi*months/12)])
    y=data.revenue_mn.to_numpy(float); rows=[]
    for end in range(initial_months,len(data)):
        beta=np.linalg.lstsq(matrix[:end],y[:end],rcond=None)[0]
        prediction=float(matrix[end]@beta); error=y[end]-prediction
        rows.append(dict(month=data.iloc[end].month,actual_revenue_mn=y[end],predicted_revenue_mn=prediction,
                         error_mn=error,absolute_percentage_error=abs(error)/y[end]))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',default='docs/benchmark-results'); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    history=build_synthetic_dataset().history
    methods={'seasonal_naive':benchmark_backtest(history),'trend_season':trend_season_backtest(history),
             'conditional_macro':rolling_backtest(history),'lag_assumed_macro':lagged_backtest(history)}
    common=set.intersection(*(set(frame.month) for frame in methods.values()))
    scores=[]; details=[]
    for name,frame in methods.items():
        part=frame[frame.month.isin(common)].copy(); part['method']=name; details.append(part)
        scores.append(dict(method=name,n=len(part),**error_metrics(part)))
    pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(out/'backtest.csv',index=False)
    print(pd.DataFrame(scores).to_string(index=False))
    print('Synthetic data; conditional_macro has different information availability. No causal or enterprise accuracy claim.')

if __name__=='__main__': main()
