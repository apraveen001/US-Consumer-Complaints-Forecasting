# src/utils.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_forecasts(
    y_true: np.ndarray,
    y_preds: dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Compare multiple forecast arrays against the true values.
    
    Parameters
    ----------
    y_true : array-like
        The ground truth time-series values.
    y_preds : dict
        A dict mapping model names to their forecast arrays.

    Returns
    -------
    pd.DataFrame
        Indexed by model name, with columns [MAE, MAPE].
    """
    records = []
    for name, pred in y_preds.items():
        mae  = mean_absolute_error(y_true, pred)
        mape = mean_absolute_percentage_error(y_true, pred)
        records.append({'model': name, 'MAE': mae, 'MAPE': mape})
    return pd.DataFrame(records).set_index('model')


def plot_with_alerts(series, alerts):
    """
    series : pd.Series indexed by date
    alerts : DataFrame with columns ['date','alert_type']
    """
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(series.index, series.values, label='Monthly count')
    for _, row in alerts.iterrows():
        ax.axvline(row['date'], linestyle='--', 
                   label=row['alert_type'], alpha=0.7)
    ax.set_title('Time Series with Detected Change-Points / Spikes')
    ax.set_ylabel('Complaint volume')
    ax.legend()
    plt.tight_layout()
    plt.show()
