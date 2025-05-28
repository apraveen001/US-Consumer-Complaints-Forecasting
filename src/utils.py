# src/utils.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

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
