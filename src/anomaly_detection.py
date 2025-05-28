# src/anomaly_detection.py

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import zscore
from typing import List, Union

def detect_spikes(
    series: pd.Series,
    window: int = 12,
    z_thresh: float = 3.0
) -> pd.DatetimeIndex:
    """
    Identify timestamps where the rolling z-score exceeds threshold.
    
    Args:
        series: time-indexed series of counts (e.g. monthly complaint counts).
        window: rolling window size (in number of periods) for mean/std.
        z_thresh: absolute z-score threshold to flag a spike.
    
    Returns:
        DatetimeIndex of spike dates.
    """
    # compute rolling mean & std
    rol_mean = series.rolling(window, center=True, min_periods=1).mean()
    rol_std  = series.rolling(window, center=True, min_periods=1).std(ddof=0)
    z_scores = (series - rol_mean) / rol_std
    spikes   = z_scores.abs() > z_thresh
    return series.index[spikes]

def detect_change_points(
    series: pd.Series,
    model: str = "rbf",
    pen: Union[float, int] = 10
) -> List[pd.Timestamp]:
    """
    Detect change-points in the series using the PELT algorithm.
    
    Args:
        series: time-indexed series of counts.
        model: cost function model for ruptures (e.g. "l1", "l2", "rbf").
        pen: penalty value controlling sensitivity (higher → fewer breaks).
    
    Returns:
        List of pd.Timestamp where a change-point is detected.
    """
    # prepare data for ruptures (needs 2D array)
    arr = series.values.reshape(-1, 1)
    algo = rpt.Pelt(model=model).fit(arr)
    # breakpoints includes the end of the series; drop it
    bkps = algo.predict(pen=pen)
    # map break indices to timestamps, ignore last index (== len)
    change_idxs = [i for i in bkps if i < len(series)]
    return [series.index[i] for i in change_idxs]

def generate_alerts_report(
    series: pd.Series,
    window: int = 12,
    z_thresh: float = 3.0,
    model: str = "rbf",
    pen: Union[float, int] = 10
) -> pd.DataFrame:
    """
    Build a report of spike and change-point dates.
    
    Args:
        series: time-indexed series of counts.
        window, z_thresh: passed to detect_spikes.
        model, pen: passed to detect_change_points.
    
    Returns:
        DataFrame with columns ['date','alert_type'].
    """
    spikes = detect_spikes(series, window=window, z_thresh=z_thresh)
    cps    = detect_change_points(series, model=model, pen=pen)
    
    records = []
    for dt in spikes:
        records.append({"date": dt, "alert_type": "spike"})
    for dt in cps:
        records.append({"date": dt, "alert_type": "change_point"})
    
    report = pd.DataFrame(records)
    # ensure chronological order and drop duplicates if overlap
    report = (
        report
        .drop_duplicates(["date","alert_type"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return report
