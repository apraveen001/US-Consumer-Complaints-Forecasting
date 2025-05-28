# src/feature_engineering.py

import pandas as pd

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with a DatetimeIndex, adds:
      - month
      - day_of_week
      - week_of_year
    """
    df = df.copy()
    df['month']        = df.index.month
    df['day_of_week']  = df.index.dayofweek
    # for pandas ≥1.1, .isocalendar() returns a DataFrame
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    return df

def series_to_supervised(series: pd.Series, lags: int) -> pd.DataFrame:
    """
    Turns a univariate series into a supervised-learning DataFrame
    with columns [lag_0 (y), lag_1, ..., lag_n].
    """
    data = {}
    for i in range(lags + 1):
        data[f'lag_{i}'] = series.shift(i)
    df_sup = pd.DataFrame(data)
    return df_sup.dropna()
