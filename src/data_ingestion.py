# src/data_ingestion.py
import pandas as pd

def load_complaints(csv_path: str) -> pd.DataFrame:
    """
    Load the cleaned complaints CSV, parse dates, set index, sort.
    """
    df = pd.read_csv(csv_path, parse_dates=['date_received'])
    df = df.set_index('date_received').sort_index()
    return df

def train_test_split_ts(
    series: pd.Series, 
    test_periods: int
) -> tuple[pd.Series, pd.Series]:
    """
    Leave the last `test_periods` observations for testing.
    """
    train = series.iloc[:-test_periods]
    test  = series.iloc[-test_periods:]
    return train, test
