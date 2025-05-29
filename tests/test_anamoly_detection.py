# tests/test_anomaly_detection.py

import pandas as pd
import numpy as np
import pytest

from src.anomaly_detection import generate_alerts_report

@pytest.fixture
def spike_series():
    # 10 months of data, one big spike in month 6
    dates = pd.date_range("2020-01-01", periods=10, freq="M")
    values = np.ones(10)
    values[5] = 10
    return pd.Series(values, index=dates)

@pytest.fixture
def flat_series():
    dates = pd.date_range("2020-01-01", periods=10, freq="M")
    return pd.Series(np.ones(10), index=dates)

def test_spike_detection(spike_series):
    alerts = generate_alerts_report(
        spike_series,
        window=3,        # small rolling window
        z_thresh=2.0,    # threshold low enough to catch our spike
        model="rbf",
        pen=10
    )
    # Expect exactly one alert at the spike date
    assert len(alerts) == 1
    alert_date = pd.Timestamp("2020-06-30")
    assert alert_date in list(alerts["date"])
    # Depending on your implementation you may label it "anomaly"
    assert alerts.loc[alerts["date"] == alert_date, "alert_type"].iloc[0] in (
        "anomaly", "spike", "change_point"
    )

def test_no_alerts_for_flat_series(flat_series):
    alerts = generate_alerts_report(
        flat_series,
        window=3,
        z_thresh=3.0,
        model="rbf",
        pen=10
    )
    assert alerts.empty
