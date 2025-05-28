# src/forecasting.py

import sys
import numpy as np
import pandas as pd

from src.anomaly_detection import generate_alerts_report
from src.data_ingestion      import load_complaints, train_test_split_ts
from src.feature_engineering import create_time_features, series_to_supervised
from src.model_training      import train_arima, train_prophet, train_lstm
from src.utils               import evaluate_forecasts


def main(csv_path: str):
    # 1. Load & prepare series
    df = load_complaints(csv_path)
    series = df.resample('M').size()            # Monthly complaint counts
    train, test = train_test_split_ts(series, test_periods=3)

    # 2. ARIMA/SARIMA forecast
    arima_model = train_arima(train.values)
    arima_pred  = arima_model.predict(n_periods=len(test))

    # 3. Prophet forecast (with fallback on error)
    prophet_df = (
        train
        .reset_index()
        .rename(columns={'date_received':'ds', 0:'y'})
    )
    try:
        m_prophet   = train_prophet(prophet_df)
        future      = m_prophet.make_future_dataframe(periods=len(test), freq='M')
        forecast    = m_prophet.predict(future)
        prophet_pred = forecast['yhat'].iloc[-len(test):].values
    except Exception as e:
        print(f"Prophet failed: {e}")
        prophet_pred = np.full(len(test), np.nan)

    # 4. LSTM forecast
    n_lags = 3
    sup = series_to_supervised(train, lags=n_lags)
    lstm_model = train_lstm(sup.values, n_lags)
    last_obs = train.values[-n_lags:]
    lstm_preds = []
    for _ in range(len(test)):
        X = last_obs.reshape(1, n_lags, 1)
        yhat = lstm_model.predict(X, verbose=0)[0, 0]
        lstm_preds.append(yhat)
        last_obs = np.roll(last_obs, -1)
        last_obs[-1] = yhat
    lstm_pred = np.array(lstm_preds)

    # 5. Evaluate - filter out NaN-containing predictions
    y_true = test.values
    y_preds = {
        'ARIMA':   np.array(arima_pred),
        'Prophet': prophet_pred,
        'LSTM':    lstm_pred
    }
    # Exclude any model with NaN predictions
    valid_preds = {name: pred for name, pred in y_preds.items() if not np.isnan(pred).any()}
    skipped = [name for name in y_preds if name not in valid_preds]
    if skipped:
        print(f"Skipping models due to NaN predictions: {skipped}")

    results = evaluate_forecasts(y_true, valid_preds)
    print("\nModel comparison:")
    print(results)

    return results


# after you have your `series` (monthly counts) and forecasts:
alerts = generate_alerts_report(series, window=12, z_thresh=3, model="rbf", pen=10)
alerts.to_csv("reports/alerts_report.csv", index=False)


if __name__ == "__main__":
    # Accept CSV path as first argument, or default path
    path = sys.argv[1] if len(sys.argv) > 1 else "C://Users//ssbap//US-Consumer-Complaints-Forecasting//data//processed//cleaned_consumer_complaints.csv"
    main(path)




# if __name__ == "__main__":
#     path = sys.argv[1] if len(sys.argv) > 1 else \
#         "C://Users//ssbap//US-Consumer-Complaints-Forecasting//data//processed//cleaned_consumer_complaints.csv"
#     main(path)
