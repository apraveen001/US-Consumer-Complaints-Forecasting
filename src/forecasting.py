import pandas as pd
from data_ingestion import load_complaints, train_test_split_ts
from feature_engineering import create_time_features, series_to_supervised
from model_training import train_arima, train_prophet, train_lstm
from utils import evaluate_forecasts

def main(csv_path: str):
    # 1. Load & split
    df = load_complaints(csv_path)
    series = df.resample('M').size()
    train, test = train_test_split_ts(series, test_periods=3)
    
    # 2. ARIMA Forecast
    arima_model = train_arima(train.values)
    arima_pred  = arima_model.predict(n_periods=len(test))
    
    # 3. Prophet Forecast
    prophet_df = train.reset_index().rename(columns={'date_received':'ds', 0:'y'})
    prophet_model = train_prophet(prophet_df)
    future = prophet_model.make_future_dataframe(periods=len(test), freq='M')
    forecast = prophet_model.predict(future)
    prophet_pred = forecast.iloc[-len(test):]['yhat'].values
    
    # 4. LSTM Forecast
    sup = series_to_supervised(train, lags=3)
    lstm_model = train_lstm(sup.values, n_lags=3)
    # rolling window prediction
    last_obs = train.values[-3:]
    lstm_preds = []
    for _ in range(len(test)):
        X = last_obs.reshape(1, 3, 1)
        yhat = lstm_model.predict(X, verbose=0)[0,0]
        lstm_preds.append(yhat)
        last_obs = np.roll(last_obs, -1)
        last_obs[-1] = yhat
    
    # 5. Evaluate
    y_true = test.values
    y_preds = {
        'ARIMA':    arima_pred,
        'Prophet':  prophet_pred,
        'LSTM':     lstm_preds
    }
    results = evaluate_forecasts(y_true, y_preds)
    print("\nModel Comparison:\n", results)
    return results

if __name__ == "__main__":
    results = main('../data/cleaned_consumer_complaints.csv')
