# src/model_training.py

from pmdarima import auto_arima
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_arima(train_array):
    """
    Fit an auto-ARIMA on the training array and return the fitted model.
    """
    # Automatically fits the best ARIMA model to the data
    # `seasonal=True`: assumes the data has seasonality
    # `m=12`: sets the seasonal period (e.g., 12 for monthly data with yearly seasonality)
    # `stepwise=True`: uses a stepwise search to reduce computation time
    # `suppress_warnings=True`: suppresses convergence and other warnings
    
    model = auto_arima(
        train_array,            # Input time series data (e.g., a 1D NumPy array or Pandas Series)
        seasonal=True,          # Enables modeling of seasonal effects
        m=12,                   # Seasonality period (e.g., 12 months)
        stepwise=True,          # Uses a faster stepwise approach for model selection
        suppress_warnings=True  # Suppresses warnings during model fitting
    )
    return model

def train_prophet(df_prophet):
    """
    Fit a Prophet model. 
    Expects df_prophet with columns ['ds', 'y'].
    """
    m = Prophet(yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False)
    m.fit(df_prophet)
    return m

def train_lstm(train_supervised, n_lags, epochs=50, batch_size=32):
    """
    Train a simple LSTM on lagged data.
    train_supervised should be a NumPy array where:
      - columns 1..n_lags are inputs
      - column 0 is the target y(t)
    """
    # reshape for LSTM [samples, timesteps, features]
    X = train_supervised[:, 1:].reshape(-1, n_lags, 1)
    y = train_supervised[:, 0]

    model = Sequential([
        LSTM(50, input_shape=(n_lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)
    return model
