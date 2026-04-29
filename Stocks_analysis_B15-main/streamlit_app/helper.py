

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime as dt
import os
from pathlib import Path

import pandas as pd

import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests

from statsmodels.tsa.ar_model import AutoReg

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


ALPHA_VANTAGE_API_KEY = 'ECP7NGZZHBIK99YH'

def fetch_stocks():
    csv_path = Path(__file__).resolve().parent / "../data/equity_issuers.csv"
    df = pd.read_csv(
        csv_path,
        index_col=False,
        dtype={"Security Code": str, "Security Id": str},
    )
    df = df[["Issuer Name", "Security Code", "Security Id"]].dropna(subset=["Security Code", "Security Id"])
    stock_dict = dict(
        zip(
            df["Issuer Name"],
            df[["Security Code", "Security Id"]].to_dict("records"),
        )
    )
    return stock_dict


def build_stock_ticker(stock_entry, stock_exchange):
    if stock_exchange == "BSE":
        return f"{stock_entry['Security Code']}.BO"
    return f"{stock_entry['Security Id']}.NS"


def fetch_periods_intervals():
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }
    return periods
def safe_get(data_dict, key):
    return data_dict.get(key, "N/A")

# Fetch stock info from Alpha Vantage
def fetch_stock_history(
    stock_ticker,
    period=None,
    interval=None,
    api_key=ALPHA_VANTAGE_API_KEY,
):
    """
    Fetch stock price history using yfinance when period/interval are provided,
    otherwise fall back to Alpha Vantage daily data.

    Returns:
        pd.DataFrame: DataFrame with columns Open, High, Low, Close.
    """
    try:
        if period and interval:
            stock_data = yf.Ticker(stock_ticker)
            data = stock_data.history(period=period, interval=interval)
            return data[["Open", "High", "Low", "Close"]]

        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=stock_ticker, outputsize='compact')

        # Rename columns to standard format
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close'
        })

        return data[["Open", "High", "Low", "Close"]]
    except Exception as e:
        print("Error fetching stock history:", e)
        return pd.DataFrame()

def fetch_stock_info(stock_ticker, api_key=ALPHA_VANTAGE_API_KEY):
    try:
        stock_data = yf.Ticker(stock_ticker)
        info = stock_data.info

        if not info or not info.get("symbol"):
            raise ValueError("Invalid symbol or data not found.")

        stock_data_info = {
            "Basic Information": {
                "Symbol": safe_get(info, "symbol"),
                "Name": safe_get(info, "longName"),
                "Currency": safe_get(info, "currency"),
                "Exchange": safe_get(info, "exchange"),
                "Sector": safe_get(info, "sector"),
                "Industry": safe_get(info, "industry"),
            },
            "Market Data": {
                "CurrentPrice": safe_get(info, "currentPrice"),
                "Open": safe_get(info, "open"),
                "Close": safe_get(info, "previousClose"),
                "MarketCap": safe_get(info, "marketCap"),
                "EBITDA": safe_get(info, "ebitda"),
                "PERatio": safe_get(info, "trailingPE"),
                "PEGRatio": safe_get(info, "pegRatio"),
                "BookValue": safe_get(info, "bookValue"),
                "DividendPerShare": safe_get(info, "dividendRate"),
                "DividendYield": safe_get(info, "dividendYield"),
                "EPS": safe_get(info, "trailingEps"),
                "RevenueTTM": safe_get(info, "totalRevenue"),
                "ProfitMargin": safe_get(info, "profitMargins"),
                "52WeekHigh": safe_get(info, "fiftyTwoWeekHigh"),
                "52WeekLow": safe_get(info, "fiftyTwoWeekLow"),
            }
        }

        return stock_data_info

    except Exception as e:
        print("Error fetching stock info:", e)
        return {}
def generate_stock_prediction(stock_ticker):
    try:
        if tf is None:
            raise ModuleNotFoundError(
                "TensorFlow is not installed. The prediction model currently requires a Python version supported by TensorFlow."
            )

        # Fetch stock data
        stock_data = yf.Ticker(stock_ticker)
        stock_data_hist = stock_data.history(period="2y", interval="1d")
        stock_data_close = stock_data_hist[["Close"]]

        # Fill missing data
        stock_data_close = stock_data_close.asfreq("D", method="ffill").ffill()

        if stock_data_close.empty or len(stock_data_close) <= 300:
            raise ValueError("Not enough historical data for prediction.")

        # Train-test split for AutoReg
        train_df = stock_data_close.iloc[: int(len(stock_data_close) * 0.9) + 1]
        test_df = stock_data_close.iloc[int(len(stock_data_close) * 0.9) :]

        # AutoReg Model (AR)
        model_ar = AutoReg(train_df["Close"], 250).fit(cov_type="HC0")
        predictions_ar = model_ar.predict(start=test_df.index[0], end=test_df.index[-1], dynamic=True)
        forecast_ar = model_ar.predict(start=test_df.index[0], end=test_df.index[-1] + dt.timedelta(days=90), dynamic=True)

        # MinMax scaling for LSTM and RNN
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data_close)

        # Prepare training and testing datasets for LSTM and RNN
        def create_dataset(data, time_step=60):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i : i + time_step])
                y.append(data[i + time_step])
            return np.array(X), np.array(y)

        time_step = 60
        train_size = int(len(scaled_data) * 0.9)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Not enough sequence data for LSTM/RNN prediction.")

        # Reshape data for LSTM/RNN models
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTM model
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dense(1),
        ])
        lstm_model.compile(optimizer="adam", loss="mean_squared_error")
        lstm_model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
        lstm_predictions = lstm_model.predict(X_test)
        lstm_predictions = scaler.inverse_transform(lstm_predictions)

        # RNN model
        rnn_model = tf.keras.models.Sequential([
            tf.keras.layers.SimpleRNN(100, return_sequences=True, input_shape=(time_step, 1)),
            tf.keras.layers.SimpleRNN(100),
            tf.keras.layers.Dense(1),
        ])
        rnn_model.compile(optimizer="adam", loss="mean_squared_error")
        rnn_model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
        rnn_predictions = rnn_model.predict(X_test)
        rnn_predictions = scaler.inverse_transform(rnn_predictions)
        rnn_predictions_flat = rnn_predictions.flatten()
        actual_values = test_df["Close"].values
        # Calculate RMSE
        def calculate_rmse(actual, predicted):
            return np.sqrt(mean_squared_error(actual, predicted))

        rnn_rmse = calculate_rmse(actual_values, predictions_ar)

        # Calculate MAPE (Mean Absolute Percentage Error)
        def calculate_mape(actual, predicted):
            actual = np.where(actual == 0, np.nan, actual)
            return np.nanmean(np.abs((actual - predicted) / actual)) * 100

        rnn_mape = calculate_mape(actual_values, predictions_ar)

        return train_df, test_df, forecast_ar, lstm_predictions, rnn_predictions, predictions_ar, rnn_mape

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None, None, None
