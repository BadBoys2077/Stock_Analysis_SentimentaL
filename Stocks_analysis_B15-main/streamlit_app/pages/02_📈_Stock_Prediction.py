import os
import sys

import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import *
from style_utils import inject_global_css, render_sidebar_brand, render_page_header, render_footer, get_plotly_template

st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📈",
    layout="wide",
)

inject_global_css()
render_sidebar_brand()

# Sidebar
st.sidebar.markdown("## **Controls**")
stock_dict = fetch_stocks()
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))
stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)
stock_ticker = build_stock_ticker(stock_dict[stock], stock_exchange)
st.sidebar.text_input("Stock ticker code", value=stock_ticker, disabled=True)
periods = fetch_periods_intervals()
period = st.sidebar.selectbox("Choose a period", list(periods.keys()))
interval = st.sidebar.selectbox("Choose an interval", periods[period])

render_page_header("Stock Prediction", f"Data-driven forecasting for {stock}")

plotly_style = get_plotly_template()

# Historical data candlestick chart
stock_data = fetch_stock_history(stock_ticker, period, interval)
st.markdown("## **Historical Data**")

if stock_data is None or stock_data.empty or not {"Open", "High", "Low", "Close"}.issubset(stock_data.columns):
    st.warning(
        f"Could not load historical data for {stock_ticker}. "
        "Yahoo Finance is rate-limiting the request — try a different stock, "
        "switch the period/interval, or refresh in a minute."
    )
    st.stop()

fig = go.Figure(
    data=[
        go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
            increasing_line_color="#c3e88d",
            decreasing_line_color="#f07178",
            increasing_fillcolor="#c3e88d",
            decreasing_fillcolor="#f07178",
        )
    ]
)
fig.update_layout(**plotly_style)
fig.update_layout(
    xaxis_rangeslider_visible=False,
    title="Historical Price Data",
    height=500,
    margin=dict(l=20, r=20, t=50, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# ML Predictions
train_df, test_df, forecast_ar, lstm_predictions, rnn_predictions, predictions_ar, rnn_mape = generate_stock_prediction(stock_ticker)

if train_df is not None and (forecast_ar >= 0).all() and (predictions_ar >= 0).all() and (lstm_predictions >= 0).all() and (rnn_predictions >= 0).all():
    st.markdown("## **Stock Prediction**")

    fig = go.Figure(
        data=[
            go.Scatter(
                x=train_df.index,
                y=train_df["Close"],
                name="Train",
                mode="lines",
                line=dict(color="#82aaff"),
            ),
            go.Scatter(
                x=test_df.index,
                y=test_df["Close"],
                name="Test",
                mode="lines",
                line=dict(color="#ffcb6b"),
            ),
            go.Scatter(
                x=forecast_ar.index,
                y=forecast_ar,
                name="Forecast (AR)",
                mode="lines",
                line=dict(color="#f07178"),
            ),
            go.Scatter(
                x=test_df.index,
                y=predictions_ar,
                name="AR Predictions",
                mode="lines",
                line=dict(color="#c3e88d"),
            ),
            go.Scatter(
                x=test_df.index,
                y=lstm_predictions.flatten(),
                name="LSTM Predictions",
                mode="lines",
                line=dict(color="#c792ea"),
            ),
            go.Scatter(
                x=test_df.index,
                y=rnn_predictions.flatten(),
                name="RNN Predictions",
                mode="lines",
                line=dict(color="#56c6cc"),
            ),
        ]
    )

    fig.update_layout(**plotly_style)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        title="Multi-Model Price Prediction",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAPE (Mean Absolute Percentage Error)", f"{rnn_mape:.2f}%")
    with col2:
        st.metric("Model Accuracy", f"{100 - rnn_mape:.2f}%")
else:
    st.markdown("## **Stock Prediction**")
    st.warning("No prediction data available for the selected stock. Ensure sufficient historical data exists.")
    if tf is None:
        st.info("The prediction model is unavailable in this environment because TensorFlow does not support Python 3.14 here. Use Python 3.12 or 3.11 to enable LSTM/RNN prediction.")

render_footer()
