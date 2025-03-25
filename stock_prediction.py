import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title("📈 Stock Trend Prediction Web App")

# Sidebar Inputs
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
forecast_days = st.sidebar.slider("Forecast Days", 1, 100, 30)

# ---- FETCH STOCK DATA ----
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

st.subheader(f"Stock Data for {ticker}")
data = load_data(ticker, start_date, end_date)
st.write(data.tail())

# ---- PLOT CLOSING PRICE ----
st.subheader(f"Closing Price Trend for {ticker}")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data["Close"], label="Closing Price", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# ---- TRAIN ARIMA MODEL ----
st.subheader("Stock Price Forecasting using ARIMA")
train_data = data["Close"].dropna()
model = ARIMA(train_data, order=(5,1,0))  # ARIMA(p=5, d=1, q=0)
fitted_model = model.fit()

# ---- FORECAST PRICES ----
forecast = fitted_model.forecast(steps=forecast_days)
forecast_dates = pd.date_range(data.index[-1], periods=forecast_days + 1)[1:]

# Show forecast results
st.write(f"📊 Forecasted Prices for Next {forecast_days} Days")
pred_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast.values})
st.write(pred_df)

# ---- PLOT FORECAST ----
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(data.index, data["Close"], label="Historical Price", color="blue")
ax2.plot(forecast_dates, forecast, label="Forecasted Price", color="red")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
st.pyplot(fig2)

st.success("✅ Prediction Completed!")
st.sidebar.info("This app predicts stock prices using ARIMA. For better accuracy, try LSTM or Facebook Prophet.")
