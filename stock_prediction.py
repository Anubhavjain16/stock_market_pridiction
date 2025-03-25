import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title("📈 Stock Trend Prediction Web App")

# Sidebar Inputs
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
forecast_days = st.sidebar.slider("Forecast Days", 1, 100, 30)

# ---- FETCH STOCK DATA ----
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        return pd.DataFrame()

data = load_data(ticker, start_date, end_date)
if data.empty:
    st.error("❌ No data found for the given ticker. Please enter a valid stock ticker.")
    st.stop()

st.subheader(f"Stock Data for {ticker}")
st.write(data.tail())

# ---- PLOT CLOSING PRICE ----
st.subheader(f"📊 Closing Price Trend for {ticker}")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data["Close"], label="Closing Price", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# ---- TRAIN ARIMA MODEL ----
st.subheader("🔮 Stock Price Forecasting using ARIMA")
train_data = data["Close"].dropna()

# Check if there is enough data for ARIMA
if len(train_data) < 30:
    st.error("⚠️ Not enough data to train ARIMA. Please select a longer date range.")
    st.stop()

# Auto ARIMA for parameter tuning
st.write("🔄 Optimizing ARIMA parameters...")
try:
    best_model = pm.auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
    order = best_model.order
    st.write(f"📌 Best ARIMA Order: {order}")

    # Fit ARIMA model
    model = ARIMA(train_data, order=order)
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
    ax2.plot(forecast_dates, forecast, label="Forecasted Price", color="red", linestyle="dashed")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    st.pyplot(fig2)

    st.success("✅ Prediction Completed! Optimized with best ARIMA order.")

except Exception as e:
    st.error(f"❌ ARIMA Model failed: {e}")
    st.stop()

st.sidebar.info("📌 For better accuracy, consider LSTM or Facebook Prophet models.")
