import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title("📈 Stock Trend Prediction using LSTM")

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

# ---- LSTM MODEL IMPLEMENTATION ----
st.subheader("🔮 Stock Price Forecasting using LSTM")

train_data = data["Close"].dropna().values.reshape(-1, 1)

if len(train_data) < 90:
    st.error("⚠️ Not enough data for LSTM. Select a longer date range.")
    st.stop()

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(train_data)

# Create sequences for LSTM
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(scaled_data, time_step)

# Split into train & test sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Reshape input for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mean_squared_error')
st.write("🔄 Training LSTM model...")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# ---- FORECAST FUTURE PRICES ----
st.write(f"📊 Forecasting Next {forecast_days} Days...")
last_sequence = X_test[-1]  # Take the last sequence from test data
predictions = []

for _ in range(forecast_days):
    pred = model.predict(last_sequence.reshape(1, time_step, 1), verbose=0)
    predictions.append(pred[0, 0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = pred  # Add prediction to sequence

# Transform back to original scale
forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Generate forecast dates
forecast_dates = pd.date_range(data.index[-1], periods=forecast_days + 1)[1:]

# Show forecast results
pred_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast.flatten()})
st.write(pred_df)

# ---- PLOT FORECAST ----
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(data.index, data["Close"], label="Historical Price", color="blue")
ax2.plot(forecast_dates, forecast, label="Forecasted Price", color="red", linestyle="dashed")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
st.pyplot(fig2)

st.success("✅ Prediction Completed! LSTM model successfully trained.")
st.sidebar.info("📌 LSTM is better for long-term patterns, but tuning is crucial for accuracy.")
