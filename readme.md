# 📈 Stock Trend Prediction using LSTM

## 🚀 Overview
This project is a **Stock Trend Prediction System** built with **Streamlit** for the UI and **LSTM (Long Short-Term Memory) neural networks** for time-series forecasting. The system allows users to enter a stock ticker, fetch historical stock data, train an LSTM model, and predict future stock prices.

## 🏗️ Features
- **User-friendly UI** with Streamlit
- **Live stock data fetching** from Yahoo Finance
- **Interactive visualizations** of stock trends
- **LSTM-based forecasting** for stock price prediction
- **Customizable forecasting period** (1 to 100 days)
- **Automatic data normalization & preprocessing**

## 🔧 Installation & Setup
### 1️⃣ Install Dependencies
Ensure you have Python installed (Python 3.7+ recommended). Install the required packages:
```sh
pip install streamlit yfinance pandas numpy matplotlib tensorflow scikit-learn
```

### 2️⃣ Run the Application
Execute the following command in your terminal:
```sh
streamlit run app.py
```

## 🖥️ How to Use
1. **Enter a stock ticker** in the sidebar (e.g., AAPL, TSLA).
2. **Select a date range** for historical stock data.
3. **Choose the number of days to forecast**.
4. **View real-time stock trends** and predicted prices.
5. **Analyze stock performance** with interactive charts.

## 📊 Model Architecture
The system implements an **LSTM-based deep learning model** with the following structure:
- **Input Layer:** Takes a 60-day historical window as input
- **LSTM Layer 1:** 50 units with return sequences enabled
- **Dropout Layer:** 20% dropout for regularization
- **LSTM Layer 2:** 50 units (no return sequences)
- **Dense Layers:** Two fully connected layers (25 & 1 neurons)
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)

## 📈 Results & Visualization
- The **Closing Price Trend** is plotted using historical data.
- The **LSTM-based prediction** is visualized alongside real stock prices.
- The **forecast is dynamically updated** based on user input.

## ⚠️ Limitations
- **Stock market predictions are inherently uncertain**.
- **LSTMs require fine-tuning** and hyperparameter optimization.
- **Short-term predictions tend to be more accurate than long-term ones**.

## 🛠️ Future Improvements
- Adding **Sentiment Analysis** on financial news.
- Implementing **GRU** and **Transformer-based models**.
- Fine-tuning **hyperparameters** dynamically.
- Deploying on **AWS/GCP for scalability**.

## 👨‍💻 Contributors
- **Your Name** - Developer

## 📜 License
This project is licensed under the MIT License.

---

🌟 **If you find this useful, give it a star!** 🌟

