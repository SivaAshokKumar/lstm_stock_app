import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import datetime

st.set_page_config(layout="wide")

# App title
st.title("📊 LSTM Stock Price Predictor")
st.markdown("Enter stock tickers, date range, and click **Predict** to visualize forecasts. Compare multiple stocks too!")

# Sidebar Inputs
st.sidebar.header("User Inputs")
tickers = st.sidebar.text_input("Enter comma-separated Ticker Symbols (e.g., AAPL,MSFT,GOOGL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2023, 10, 1))
time_steps = st.sidebar.slider("Time Steps (Days)", 30, 100, 60)
predict_button = st.sidebar.button("📈 Predict")

# Create sequences for LSTM
@st.cache_data(show_spinner=False)
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Normalize and prepare data for each ticker
def prepare_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = create_sequences(scaled, time_steps)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return df, scaler, X_train, y_train, X_test, y_test

# Build LSTM Model
def build_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Metrics
@st.cache_data(show_spinner=False)
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, r2, mape

# Predict future 30 days
def forecast_next_days(model, last_seq, scaler):
    future = []
    input_seq = last_seq.reshape(1, time_steps, 1)
    for _ in range(30):
        next_val = model.predict(input_seq, verbose=0)[0][0]
        future.append(next_val)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_val]]], axis=1)
    return scaler.inverse_transform(np.array(future).reshape(-1, 1))

# Run prediction
if predict_button:
    st.subheader("🔍 Forecast Results")
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    future_df_all = pd.DataFrame()

    for ticker in tickers_list:
        st.markdown(f"### {ticker} Results")
        with st.spinner(f"Processing {ticker}..."):
            data, scaler, X_train, y_train, X_test, y_test = prepare_data(ticker)
            model = build_model((time_steps, 1))

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

            history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                                validation_data=(X_test, y_test),
                                callbacks=[early_stop, reduce_lr], verbose=0)

            y_pred = model.predict(X_test, verbose=0)

            # Rescale
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Metrics
            mae, rmse, r2, mape = evaluate(y_test_rescaled, y_pred_rescaled)
            st.markdown(f"**R²:** `{r2:.4f}`  |  **RMSE:** `{rmse:.2f}`  |  **MAPE:** `{mape:.2f}%`  |  **Accuracy:** `{100-mape:.2f}%`")

            # Plot predicted vs actual
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y_test_rescaled, label='Actual')
            ax.plot(y_pred_rescaled, label='Predicted')
            ax.set_title(f"{ticker} - Test Prediction")
            ax.legend()
            st.pyplot(fig)

            # Forecast next 30 days
            last_60_days = scaler.transform(data[-time_steps:])
            future_pred = forecast_next_days(model, last_60_days, scaler)
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')

            df_forecast = pd.DataFrame({"Date": future_dates, f"{ticker}_Forecast": future_pred.flatten()})
            future_df_all = pd.merge(future_df_all, df_forecast, on="Date", how="outer") if not future_df_all.empty else df_forecast

            # Line Chart
            st.line_chart(df_forecast.set_index("Date"))

    # Show combined forecast & export
    if len(tickers_list) > 1:
        st.markdown("### 📁 Combined Forecast")
        st.dataframe(future_df_all.set_index("Date"))
        csv = future_df_all.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Combined Forecast CSV", csv, file_name="multi_stock_forecast.csv", mime="text/csv")
