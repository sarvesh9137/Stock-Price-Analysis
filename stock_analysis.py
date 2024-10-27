# LSTM is the better model in this case because it has a lower RMSE,
# suggesting that its predictions are closer to the actual stock prices
# than those of the SVM model.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math

# Helper functions
def calculate_annual_return(data):
    """Calculate annual return of the stock based on closing prices."""
    return (data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (252 / len(data)) - 1

def calculate_standard_deviation(data):
    """Calculate standard deviation of the stock's closing price."""
    return np.std(data['Close'])

def calculate_risk_adjusted_return(annual_return, std_dev):
    """Calculate risk-adjusted return (Sharpe Ratio)."""
    return annual_return / std_dev

# Streamlit Sidebar - Ticker Input
st.sidebar.title("Stock Price Analysis using LSTM and SVM")
ticker = st.sidebar.text_input("Enter the Stock Ticker Symbol (e.g., AAPL, TSLA)", value='')
start_date = st.sidebar.date_input('Start date')
end_date = st.sidebar.date_input('End date')

# Fetch data from Yahoo Finance
if ticker:
    # Download stock data using yfinance
    df = yf.download(ticker, start=start_date, end=end_date)
    if not df.empty:
        st.write(f"## {ticker} Stock Price Data")
        st.write(df)

        # Show basic statistics
        st.write("### Data Statistics")
        data2 = df.copy()
        data2['% Change'] = df['Adj Close'] / df['Adj Close'].shift(1) - 1
        data2.dropna(inplace=True)

        # Plot the Closing Price
        st.write("### Closing Price Over Time")
        fig = px.line(df, x=df.index, y=df['Adj Close'], title=f'{ticker} Adjusted Close Price')
        st.plotly_chart(fig)

        # Calculate and display financial metrics
        st.write("### Financial Metrics")
        annual_return = data2['% Change'].mean() * 252 * 100
        st.write(f'Annual Return is: {annual_return:.2f}%')
        
        stdev = np.std(data2['% Change']) * np.sqrt(252)
        st.write(f'Standard Deviation is: {stdev * 100:.2f}%')
        
        risk_adj_return = annual_return / (stdev * 100)
        st.write(f'Risk Adjusted Return is: {risk_adj_return:.2f}')

        # Preprocessing for LSTM Model
        scaler = StandardScaler()
        df['Scaled Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        # Preparing data for LSTM
        def prepare_data_lstm(data, time_step=30):
            """Prepares data for LSTM model by creating sequences of data."""
            X, y = [], []
            for i in range(time_step, len(data)):
                X.append(data[i-time_step:i])  # Create the input sequence
                y.append(data[i])  # Create the output value
            return np.array(X), np.array(y)

        # Split data into training and testing for LSTM
        train_size = int(len(df) * 0.7)
        train_data = df['Scaled Close'].values[:train_size]
        test_data = df['Scaled Close'].values[train_size - 30:]

        X_train, y_train = prepare_data_lstm(train_data)
        X_test, y_test = prepare_data_lstm(test_data)

        # Reshape data for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTM Model
        st.write("### LSTM Model")
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Check if data is not empty before fitting the model
        if X_train.size > 0 and y_train.size > 0:
            model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
        else:
            st.warning("Training data is empty, please check your input data.")

        # LSTM Predictions
        lstm_pred = model.predict(X_test)
        lstm_pred = scaler.inverse_transform(lstm_pred)

        # SVM Model
        st.write("### SVM Model")
        svm_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svm_model.fit(np.arange(len(train_data)).reshape(-1, 1), train_data)

        # Correct prediction length
        svm_pred = svm_model.predict(np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1))
        svm_pred = scaler.inverse_transform(svm_pred.reshape(-1, 1))

        # Align lengths for plotting
        if len(lstm_pred) > len(df.index[train_size:]):
            lstm_pred = lstm_pred[:len(df.index[train_size:])]
        if len(svm_pred) > len(df.index[train_size:]):
            svm_pred = svm_pred[:len(df.index[train_size:])]

        # Create a DataFrame to compare original vs predictions
        comparison_df = pd.DataFrame({
            'Date': df.index[train_size:],
            'Original': df['Close'].iloc[train_size:].values,
            'LSTM Predictions': lstm_pred.flatten(),
            'SVM Predictions': svm_pred.flatten()
        })

        st.write("### Original Values vs Predicted Values")
        st.dataframe(comparison_df)

        # Plot Predictions
        st.write("### Model Predictions")
        fig, ax = plt.subplots()
        ax.plot(df.index[train_size:], df['Close'].iloc[train_size:], label='Original Price')
        ax.plot(df.index[train_size:], lstm_pred, label='LSTM Predictions')
        ax.plot(df.index[train_size:], svm_pred, label='SVM Predictions')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Show error metrics
        st.write("### Model Evaluation")
        lstm_rmse = math.sqrt(mean_squared_error(df['Close'].iloc[train_size:], lstm_pred))
        svm_rmse = math.sqrt(mean_squared_error(df['Close'].iloc[train_size:], svm_pred))

        st.write(f"**LSTM RMSE:** {lstm_rmse:.4f}")
        st.write(f"**SVM RMSE:** {svm_rmse:.4f}")
    else:
        st.write("Could not fetch the data. Please check the ticker symbol or date range.")
else:
    st.write("Please enter a ticker symbol to get started!")

st.write("Copyrights Ⓒ 2024. All Rights Reserved. Code With Sarvesh(CWS) ")