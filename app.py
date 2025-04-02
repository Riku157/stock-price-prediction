import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def predict_stock_price(model, data, scaler, days=30):
    last_60_days = data[-60:]
    predicted_prices = []
    
    for _ in range(days):
        input_data = np.array(last_60_days).reshape(1, 60, 1)
        predicted_price = model.predict(input_data)[0][0]
        predicted_prices.append(predicted_price)
        last_60_days = np.append(last_60_days[1:], predicted_price)
    
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    future_dates = pd.date_range(start=pd.to_datetime('today'), periods=days+1, freq='D')[1:]
    
    return pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices.flatten()})

def main():
    st.title("Stock Price Prediction with Deep Learning")
    st.sidebar.header("Input Parameters")
    
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    period = st.sidebar.selectbox("Select Data Period", ["1y", "3Y","5Y","max"], index=3)
    days = st.sidebar.slider("Days to Predict", min_value=1, max_value=60, value=30)
    model_path = "DSSL.NS_stock_pd_model.h5"
    
    if st.sidebar.button("Predict"):
        st.subheader(f"Stock Price Prediction for {ticker} ({period} Data)")
        data = fetch_stock_data(ticker, period)
        
        if not data.empty:
            scaled_data, scaler = preprocess_data(data)
            model = tf.keras.models.load_model(model_path)
            prediction = predict_stock_price(model, scaled_data, scaler, days)
            
            plt.figure(figsize=(10,5))
            plt.plot(data.index, data['Close'], label='Historical Prices', color='blue')
            plt.plot(prediction['Date'], prediction['Predicted Price'], label='Predicted Prices', color='red')
            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.legend()
            st.pyplot(plt)
            
            st.write("Predicted Prices:")
            st.dataframe(prediction)
        else:
            st.error("Invalid Stock Ticker or No Data Available")
    
if __name__ == "__main__":
    main()
