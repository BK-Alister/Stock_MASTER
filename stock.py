import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

# Date range setup
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Download stock data
google_data = yf.download(stock, start, end)

st.subheader("Stock Data")
st.write(google_data)

# Split data
splitting_len = int(len(google_data)*0.7)
x_test = google_data[['Close']][splitting_len:]

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange', label='Moving Average')
    plt.plot(full_data.Close, 'b', label='Close Price')
    if extra_data:
        plt.plot(extra_dataset, 'g', label='Extra MA')
    plt.legend()
    return fig

# Moving averages
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Prepare data for prediction
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test)

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i].flatten())
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data).ravel()  # Added ravel() here

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_data, y_data)

# Make predictions
predictions = model.predict(x_data)
predictions = predictions.reshape(-1, 1)

# Inverse transform predictions
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data.reshape(-1, 1))

# Create plotting dataframe
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Final plot - Fixed version
st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))

# Plot each series separately
plt.plot(google_data.index[:splitting_len+100], 
         google_data.Close[:splitting_len+100], 
         'b-', 
         label='Training Data')
plt.plot(ploting_data.index, 
         ploting_data['original_test_data'], 
         'g-', 
         label='Original Test')
plt.plot(ploting_data.index, 
         ploting_data['predictions'], 
         'r--', 
         label='Predictions')

plt.legend()
plt.title(f'{stock} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig)