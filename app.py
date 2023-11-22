'''
Goal of LSTM microservice:
1. LSTM microservice will accept the GitHub data from Flask microservice and will forecast the data for next 1 year based on past 30 days
2. It will also plot three different graph (i.e.  "Model Loss", "LSTM Generated Data", "All Issues Data") using matplot lib 
3. This graph will be stored as image in Google Cloud Storage.
4. The image URL are then returned back to Flask microservice.
'''
# Import all the required packages
from flask import Flask, jsonify, request, make_response
import os
from dateutil import *
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import time
from flask_cors import CORS

# Tensorflow (Keras & LSTM) related packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import required storage package from Google Cloud Storage
# from google.cloud import storage

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)

# Add response headers to accept all types of  requests

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add(
        "Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    return response

#  Modify response headers when returning to the origin

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set(
        "Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    return response

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Build LSTM Model
def build_LSTM_model(look_back):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Forecasting
def forecasta(model, data, look_back, future_periods=2):
    latest_data = data[-look_back:].reshape(1, 1, look_back)
    forecast = model.predict(latest_data)
    future_forecasts = [forecast[0][0]]

    for _ in range(1, future_periods):
        latest_data = np.append(latest_data[0][0][1:], forecast)
        latest_data = latest_data.reshape(1, 1, look_back)
        forecast = model.predict(latest_data)
        future_forecasts.append(forecast[0][0])

    return np.array(future_forecasts)


def process(data, column_name, time_col, repo_name, start_date):
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data[column_name])

    # Convert 'created_at' to datetime
    df['created_at'] = pd.to_datetime(df[time_col])

    # Set 'created_at' as the DataFrame index
    df.set_index('created_at', inplace=True)
    df.drop(time_col, axis=1, inplace=True)

    # Resample and sum counts (if data is daily, use 'D'; if weekly, use 'W')
    df_resampled = df['count'].resample('D').sum()

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_resampled.values.reshape(-1, 1))

    # Create the dataset
    look_back = 30
    X, y = create_dataset(scaled_data, look_back)

    # Reshape for LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # Build the LSTM model
    model = build_LSTM_model(look_back)

    # Train the model
    model.fit(X, y, batch_size=1, epochs=100)

    # Forecast
    future_forecasts = forecasta(model, scaled_data[-look_back:], look_back, future_periods=60)

    # Inverse transform forecasts
    future_forecasts = scaler.inverse_transform(future_forecasts.reshape(-1, 1))

    # Get the last date from the data
    last_date = df_resampled.index[-1]

    # Generate future dates for the forecast
    future_dates = [last_date + timedelta(days=i) for i in range(1, len(future_forecasts) + 1)]

    # Combine forecasts with dates

    forecast_data = [{'count': round(forecast[0]), 'day': date.strftime('%Y-%m-%d')} for forecast, date in zip(future_forecasts, future_dates)]

    return forecast_data

'''
API route path is  "/api/forecast"
This API will accept only POST request
'''

@app.route('/api/forecast', methods=['POST'])
def forecast():
    body = request.get_json()
    repo_name = body["repo_name"]

    # Determine the start date for forecasting
    # Get the current date
    start_date = datetime.now().date()

    created_issues = process(body, 'created_issues', 'day', repo_name, start_date)
    closed_issues = process(body, 'closed_issues', 'day', repo_name, start_date)

    # Construct the response
    json_response = {
        "created_issues_predict": created_issues,
        "closed_issues_predict": closed_issues,
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


# Run LSTM app server on port 8080
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
