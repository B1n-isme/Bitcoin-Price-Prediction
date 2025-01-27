import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(data_path):
    data_path_df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    return data_path_df

def scale_data(data, scaler):
    scaled_data = scaler.transform(data.values.reshape(-1, 1))
    return scaled_data

def create_lstm_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back, 0])
        y.append(data[i+look_back, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Add feature dimension
    y = y.reshape(-1, 1)
    return X, y

# create 1 step ahead dataset
def create_tft_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])  # Input sequence
        y.append(data[i + look_back, 0])   # Target
    
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Add feature dimension
    y = y.reshape(-1, 1)
    return X, y

