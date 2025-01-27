import os
import pandas as pd
import joblib
from utils.lstm_pred import predict_future

# Paths
test_path = "../data/final/test_residuals_df.csv"
model_dir = "../models"
future_pred_path = "../results/predictions/future_predictions.csv"

# Load test residuals and scaler
test_residuals_df = pd.read_csv(test_path, parse_dates=["Date"], index_col="Date")
test_residual = test_residuals_df["Residuals"]
test_arima_garch_pred = test_residuals_df["SARIMA-GARCH Prediction"]

scaler = joblib.load(f"{model_dir}/residual_scaler.pkl")
test_residual_scaled = scaler.transform(test_residual.values.reshape(-1, 1))

# Future predictions
look_back = 20
future_days = 7

future_predictions_df = predict_future(
    model_dir, ["LSTM", "BiLSTM", "Attention-LSTM", "Attention-BiLSTM"], look_back,
    scaler, test_residual_scaled, test_arima_garch_pred, future_days, future_pred_path
)

print("Future Predictions:")
print(future_predictions_df)
