import os
import pandas as pd
import joblib
from utils.data_preparation import load_data, scale_data, create_lstm_dataset
# from utils.gridsearch import tune_hyperparameters, lstm_model_builder
from utils.lstm_model import build_lstm, build_bilstm, build_attention_lstm, build_attention_bilstm, tune_hyperparameters, lstm_model_builder
from utils.lstm_pred import process_predictions_and_evaluation, save_predictions_and_uncertainties

# Paths
train_path = "../data/final/train_residuals_df.csv"
test_path = "../data/final/test_residuals_df.csv"
model_dir = "../models"
metrics_path = "../results/metrics/lstm_metrics.csv"
pred_dir = "../results/predictions/test"
future_pred_path = "../results/predictions/future_predictions.csv"
os.makedirs(pred_dir, exist_ok=True)

# Load and preprocess data
train_residuals_df, test_residuals_df = load_data(None, test_path)
train_residual = train_residuals_df["Residuals"]
test_residual = test_residuals_df["Residuals"]

scaler = joblib.load(f"{model_dir}/residual_scaler.pkl")
train_residual_scaled, test_residual_scaled, scaler = scale_data(train_residual, test_residual, scaler=scaler)

look_back = 7
X_test, y_test = create_lstm_dataset(test_residual_scaled, look_back)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Model types for prediction
model_types = ["LSTM", "BiLSTM", "Attention-LSTM", "Attention-BiLSTM"]

# Process predictions and evaluation
evaluation_metrics, predictions, uncertainties = process_predictions_and_evaluation(
    model_dir, model_types, X_test, y_test, scaler, n_simulations=100
)

# Save predictions, uncertainties, and evaluation metrics
save_predictions_and_uncertainties(predictions, uncertainties, pred_dir)

print("Evaluation Metrics:")
print(evaluation_metrics)
