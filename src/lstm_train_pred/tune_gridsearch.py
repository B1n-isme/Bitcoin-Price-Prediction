import os
import pandas as pd
import joblib
from utils.data_preparation import load_data, scale_data, create_lstm_dataset
# from utils.gridsearch import tune_hyperparameters, lstm_model_builder
from utils.lstm_model import build_lstm, build_bilstm, build_attention_lstm, build_attention_bilstm, tune_hyperparameters, lstm_model_builder

# Paths
train_path = "../data/final/train_residuals_df.csv"
test_path = "../data/final/test_residuals_df.csv"
model_dir = "../models"
metrics_path = "../results/metrics/lstm_metrics.csv"
pred_dir = "../results/predictions/test"
future_pred_path = "../results/predictions/future_predictions.csv"
os.makedirs(model_dir, exist_ok=True)

# Load and preprocess data
train_residuals_df= load_data(train_path)
test_residuals_df= load_data(test_path)
train_residual = train_residuals_df["Residuals"]
test_residual = test_residuals_df["Residuals"]

train_residual_scaled, test_residuals_scaled, scaler = scale_data(train_residual, test_residual)

look_back = 7
X_train, y_train = create_lstm_dataset(train_residual_scaled, look_back)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# # Save scaler
# scaler_path = f"{model_dir}/residual_scaler.pkl"
# joblib.dump(scaler, scaler_path)

# Hyperparameter tuning
param_grid = {
    "units": [50, 100],
    "dropout": [0.2, 0.3],
    "learning_rate": [0.001, 0.0001],
    "batch_size": [16, 32]
}
model_types = ["LSTM", "BiLSTM", "Attention-LSTM", "Attention-BiLSTM"]

best_params, best_models = tune_hyperparameters(
    X_train, y_train, param_grid, model_types, input_shape=(look_back, 1)
)

# Save the best models and parameters
for model_type, model in best_models.items():
    model_file = f"{model_dir}/{model_type}_best_model.pkl"
    param_file = f"{model_dir}/{model_type}_best_params.pkl"
    joblib.dump(model, model_file)
    joblib.dump(best_params[model_type], param_file)

print("Training completed. Models and parameters saved.")
