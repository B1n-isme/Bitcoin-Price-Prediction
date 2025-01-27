import os
import pandas as pd
import joblib
from utils.data_preparation import load_data, scale_data, create_lstm_dataset
# from utils.gridsearch import tune_hyperparameters, lstm_model_builder
from utils.lstm_pred import process_predictions_and_evaluation, save_predictions_and_uncertainties, predict_future

from utils.lstm_model import build_lstm, build_bilstm, build_attention_lstm, build_attention_bilstm, tune_hyperparameters, lstm_model_builder

# Paths
train_path = "../data/final/train_residuals_df.csv"
test_path = "../data/final/test_residuals_df.csv"
model_dir = "../models"
metrics_path = "../results/metrics/lstm_metrics.csv"
pred_dir = "../results/predictions/test"
future_pred_path = "../results/predictions/future_predictions.csv"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

# Load and preprocess data
train_residuals_df = load_data(train_path)
test_residuals_df = load_data(test_path)
train_residual = train_residuals_df["Residuals"]
test_residual = test_residuals_df["Residuals"]
test_arima_garch_pred = test_residuals_df["SARIMA-GARCH Prediction"]

train_residual_scaled, test_residual_scaled, scaler = scale_data(train_residual, test_residual)

look_back = 20
X_train, y_train = create_lstm_dataset(train_residual_scaled, look_back)
X_test, y_test = create_lstm_dataset(test_residual_scaled, look_back)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

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

# Model types for prediction
model_types = ["LSTM", "BiLSTM", "Attention-LSTM", "Attention-BiLSTM"]

# Process predictions and evaluation
evaluation_metrics, predictions, uncertainties = process_predictions_and_evaluation(
    model_dir, model_types, X_test, y_test, scaler, n_simulations=100
)

# Save predictions, uncertainties, and evaluation metrics
save_predictions_and_uncertainties(predictions, uncertainties, pred_dir)

metrics_df = pd.DataFrame.from_dict(evaluation_metrics, orient="index").reset_index()
metrics_df.columns = ["Model_Type", "RMSE", "MAE", "MAPE", "Best params"]
metrics_df.to_csv(metrics_path, index=False)

print("Evaluation Metrics:")
print(metrics_df)

# Future predictions
future_days = 7

future_predictions_df = predict_future(
    model_dir, model_types, look_back, scaler, test_residual_scaled, test_arima_garch_pred, future_days, future_pred_path
)

print("Future Predictions:")
print(future_predictions_df)



