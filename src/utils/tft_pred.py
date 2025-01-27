import joblib
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# from tft_model import TemporalFusionTransformer, TemporalFusionTransformerWrapper, tft_model_builder
# from data_preparation import *
# from arima_garch_pred import arima_garch_forecast
from utils.tft_model import TemporalFusionTransformer, TemporalFusionTransformerWrapper, tft_model_builder
from utils.data_preparation import *
from utils.arima_garch_pred import arima_garch_forecast

def load_best_model(model_path, look_back, device='cpu'):
    # Load best parameters
    params_path = f"{model_path}/tft_recursive_best_params.pkl"
    with open(params_path, 'rb') as f:
        best_params = joblib.load(f)

    # Recreate the model
    model = TemporalFusionTransformerWrapper(
        look_back=look_back,
        num_heads=best_params["module__num_heads"],
        head_dim=best_params["module__head_dim"],
        feed_forward_dim=best_params["module__feed_forward_dim"],
        dropout_rate=best_params["module__dropout_rate"],
        num_layers=best_params["module__num_layers"],
        activation=best_params["module__activation"],
        n_steps_ahead=1
    )

    # Load model weights
    weight_path = f"{model_path}/tft_recursive_best_model_weights.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load optimizer state
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    optimizer_path = f"{model_path}/tft_recursive_optimizer_state.pth"
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))

    # Move optimizer state to the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return model

def mc_dropout_predictions(model, X, n_simulations=100):
    model.train()  # Keep dropout layers active
    predictions = []

    with torch.no_grad():
        for _ in range(n_simulations):
            pred = model(X)  # Forward pass with active dropout
            predictions.append(pred.cpu().numpy())

    return np.array(predictions)

def TFT_eval_test(
    model_dir, results_dir, data_index, X_test, y_test, look_back, scaler, n_simulations=100
):
    device = 'cpu'
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

    # Perform MC Dropout
    model = load_best_model(model_dir, look_back, device)
    mc_predictions = mc_dropout_predictions(model, X_test_tensor, n_simulations=n_simulations)

    # Compute mean and uncertainty (standard deviation)
    y_pred = mc_predictions.mean(axis=0)  # Mean over num_samples
    uncertainty = mc_predictions.std(axis=0)  # Standard deviation over num_samples

    # Inverse transform the predictions
    y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_inverse = scaler.inverse_transform(y_test).flatten()

    # Evaluate the predictions
    rmse = root_mean_squared_error(y_test_inverse, y_pred_inverse)
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    mape = mean_absolute_percentage_error(y_test_inverse, y_pred_inverse)

    # save metrics as dataframe and save to csv
    metrics = pd.DataFrame({
        'Model_Type': ['Temporal-Fusion-Transformer'],
        'MSE': [rmse], 
        'MAE': [mae], 
        'MAPE': [mape],
        'Length': [len(y_test_inverse)]
    })
    metrics.to_csv(f'{results_dir}/metrics/tft_metrics.csv', index=False)

    # Flatten y_pred_inverse and uncertainty
    y_pred_inverse = y_pred_inverse.flatten()
    uncertainty = uncertainty.flatten()

    # create df to store predictions and uncertainty
    predictions_df = pd.DataFrame({
        "Temporal-Fusion-Transformer": y_pred_inverse
    }).set_index(data_index)

    uncertainties_df = pd.DataFrame({
        "Temporal-Fusion-Transformer": uncertainty
    }).set_index(data_index)

    predictions_df.to_csv(f'{results_dir}/predictions/test/tft_predictions.csv')
    uncertainties_df.to_csv(f'{results_dir}/predictions/test/tft_uncertainties.csv')

def TFT_eval_new(
    model_dir, results_dir, data_index, X_test, y_test, look_back, scaler, n_simulations=100
):
    device = 'cpu'
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

    # Perform MC Dropout
    model = load_best_model(model_dir, look_back, device)
    mc_predictions = mc_dropout_predictions(model, X_test_tensor, n_simulations=n_simulations)

    # Compute mean and uncertainty (standard deviation)
    y_pred = mc_predictions.mean(axis=0)  # Mean over num_samples
    uncertainty = mc_predictions.std(axis=0)  # Standard deviation over num_samples

    # Inverse transform the predictions
    y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_inverse = scaler.inverse_transform(y_test).flatten()

    # Evaluate the predictions
    rmse = root_mean_squared_error(y_test_inverse, y_pred_inverse)
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    mape = mean_absolute_percentage_error(y_test_inverse, y_pred_inverse)

    tft_metrics = pd.read_csv(f'{results_dir}/metrics/tft_metrics.csv')
    rmse2, mae2, mape2, length2 = tft_metrics[['MSE', 'MAE', 'MAPE', 'Length']].values[0]

    # Compute weighted averages
    total_length = len(y_test_inverse) + length2
    combined_rmse = (rmse * len(y_test_inverse) + rmse2 * length2) / total_length
    combined_mae = (mae * len(y_test_inverse) + mae2 * length2) / total_length
    combined_mape = (mape * len(y_test_inverse) + mape2 * length2) / total_length


    # save metrics as dataframe and save to csv
    metrics = pd.DataFrame({
        'Model_Type': ['Temporal-Fusion-Transformer'],
        'MSE': [combined_rmse], 
        'MAE': [combined_mae], 
        'MAPE': [combined_mape],
        'Length': [total_length]
    })
    metrics.index = metrics['Model_Type']
    # metrics.to_csv(f'{results_dir}/metrics/test_tft_metrics.csv', index=True)

    # Flatten y_pred_inverse and uncertainty
    y_pred_inverse = y_pred_inverse.flatten()
    uncertainty = uncertainty.flatten()

    # create df to store predictions and uncertainty
    predictions_df = pd.DataFrame({
        "Temporal-Fusion-Transformer": y_pred_inverse
    }).set_index(data_index)

    uncertainties_df = pd.DataFrame({
        "Temporal-Fusion-Transformer": uncertainty
    }).set_index(data_index)

    return predictions_df, uncertainties_df, metrics


def future_values(model, initial_input, n_steps):
    future_predictions = []
    current_input = initial_input.clone()

    for _ in range(n_steps):
        input_for_model = current_input.unsqueeze(0)
        with torch.no_grad():
            next_prediction = model(input_for_model)[0, 0].item()
        future_predictions.append(next_prediction)
        next_prediction_tensor = torch.tensor([[next_prediction]], dtype=torch.float32, device=current_input.device)
        current_input = torch.cat((current_input[1:], next_prediction_tensor), dim=0)

    return future_predictions


def TFT_forecast(X_test, test_residuals_df, scaler, arimax_garch_future, 
                                 model_path, look_back=7, future_days=7, device='cpu'):
    # Load the model
    model= load_best_model(model_path, look_back, device)

    # Prepare the input for prediction
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    last_known_data = X_test_tensor[-1]
    last_date = test_residuals_df.index[-1]

    # Predict future values
    future_predictions = future_values(model, last_known_data, future_days)

    # Inverse transform predictions
    flat_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Add ARIMAX-GARCH adjustments
    final_forecast = flat_predictions + arimax_garch_future

    # Undo log-transform
    final_forecast_org = np.exp(final_forecast) - 1

    # Generate future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(final_forecast_org))

    # Create predictions DataFrame
    future_prediction_df = pd.DataFrame({
        "Temporal-Fusion-Transformer": final_forecast_org
    }, index=future_dates)
    # future_prediction_df.index.name = "Date"

    return future_prediction_df

def main():
    model_dir = "../../models"
    results_dir = "../../results"
    look_back = 7

    # Load scaler
    residual_scaler = joblib.load(f"{model_dir}/residual_scaler.pkl")

    # Load the test set
    test_residuals_df = load_data("../../data/final/test_residuals_df.csv")
    test_residuals_scaled = scale_data(test_residuals_df["Residuals"], residual_scaler)
    X_test, y_test = create_tft_dataset(test_residuals_scaled, look_back)

    # Evaluate the model
    TFT_eval_test(model_dir, results_dir, test_residuals_df.index[look_back:], X_test, y_test, look_back, residual_scaler, n_simulations=100)

    # # Load the new dataset
    # new_pca_df = pd.read_csv("../data/final/new_pca_df.csv", parse_dates=["Date"], index_col="Date")
    # new_exog = new_pca_df.drop(columns=["btc_close"])
    # new_residuals_df = pd.read_csv("../data/final/new_residuals_df.csv", parse_dates=["Date"], index_col="Date")

    # # Forecast future values
    # arimax_garch_future = arima_garch_forecast(new_exog, model_dir, future_days=7)
    # future_df = TFT_forecast(new_residuals_df["Residuals"], new_residuals_df, scaler, arimax_garch_future, model_dir)
    # future_df.to_csv("../results/predictions/future/Temporal-Fusion-Transformer_forecast.csv")

if __name__ == "__main__":
    main()
