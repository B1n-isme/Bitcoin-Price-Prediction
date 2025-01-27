import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

# Suppress specific ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)

def load_arima_garch_models(model_dir):
    # Load the ARIMAX & GARCH model
    arimax_results = joblib.load(f"{model_dir}/arimax_model.pkl")
    garch_fit = joblib.load(f"{model_dir}/garch_model.pkl")

    return arimax_results, garch_fit

def arima_garch_eval_old(model_dir, steps, actual, split_type, exog=None):
    # Load the ARIMAX & GARCH model
    sarima_model, garch_model = load_arima_garch_models(model_dir)

    sarima_pred = sarima_model.forecast(steps=steps, exog=exog).values.flatten()
    garch_volatility = garch_model.forecast(horizon=steps).variance.values[-1]

    arima_garch_pred = sarima_pred + garch_volatility

    arima_garch_rmse = root_mean_squared_error(actual, arima_garch_pred)  # RMSE
    arima_garch_mae = mean_absolute_error(actual, arima_garch_pred)  # MAE
    arima_garch_mape = (abs((actual - arima_garch_pred) / (actual + 1e-10)).mean()) * 100

    # Create a new DataFrame for ARIMA metrics
    arima_garch_metrics_df = pd.DataFrame({
        'Model': [f'sarima_garch_{split_type}'],
        'RMSE': [arima_garch_rmse],
        'MAE': [arima_garch_mae],
        'MAPE': [arima_garch_mape]
    })

    arima_garch_metrics_df.to_csv(f"../results/metrics/{split_type}_arima_garch_metrics.csv", index=False)

    residuals = actual - arima_garch_pred

    residuals_df = pd.DataFrame({
        'Date': actual.index,
        'SARIMA-GARCH Prediction': arima_garch_pred,
        'Residuals': residuals
    })

    residuals_df.to_csv(f"../data/final/{split_type}_residuals_df.csv", index=False)

def arima_garch_eval(model_dir, steps, actual, split_type, exog=None):
    # Load the ARIMAX & GARCH model
    sarima_model, garch_model = load_arima_garch_models(model_dir)

    sarima_pred = sarima_model.forecast(steps=steps, exog=exog).values.flatten()
    garch_volatility = garch_model.forecast(horizon=steps).variance.values[-1]

    arima_garch_pred = sarima_pred + garch_volatility

    rmse = root_mean_squared_error(actual, arima_garch_pred)  # RMSE
    mae = mean_absolute_error(actual, arima_garch_pred)  # MAE
    mape = (abs((actual - arima_garch_pred) / (actual + 1e-10)).mean()) * 100


    test_arima_garch_metrics_df = pd.read_csv(f"../results/metrics/{split_type}_arima_garch_metrics.csv")
    rmse2, mae2, mape2, steps2 = test_arima_garch_metrics_df[['RMSE', 'MAE', 'MAPE', 'Length']].values[0]

    # Compute weighted averages
    total_length = steps + steps2
    combined_rmse = (rmse * steps + rmse2 * steps2) / total_length
    combined_mae = (mae * steps + mae2 * steps2) / total_length
    combined_mape = (mape * steps + mape2 * steps2) / total_length

    # Create a new DataFrame for ARIMA metrics
    arima_garch_metrics_df = pd.DataFrame({
        'Model': [f'sarima_garch_{split_type}'],
        'RMSE': [combined_rmse],
        'MAE': [combined_mae],
        'MAPE': [combined_mape],
        'Length': [total_length]
    })

    arima_garch_metrics_df.to_csv(f"../results/metrics/{split_type}_arima_garch_metrics.csv", index=False)

    # Residuals
    residuals = actual - arima_garch_pred
    residuals_df = pd.DataFrame({
        'Date': actual.index,
        'SARIMA-GARCH Prediction': arima_garch_pred,
        'Residuals': residuals
    }).set_index("Date")

    return residuals_df


def arima_garch_forecast(exog, model_dir, future_days):

    future_dates = pd.date_range(start=exog.index[-1] + pd.Timedelta(days=1), periods=future_days, freq="D")

    # Create a DataFrame to store predicted values
    future_exog = pd.DataFrame(index=future_dates, columns=exog.columns)

    # Predict future values for each exogenous variable
    for col in exog.columns:
        # Extract SARIMA parameters for the current indicator from sarima_params
        # Fit ARIMA to the historical data
        model = SARIMAX(exog[col], order=(1,1,3))
        model_fit = model.fit(disp=False)

        # Forecast future values
        forecast = model_fit.forecast(steps=future_days, index=future_dates)
        future_exog[col] = forecast

    # Load the ARIMAX & GARCH model
    arimax_results = joblib.load(f"{model_dir}/arimax_model.pkl")
    garch_fit = joblib.load(f"{model_dir}/garch_model.pkl")

    # Forecast ARIMAX for the next 6 days
    arimax_forecast_future = arimax_results.forecast(steps=future_days, exog=future_exog).values.flatten()

    # Forecast GARCH for the next 6 days
    garch_forecast_future = garch_fit.forecast(horizon=future_days, method='simulation').variance.values[-1]

    random_noise = np.random.normal(loc=0, scale=garch_forecast_future, size=future_days)

    # Combine ARIMAX and GARCH forecasts (log-transformed scale)
    arimax_garch_future = arimax_forecast_future + random_noise

    return arimax_garch_future

def main():
    model_dir = "../../models"
    test_pca_df = pd.read_csv("../../data/final/test_pca_df.csv", parse_dates=["Date"], index_col="Date")
    test_exog = test_pca_df.drop(columns=["btc_close"])
    arima_garch_metrics_df, residuals_df= arima_garch_eval(model_dir, len(test_exog), test_pca_df["btc_close"], "test", test_exog)

    print(arima_garch_metrics_df)

if __name__ == "__main__":
    main()