import os
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

def fit_sarima(series, exog=None, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    # Fit SARIMA model
    model = SARIMAX(series, exog=exog, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # In-sample predictions
    train_predictions = model_fit.predict(start=0, end=len(series) - 1)

    return {
        'model': model_fit,
        'train_predictions': train_predictions
    }

def fit_garch(residuals_train, p=1, o=0, q=1):
    # Clip residuals to avoid extreme values
    residuals_train = np.clip(
        residuals_train, 
        np.percentile(residuals_train, 1), 
        np.percentile(residuals_train, 99)
    )

    # Fit GARCH model
    garch_model = arch_model(residuals_train, vol='GARCH', p=p, o=o, q=q)
    garch_fit = garch_model.fit(disp="off")

    # In-sample volatility
    train_volatility = garch_fit.conditional_volatility

    return {
        'model': garch_fit,
        'train_volatility': train_volatility
    }

def forecast_sarima_garch(sarima_model, garch_model, steps, exog=None):
    sarima_pred = sarima_model.forecast(steps=steps, exog=exog)
    garch_volatility = garch_model.forecast(horizon=steps).variance.values[-1]
    return sarima_pred + garch_volatility

def main():
    # Load data
    train_pca_df = pd.read_csv("../../data/final/train_pca_df.csv", parse_dates=["Date"], index_col="Date")
    val_pca_df = pd.read_csv("../../data/final/val_pca_df.csv", parse_dates=["Date"], index_col="Date")
    test_pca_df = pd.read_csv("../../data/final/test_pca_df.csv", parse_dates=["Date"], index_col="Date")

    # Extract exogenous variables and target variable
    exog_vars = [col for col in train_pca_df.columns if col != 'btc_close']
    y_train = train_pca_df['btc_close']
    exog_train = train_pca_df[exog_vars]

    # Fit SARIMA model
    sarima_results = fit_sarima(
        y_train, 
        exog=exog_train, 
        order=(1, 1, 2), 
        seasonal_order=(0, 0, 0, 0)
    )

    # Save SARIMA model
    os.makedirs("../../models", exist_ok=True)
    joblib.dump(sarima_results['model'], "../../models/sarima_model.pkl")

    # Extract residuals from SARIMA
    residuals_train = y_train - sarima_results['train_predictions']

    # Fit GARCH model
    garch_results = fit_garch(residuals_train, p=1, o=0, q=1)

    # Save GARCH model
    joblib.dump(garch_results['model'], "../../models/garch_model.pkl")

    # Forecast on validation set
    val_exog = val_pca_df[exog_vars]
    val_actual = val_pca_df['btc_close']
    val_sarima_garch_pred = forecast_sarima_garch(
        sarima_results['model'], garch_results['model'], len(val_pca_df), exog=val_exog
    )

    # Calculate validation residuals
    train_residuals = val_actual - val_sarima_garch_pred

    # Save validation residuals
    val_residuals_df = pd.DataFrame({
        'Date': val_pca_df.index,
        'SARIMA-GARCH Prediction': val_sarima_garch_pred,
        'Residuals': train_residuals
    })
    val_residuals_df.to_csv("../../data/final/train_residuals_df.csv", index=False)

    # Forecast on test set
    test_exog = test_pca_df[exog_vars]
    test_actual = test_pca_df['btc_close']
    test_sarima_garch_pred = forecast_sarima_garch(
        sarima_results['model'], garch_results['model'], len(test_pca_df), exog=test_exog
    )

    # Calculate test residuals
    test_residuals = test_actual - test_sarima_garch_pred

    # Save test residuals
    test_residuals_df = pd.DataFrame({
        'Date': test_pca_df.index,
        'SARIMA-GARCH Prediction': test_sarima_garch_pred,
        'Residuals': test_residuals
    })
    test_residuals_df.to_csv("../../data/final/test_residuals_df.csv", index=False)

if __name__ == "__main__":
    main()
