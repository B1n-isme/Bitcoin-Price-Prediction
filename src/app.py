import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_preparation import *
from utils.pca_preprocessing import *
from utils.arima_garch_pred import arima_garch_eval, arima_garch_forecast
from utils.lstm_model import *
from utils.tft_model import *
from utils.lstm_pred import LSTM_eval_new, LSTM_forecast
from utils.tft_pred import TFT_eval_new, TFT_forecast
from utils.dataset import fetch_new_data

# Paths
model_dir = "../models"
results_dir = "../results"
metrics_dir = "../results/metrics"
test_arima_garch_pred_path = "../data/final/test_arima_garch_pred.csv"

look_back = 7
model_types = ["LSTM", "BiLSTM", "Attention-LSTM", "Attention-BiLSTM"]
model_options = ["LSTM", "BiLSTM", "Attention-LSTM", "Attention-BiLSTM", "Ensemble-LSTM", "Temporal-Fusion-Transformer"]

# Load data
if "dataset" not in st.session_state:
    st.session_state.dataset = load_data("../data/final/dataset.csv")
    st.session_state.val_pca_df = load_data("../data/final/val_pca_df.csv")
    st.session_state.test_pca_df = load_data("../data/final/test_pca_df.csv")
    st.session_state.train_residuals_df = load_data("../data/final/train_residuals_df.csv")
    st.session_state.test_residuals_df = load_data("../data/final/test_residuals_df.csv")
    st.session_state.scaler = joblib.load("../models/scaler.pkl")
    st.session_state.pca = joblib.load("../models/pca.pkl")
    st.session_state.residual_scaler = joblib.load("../models/residual_scaler.pkl")
    st.session_state.test_lstm_predictions = load_data("../results/predictions/test/lstm_predictions.csv")
    st.session_state.test_lstm_uncertainties = load_data("../results/predictions/test/lstm_uncertainties.csv")
    st.session_state.test_tft_predictions = load_data("../results/predictions/test/tft_predictions.csv")
    st.session_state.test_tft_uncertainties = load_data("../results/predictions/test/tft_uncertainties.csv")
    st.session_state.arima_garch_metrics = pd.read_csv(f"{metrics_dir}/test_arima_garch_metrics.csv", index_col="Model")
    st.session_state.arima_garch_metrics = st.session_state.arima_garch_metrics.T
    st.session_state.lstm_metrics = pd.read_csv(f"{metrics_dir}/lstm_metrics.csv", index_col="Model_Type")
    st.session_state.tft_metrics = pd.read_csv(f"{metrics_dir}/tft_metrics.csv", index_col="Model_Type")
    st.session_state.lstm_metrics = st.session_state.lstm_metrics.T
    st.session_state.tft_metrics = st.session_state.tft_metrics.T


# Load scaler
test_residual_scaled = scale_data(st.session_state.test_residuals_df["Residuals"], st.session_state.residual_scaler)

# Create LSTM test set
LSTM_X_test, LSTM_y_test = create_lstm_dataset(test_residual_scaled, look_back)

# Create TFT test set
TFT_X_test, TFT_y_test = create_tft_dataset(test_residual_scaled, look_back)

# Load ARIMA-GARCH predictions
val_arima_garch_log = st.session_state.train_residuals_df["SARIMA-GARCH Prediction"]
test_arima_garch_log = st.session_state.test_residuals_df["SARIMA-GARCH Prediction"]

# Undo the log transformation
val_arima_garch_pred = np.exp(val_arima_garch_log) - 1
test_arima_garch_pred = np.exp(test_arima_garch_log) - 1

# Load exogenous variables
val_exog = st.session_state.val_pca_df.drop(columns=["btc_close"])
test_exog = st.session_state.test_pca_df.drop(columns=["btc_close"])
exog = pd.concat([val_exog, test_exog])


def fetch_data(start_date):
    # Step 1: Fetch new data
    # new_dataset = fetch_new_data(start_date)
    new_dataset = pd.read_csv("../data/final/new_dataset.csv", parse_dates=["Date"], index_col="Date")
    
    # Impute remaining missing values using backward fill method
    new_dataset.ffill(inplace=True)
    new_dataset.bfill(inplace=True)

    # Append new data to the existing dataset
    new_dataset = new_dataset[~new_dataset.index.isin(st.session_state.dataset.index)]
    st.session_state.dataset = pd.concat([st.session_state.dataset, new_dataset])
    st.session_state.dataset = st.session_state.dataset.sort_index()
    # st.session_state.dataset.to_csv("../data/final/dataset.csv")

    # Step 1: Apply PCA to the new_dataset
    suitable_col = [
        'hash_rate_blockchain',  
        'btc_sma_14', 'btc_ema_14', 
        'btc_bb_high', 'btc_bb_low', 'btc_bb_mid', 'btc_bb_width', 
        'btc_atr_14', 'btc_trading_volume', 'btc_volatility_index',
        'ARK Innovation ETF', 'CBOE Volatility Index', 'Shanghai Composite Index', 
        'btc_close'
    ]
    new_dataset = apply_log_transform(new_dataset, suitable_col)
    new_pca_df = transform_dataset(new_dataset, st.session_state.scaler, st.session_state.pca, 'btc_close')
    
    # Update the test_pca_df with new_pca_df
    new_pca_df = new_pca_df[~new_pca_df.index.isin(st.session_state.test_pca_df.index)]
    st.session_state.test_pca_df = pd.concat([st.session_state.test_pca_df, new_pca_df])
    st.session_state.test_pca_df = st.session_state.test_pca_df.sort_index()
    # st.session_state.test_pca_df.to_csv("../data/final/test_pca_df.csv") #######################################

    # Step 2: Predict new_pca_df on ARIMA-GARCH
    new_exog = new_pca_df.drop(columns=["btc_close"])
    new_residuals_df = arima_garch_eval(model_dir, len(new_exog), new_pca_df["btc_close"], "test", new_exog)
    # concat test_residuals_df and new_residuals_df
    new_residuals_df = new_residuals_df[~new_residuals_df.index.isin(st.session_state.test_residuals_df.index)]
    # Add look_back of test_residuals_df to new_residuals_df to maintain continuity when create time series dataset
    new_residuals_df_2 = pd.concat([st.session_state.test_residuals_df[-look_back:], new_residuals_df])
    
    # Update the test_residuals_df with new_residuals_df
    st.session_state.test_residuals_df = pd.concat([st.session_state.test_residuals_df, new_residuals_df])
    st.session_state.test_residuals_df = st.session_state.test_residuals_df.sort_index()
    # st.session_state.test_residuals_df.to_csv("../data/final/test_residuals_df.csv") #######################################

    # Step 3: Predict new_residuals_df_2 on LSTM and TFT 
    # Load scaler
    new_residual_scaled = scale_data(new_residuals_df_2["Residuals"], st.session_state.residual_scaler)
    data_index = new_residuals_df_2["Residuals"].index[look_back:]

    # LSTM predictions
    LSTM_X_new, LSTM_y_new = create_lstm_dataset(new_residual_scaled, look_back)
    new_lstm_pred, new_lstm_uncertainty, lstm_metrics = LSTM_eval_new(
        model_dir, results_dir, model_types, data_index, LSTM_X_new, LSTM_y_new, st.session_state.residual_scaler, n_simulations=100
    )
    # Update lstm_predictions 
    new_lstm_pred = new_lstm_pred[~new_lstm_pred.index.isin(st.session_state.test_lstm_predictions.index)]
    st.session_state.test_lstm_predictions = pd.concat([st.session_state.test_lstm_predictions, new_lstm_pred])
    st.session_state.test_lstm_predictions = st.session_state.test_lstm_predictions.sort_index()
    # st.session_state.test_lstm_predictions.to_csv("../results/predictions/test/lstm_predictions.csv") #######################################
    # Update lstm_uncertainties
    new_lstm_uncertainty = new_lstm_uncertainty[~new_lstm_uncertainty.index.isin(st.session_state.test_lstm_uncertainties.index)]
    st.session_state.test_lstm_uncertainties = pd.concat([st.session_state.test_lstm_uncertainties, new_lstm_uncertainty])
    st.session_state.test_lstm_uncertainties = st.session_state.test_lstm_uncertainties.sort_index()
    # st.session_state.test_lstm_uncertainties.to_csv("../results/predictions/test/lstm_uncertainties.csv") #######################################

    # Update lstm_metrics
    st.session_state.lstm_metrics = lstm_metrics.T
    # st.session_state.lstm_metrics.to_csv(f"{results_dir}/metrics/test_lstm_metrics.csv", index=True)

    # TFT predictions
    TFT_X_new, TFT_y_new = create_tft_dataset(new_residual_scaled, look_back)
    new_tft_pred, new_tft_uncertainty, tft_metrics = TFT_eval_new(
        model_dir, results_dir, data_index, TFT_X_new, TFT_y_new, look_back, st.session_state.residual_scaler, n_simulations=100
    )

    # Update tft_predictions
    new_tft_pred = new_tft_pred[~new_tft_pred.index.isin(st.session_state.test_tft_predictions.index)]
    st.session_state.test_tft_predictions = pd.concat([st.session_state.test_tft_predictions, new_tft_pred])
    st.session_state.test_tft_predictions = st.session_state.test_tft_predictions.sort_index()
    # st.session_state.test_tft_predictions.to_csv("../results/predictions/test/tft_predictions.csv") #######################################
    # Update tft_uncertainties
    new_tft_uncertainty = new_tft_uncertainty[~new_tft_uncertainty.index.isin(st.session_state.test_tft_uncertainties.index)]
    st.session_state.test_tft_uncertainties = pd.concat([st.session_state.test_tft_uncertainties, new_tft_uncertainty])
    st.session_state.test_tft_uncertainties = st.session_state.test_tft_uncertainties.sort_index()
    # st.session_state.test_tft_uncertainties.to_csv("../results/predictions/test/tft_uncertainties.csv") #######################################

    # Update tft_metrics
    st.session_state.tft_metrics = tft_metrics.T
    # st.session_state.tft_metrics.to_csv(f'{results_dir}/metrics/test_tft_metrics.csv', index=True)


# Streamlit App
st.title("Future Bitcoin Price Prediction")

# Sidebar inputs
st.sidebar.image("../img/logo.png", width=200)

# Fetch latest data
if st.sidebar.button("Fetch Data"):
    # Get the last date from the dataset
    last_date = st.session_state.dataset.index[-1].date().strftime('%Y-%m-%d')  # Get the last date in main dataset
    last_33th_date = st.session_state.dataset.index[-33].date().strftime('%Y-%m-%d')  # Get the last date in main dataset
    today = datetime.now().date().strftime('%Y-%m-%d')  # Get today's date
    
    # Check if we need to fetch new data
    if last_date < today:
        st.write(f"Fetching new data from {last_date}...")
        # Fetch and update the dataset
        fetch_data(last_33th_date)
        st.success("Dataset updated successfully!")
    else:
        st.info("No need to fetch new data; the dataset is up-to-date.")

# Display dataset
st.line_chart(st.session_state.dataset["btc_close"])

st.sidebar.title("Input Parameters")
future_days = st.sidebar.number_input(
    "Enter the number of days for future prediction:",
    min_value=1,
    max_value=731,
    value=7,
    step=1,
)
selected_model = st.sidebar.selectbox("Select a Model:", model_options)


if st.sidebar.button("Predict"):
    st.write(f"Generating predictions for the next {future_days} days...")

    # Generate ARIMA-GARCH future predictions
    arimax_garch_future_df = arima_garch_forecast(exog, model_dir, future_days)

    print(arimax_garch_future_df)
    
    # Generate future predictions
    if selected_model == "Temporal-Fusion-Transformer":
        future_predictions_df = TFT_forecast(
            X_test=TFT_X_test,
            test_residuals_df=st.session_state.test_residuals_df,
            scaler=st.session_state.residual_scaler,
            arimax_garch_future=arimax_garch_future_df,
            model_path=model_dir,
            look_back=7,
            future_days=future_days,
            device='cpu'
        )
    else:
        future_predictions_df = LSTM_forecast(
            model_dir=model_dir,
            model_types=model_types,
            look_back=look_back,
            scaler=st.session_state.residual_scaler,
            test_residual_scaled=test_residual_scaled,
            test_arima_garch_log=test_arima_garch_log,
            future_days=future_days,
            save_path=None,
            arimax_garch_future=arimax_garch_future_df
        )

    # Display summary metrics in columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Future Days", value=future_days)
    with col2:
        st.metric(label="Selected Model", value=selected_model)

    st.markdown("---")  # Add a horizontal line to separate sections

    # Visualization for the selected model
    st.write(f"### Predicted Prices Across Train, Validation, Test, and Future ({selected_model} Model)")
    fig = go.Figure()


    # Add actual prices
    fig.add_trace(go.Scatter(
        x=st.session_state.dataset.index, y=st.session_state.dataset["btc_close"],
        mode='lines',
        name='Actual Prices',
        line=dict(color='white')
    ))

    # Add ARIMA-GARCH predictions
    fig.add_trace(go.Scatter(
        x=val_arima_garch_pred.index, y=val_arima_garch_pred,
        mode='lines',
        name='SARIMA-GARCH Predictions (Validation)',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=test_arima_garch_pred.index, y=test_arima_garch_pred,
        mode='lines',
        name='SARIMA-GARCH Predictions (Test)',
        line=dict(color='blue')
    ))

    # Add test predictions
    if selected_model == "Temporal-Fusion-Transformer":
        pred = st.session_state.test_tft_predictions[selected_model]
    else:
        pred = st.session_state.test_lstm_predictions[selected_model] 
    
    test_final_forecast = test_arima_garch_log.iloc[look_back:] + pred
    test_final_forecast_org = np.exp(test_final_forecast) - 1

    fig.add_trace(go.Scatter(
        x=st.session_state.test_residuals_df.index[look_back:], y=test_final_forecast_org,
        mode='lines',
        name=f'SARIMA-GARCH {selected_model} Hybrid Model Test Predictions',
        line=dict(color='green')
    ))

    # Add uncertainty bands
    if selected_model == "Temporal-Fusion-Transformer":
        uncertainty = st.session_state.test_tft_uncertainties[selected_model]
    else:
        uncertainty = st.session_state.test_lstm_uncertainties[selected_model] 
    
    lower_bound = np.exp(test_final_forecast - 2 * uncertainty) - 1
    upper_bound = np.exp(test_final_forecast + 2 * uncertainty) - 1

    # Add uncertainty bands using Plotly's fill functionality
    fig.add_trace(go.Scatter(
        x=st.session_state.test_residuals_df.index[look_back:],  # X values (dates)
        y=upper_bound,  # Upper bound values
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),  # Transparent line for the upper bound
        name=f'{selected_model} Uncertainty Upper Bound',
        showlegend=False  # Hide this trace from the legend
    ))

    fig.add_trace(go.Scatter(
        x=st.session_state.test_residuals_df.index[look_back:],  # X values (dates)
        y=lower_bound,  # Lower bound values
        mode='lines',
        fill='tonexty',  # Fill the area between this trace and the previous one
        fillcolor='rgba(128, 128, 128, 0.2)',  # Gray color with 20% opacity
        line=dict(color='rgba(0,0,0,0)'),  # Transparent line for the lower bound
        name=f'{selected_model} Uncertainty (±2 std)'  # Label for the legend
    ))

    # Add future predictions
    fig.add_trace(go.Scatter(
        x=future_predictions_df.index, y=future_predictions_df[selected_model],
        mode='lines',
        name=f'{selected_model} Future Predictions',
        line=dict(color='red')
    ))

    # Update layout to include range slider and range selector
    fig.update_layout(
        title=f"Bitcoin Price Predictions ({selected_model} Model)",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_white",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    # Display interactive chart
    st.plotly_chart(fig)

    # Display evaluation metrics
    st.markdown("---")  
    st.write("### Evaluation Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### ARIMA-GARCH")
        st.dataframe(st.session_state.arima_garch_metrics['sarima_garch_test'][:-1])
    with col2:
        st.write(f"#### {selected_model}")
        if selected_model == "Temporal-Fusion-Transformer":
            st.dataframe(st.session_state.tft_metrics[selected_model][:-1].apply(pd.to_numeric, errors='coerce'))
        else:
            st.dataframe(st.session_state.lstm_metrics[selected_model][:-1].apply(pd.to_numeric, errors='coerce'))


    # Display prediction values
    st.markdown("---")
    st.write("### Prediction Data")
    if selected_model == "Temporal-Fusion-Transformer":
        st.dataframe(future_predictions_df)
    else:
        st.dataframe(future_predictions_df[selected_model])

    # Download predictions
    st.download_button(
        label="Download Future Prediction Results",
        data=future_predictions_df.to_csv().encode('utf-8'),
        file_name='future_predictions.csv',
        mime='text/csv',
    )