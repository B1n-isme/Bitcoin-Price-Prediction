from utils.arima_garch_pred import *
from utils.pca_preprocessing import *
from utils.data_preparation import *
from utils.lstm_model import *
from utils.tft_model import *
from utils.lstm_pred import *
from utils.tft_pred import *

model_dir = "../models"
test_pca_df = pd.read_csv("../data/final/test_pca_df.csv", parse_dates=["Date"], index_col="Date")
# test_exog = test_pca_df.drop(columns=["btc_close"])
# arima_garch_metrics_df, residuals_df= arima_garch_eval(model_dir, len(test_exog), test_pca_df["btc_close"], "test", test_exog)

# print(arima_garch_metrics_df)

scaler = joblib.load("../models/scaler.pkl")
pca = joblib.load("../models/pca.pkl")
residual_scaler = joblib.load("../models/residual_scaler.pkl")

# new_dataset = pd.read_csv("../data/final/new_dataset.csv", parse_dates=["Date"], index_col="Date")
# # Remove the first 30 rows
# new_dataset = new_dataset.iloc[33:]
# # # Impute missing values using forward fill method
# new_dataset.ffill(inplace=True)
# # # Impute remaining missing values using backward fill method
# new_dataset.bfill(inplace=True)
# new_dataset.to_csv("../data/final/new_dataset.csv")

# suitable_col = [
#         'hash_rate_blockchain',  
#         'btc_sma_14', 'btc_ema_14', 
#         'btc_bb_high', 'btc_bb_low', 'btc_bb_mid', 'btc_bb_width', 
#         'btc_atr_14', 'btc_trading_volume', 'btc_volatility_index',
#         'ARK Innovation ETF', 'CBOE Volatility Index', 'Shanghai Composite Index', 
#         'btc_close'
#     ]
# new_dataset = apply_log_transform(new_dataset, suitable_col)
# new_pca_df = transform_dataset(new_dataset, scaler, pca, 'btc_close')

# new_pca_df_filtered = new_pca_df[~new_pca_df.index.isin(test_pca_df.index)]
# combined_pca_df = pd.concat([test_pca_df, new_pca_df_filtered])
# new_pca_df.to_csv("../data/final/new_pca_df.csv")
# combined_pca_df.to_csv("../data/final/test_pca_df.csv")

new_pca_df = pd.read_csv("../data/final/new_pca_df.csv", parse_dates=["Date"], index_col="Date")

new_exog = new_pca_df.drop(columns=["btc_close"])
new_residuals_df = arima_garch_eval(model_dir, len(new_exog), new_pca_df["btc_close"], "test", new_exog)

model_types = ["LSTM", "BiLSTM", "Attention-LSTM", "Attention-BiLSTM"]
results_dir = "../results"
look_back = 7

test_residuals_df = pd.read_csv("../data/final/test_residuals_df.csv", parse_dates=["Date"], index_col="Date")
new_residuals_df = pd.concat([test_residuals_df[-look_back:], new_residuals_df])

# Load scaler
new_residual_scaled = scale_data(new_residuals_df["Residuals"], residual_scaler)

# Create LSTM test set
LSTM_X_new, LSTM_y_new = create_lstm_dataset(new_residual_scaled, look_back)

predictions_df, uncertainties_df = LSTM_eval_new(
    model_dir, results_dir, model_types, new_residuals_df["Residuals"].index[look_back:], LSTM_X_new, LSTM_y_new, residual_scaler, n_simulations=100
)
predictions_df.to_csv("../results/predictions/test/new_lstm_predictions.csv")
uncertainties_df.to_csv("../results/predictions/test/new_lstm_uncertainties.csv")

# Create TFT test set
TFT_X_new, TFT_y_new = create_tft_dataset(new_residual_scaled, look_back)

predictions_df, uncertainties_df = TFT_eval_new(
    model_dir, results_dir, new_residuals_df["Residuals"].index[look_back:], TFT_X_new, TFT_y_new, look_back, residual_scaler, n_simulations=100
)

predictions_df.to_csv("../results/predictions/test/new_tft_predictions.csv")
uncertainties_df.to_csv("../results/predictions/test/new_tft_uncertainties.csv")


# val_pca_df = pd.read_csv("../data/final/val_pca_df.csv", parse_dates=["Date"], index_col="Date")
# test_pca_df = pd.read_csv("../data/final/test_pca_df.csv", parse_dates=["Date"], index_col="Date")
# val_exog = val_pca_df.drop(columns=["btc_close"])
# test_exog = test_pca_df.drop(columns=["btc_close"])
# exog = pd.concat([val_exog, test_exog])

# future_days = 7
# arimax_garch_future_df = arima_garch_forecast(exog, model_dir, future_days)
# print(arimax_garch_future_df)