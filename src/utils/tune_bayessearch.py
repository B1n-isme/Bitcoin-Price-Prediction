import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skopt import BayesSearchCV
from skorch import NeuralNetRegressor
from sklearn.preprocessing import MinMaxScaler
from skopt.space import Real, Categorical
import math
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from skorch.callbacks import Callback
from tqdm import tqdm
from utils.data_preparation import load_data, scale_data, create_tft_dataset
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from utils.tft_model import TemporalFusionTransformer, TemporalFusionTransformerWrapper, tft_model_builder, SKTimeToSKLearnCV

device = torch.device("cpu")

train_path = "../data/final/train_residuals_df.csv"
test_path = "../data/final/test_residuals_df.csv"
model_dir = "../models"
metrics_path = "../results/metrics/tft_metrics.csv"
pred_dir = "../results/predictions/test/"
future_pred_path = "../results/predictions/future_predictions.csv"

# Load and preprocess data
train_residuals_df= load_data(train_path)
test_residuals_df= load_data(test_path)
train_residual = train_residuals_df["Residuals"]
test_residual = test_residuals_df["Residuals"]
train_residual_scaled, test_residual_scaled, scaler = scale_data(train_residual, test_residual)
train_residual_scaled = train_residual_scaled.astype(np.float32)
test_residual_scaled = test_residual_scaled.astype(np.float32)

# Create train and test datasets
look_back = 7
X_train, y_train = create_tft_dataset(train_residual_scaled, look_back)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Add feature dimension
y_train = y_train.reshape(-1, 1)
X_test, y_test = create_tft_dataset(test_residual_scaled, look_back)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # Add feature dimension
y_test = y_test.reshape(-1, 1)

# Define custom ExpandingWindowSplitter
expanding_splitter = ExpandingWindowSplitter(initial_window=365, step_length=30, fh=[1, 7])

# Define search space for BayesSearchCV
search_space = {
    "module__num_heads": Categorical([2, 4, 8]),  
    "module__head_dim": Categorical([8, 16, 32]),  
    "module__feed_forward_dim": Categorical([128, 256, 512, 1024]),  
    "module__dropout_rate": Real(0.1, 0.4),  
    "lr": Real(5e-5,5e-3, prior="log-uniform"), 
    "batch_size": Categorical([16, 32, 64]), 
    "module__num_layers": Categorical([1, 2, 3, 4]),
    "module__activation": Categorical(["elu", "relu", "gelu"]), # gelu
}

# Wrap the model with BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=tft_model_builder(
        look_back=look_back,
        num_heads=None,
        head_dim=None,
        feed_forward_dim=None,
        dropout_rate=None,
        learning_rate=None,
        num_layers=None,
        activation=None,
        n_steps_ahead=1
    ),
    search_spaces=search_space,
    n_iter=200,
    # cv=TimeSeriesSplit(n_splits=5),
    cv=SKTimeToSKLearnCV(expanding_splitter, y_train),
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
    # random_state=42,
    # return_train_score=True
)

# Fit the search
bayes_search.fit(X_train, y_train)


best_model = bayes_search.best_estimator_
best_params = bayes_search.best_params_
best_score = -bayes_search.best_score_

# Save the best model
# Save the model's weights (state_dict)
torch.save(best_model.module_.state_dict(), '../models/tft_recursive_best_model_weights.pth')

# Save optimizer state (optional)
torch.save(best_model.optimizer_.state_dict(), '../models/tft_recursive_optimizer_state.pth')

with open("../models/tft_recursive_best_params.pkl", 'wb') as f:
    joblib.dump(best_params, f)

# Save the best score
with open('../models/recursive_best_score.txt', 'w') as f:
    f.write(f"Best Validation Score (MAE): {best_score}\n")



