import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from utils.lstm_model import build_lstm, build_bilstm, build_attention_lstm, build_attention_bilstm

def lstm_model_builder(model_type, input_shape, units, dropout, learning_rate):
    if model_type == "LSTM":
        model = build_lstm(input_shape=input_shape, units=units, dropout=dropout)
    elif model_type == "BiLSTM":
        model = build_bilstm(input_shape=input_shape, units=units, dropout=dropout)
    elif model_type == "Attention-LSTM":
        model = build_attention_lstm(input_shape=input_shape, units=units, dropout=dropout)
    elif model_type == "Attention-BiLSTM":
        model = build_attention_bilstm(input_shape=input_shape, units=units, dropout=dropout)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

def tune_hyperparameters(X_train, y_train, param_grid, model_types, input_shape, cv_splits=5):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    best_params = {}
    best_models = {}

    for model_type in model_types:
        print(f"Tuning {model_type}...")
        model_wrapper = KerasRegressor(
            model=lstm_model_builder,
            model_type=model_type,
            input_shape=input_shape,
            units=100,
            dropout=0.2,
            learning_rate=0.001,
            epochs=20,
            verbose=0,
            validation_split=0.2
        )
        grid_search = GridSearchCV(
            estimator=model_wrapper,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=tscv
        )
        grid_search.fit(X_train, y_train)
        best_params[model_type] = grid_search.best_params_
        best_models[model_type] = grid_search.best_estimator_
    return best_params, best_models
