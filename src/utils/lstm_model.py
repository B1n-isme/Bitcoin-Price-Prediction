from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Layer
from tensorflow.keras import activations, backend as K
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias", shape=(1,), initializer="zeros", trainable=True
        )
        super(Attention, self).build(input_shape)

    @tf.function(reduce_retracing=True)
    def call(self, x):
        score = activations.tanh(K.dot(x, self.W) + self.b)
        attention_weights = activations.softmax(score / 0.1, axis=1)
        context_vector = x * attention_weights
        return tf.reduce_sum(context_vector, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_lstm(input_shape, units=100, dropout=0.2):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, activation='relu', return_sequences=True),
        Dropout(dropout),
        LSTM(units//2, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    return model

def build_bilstm(input_shape, units=100, dropout=0.2):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(units, activation='relu', return_sequences=True)),
        Dropout(dropout),
        LSTM(units // 2, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    return model

def build_attention_lstm(input_shape, units=100, dropout=0.2):
    inputs = Input(shape=input_shape)
    x = LSTM(units, activation='relu', return_sequences=True)(inputs)
    x = Dropout(dropout)(x)
    x = LSTM(units // 2, activation='relu', return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = Attention()(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

def build_attention_bilstm(input_shape, units=100, dropout=0.25):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(units, activation='relu', return_sequences=True))(inputs)
    x = Dropout(dropout)(x)
    x = LSTM(units // 2, activation='relu', return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = Attention()(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

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

