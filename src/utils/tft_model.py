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
from sklearn.model_selection import BaseCrossValidator

device = torch.device("cpu")

class TqdmProgressBar(Callback):
    def on_train_begin(self, net, **kwargs):
        self.epochs = net.max_epochs
        self.pbar = tqdm(total=self.epochs, desc="Training Progress", unit="epoch")

    def on_epoch_end(self, net, **kwargs):
        self.pbar.update(1)  # Increment the progress bar by one epoch

    def on_train_end(self, net, **kwargs):
        self.pbar.close()

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_dim):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1), :]  # Add positional encoding to the input
    
# Temporal Fusion Transformer architecture
class TemporalFusionTransformer(nn.Module):
    def __init__(self, look_back, num_heads, hidden_dim, feed_forward_dim, dropout_rate,
                 num_layers=1, activation="relu", input_feature_dim=1, n_steps_ahead=1):
        super(TemporalFusionTransformer, self).__init__()
        self.look_back = look_back
        self.hidden_dim = hidden_dim
        self.n_steps_ahead = n_steps_ahead
        
        self.input_projection = nn.Linear(input_feature_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(max_len=look_back, hidden_dim=hidden_dim)
        
        # Define the activation function dynamically
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        elif activation == "elu":
            self.activation_fn = nn.ELU(alpha=1.0)  # Add support for ELU
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Create a stack of transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                "multi_head_attention": nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True),
                "layer_norm1": nn.LayerNorm(hidden_dim),
                "feed_forward": nn.Sequential(
                    nn.Linear(hidden_dim, feed_forward_dim),
                    self.activation_fn,
                    nn.Dropout(dropout_rate),
                    nn.Linear(feed_forward_dim, hidden_dim)
                ),
                "layer_norm2": nn.LayerNorm(hidden_dim)
            })
            for _ in range(num_layers)
        ])
        
        # Predict n_steps_ahead for each input sequence
        self.output_layer = nn.Linear(hidden_dim, n_steps_ahead)

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        position_encoding = self.positional_encoding(x)
        x = x + position_encoding
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            # Apply multi-head attention
            attn_output, _ = layer["multi_head_attention"](x, x, x)
            x = layer["layer_norm1"](x + attn_output)
            
            # Apply feed-forward network
            ff_output = layer["feed_forward"](x)
            x = layer["layer_norm2"](x + ff_output)

        # Focus on the last time step for forecasting future steps
        # Shape before convert: (batch_size, look_back, n_steps_ahead)
        # Shape with after convert: (batch_size, n_steps_ahead)
        x = x[:, -1, :]  
        
        # Pass the entire sequence through the output layer
        outputs = self.output_layer(x)  
        return outputs
    
class TemporalFusionTransformerWrapper(nn.Module):
    def __init__(self, look_back, num_heads, head_dim, feed_forward_dim, dropout_rate, num_layers, activation, n_steps_ahead):
        super(TemporalFusionTransformerWrapper, self).__init__()
        self.model = TemporalFusionTransformer(
            look_back=look_back,
            num_heads=num_heads,
            hidden_dim=num_heads * head_dim,
            feed_forward_dim=feed_forward_dim,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            activation=activation,
            input_feature_dim=1,
            n_steps_ahead= n_steps_ahead
        )

    def forward(self, x):
        x = x.view(x.size(0), self.model.look_back, -1)
        return self.model(x)

def tft_model_builder(look_back, num_heads, head_dim, feed_forward_dim, dropout_rate, learning_rate, num_layers, activation, n_steps_ahead):
    return NeuralNetRegressor(
        module=TemporalFusionTransformerWrapper,
        module__look_back=look_back,
        module__num_heads=num_heads,
        module__head_dim=head_dim,
        module__feed_forward_dim=feed_forward_dim,
        module__dropout_rate=dropout_rate,
        module__num_layers=num_layers,
        module__activation=activation,
        module__n_steps_ahead=n_steps_ahead,
        max_epochs=20,
        lr=learning_rate,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss,
        batch_size=32,
        train_split=None,
        device='cpu',
        verbose=0,
        callbacks=[
            TqdmProgressBar()  # Add progress bar callback here
        ]
    )

# Custom Wrapper for ExpandingWindowSplitter
class SKTimeToSKLearnCV(BaseCrossValidator):
    def __init__(self, sktime_splitter, y):
        self.sktime_splitter = sktime_splitter
        self.y = y

    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in self.sktime_splitter.split(self.y):
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.sktime_splitter.get_n_splits(self.y)