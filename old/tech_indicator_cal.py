import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def calculate_technical_indicators(input_data):
    """
    Calculate technical indicators for a single date's BTC price data.
    
    Parameters:
        input_data (pd.DataFrame): A DataFrame with the following required columns:
            ['Close', 'High', 'Low', 'Volume'].
    
    Returns:
        dict: A dictionary of technical indicators for the given input data.
    """
    # Ensure the input_data is a DataFrame with a single row
    if not isinstance(input_data, pd.DataFrame) or len(input_data) != 1:
        raise ValueError("input_data must be a DataFrame with exactly one row.")
    
    # Close prices (required for most indicators)
    close = input_data['Close']
    
    # Technical Indicators
    indicators = {}
    
    # Simple Moving Average (14-day)
    sma_indicator = SMAIndicator(close=close, window=14)
    indicators['btc_sma_14'] = sma_indicator.sma_indicator().iloc[-1]
    
    # Exponential Moving Average (14-day)
    ema_indicator = EMAIndicator(close=close, window=14)
    indicators['btc_ema_14'] = ema_indicator.ema_indicator().iloc[-1]
    
    # Relative Strength Index (14-day)
    rsi_indicator = RSIIndicator(close=close, window=14)
    indicators['btc_rsi_14'] = rsi_indicator.rsi().iloc[-1]
    
    # MACD
    macd = MACD(close=close)
    indicators['btc_macd'] = macd.macd().iloc[-1]
    indicators['btc_macd_signal'] = macd.macd_signal().iloc[-1]
    indicators['btc_macd_diff'] = macd.macd_diff().iloc[-1]
    
    # Bollinger Bands (20-day, 2 std deviations)
    bb = BollingerBands(close=close, window=20, window_dev=2)
    indicators['btc_bb_high'] = bb.bollinger_hband().iloc[-1]
    indicators['btc_bb_low'] = bb.bollinger_lband().iloc[-1]
    indicators['btc_bb_mid'] = bb.bollinger_mavg().iloc[-1]
    indicators['btc_bb_width'] = bb.bollinger_wband().iloc[-1]
    
    # Average True Range (14-day)
    atr = AverageTrueRange(high=input_data['High'], low=input_data['Low'], close=close, window=14)
    indicators['btc_atr_14'] = atr.average_true_range().iloc[-1]
    
    # Trading Volume (direct from input)
    indicators['btc_trading_volume'] = input_data['Volume'].iloc[0]
    
    # Volatility Index (High - Low)
    indicators['btc_volatility_index'] = (input_data['High'] - input_data['Low']).iloc[0]
    
    return indicators
