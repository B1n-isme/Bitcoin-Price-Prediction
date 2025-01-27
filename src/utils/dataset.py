import os
import time
from datetime import datetime, timedelta
from functools import reduce
import pandas as pd
import requests
import yfinance as yf
from pytrends.request import TrendReq
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from statsmodels.tsa.arima.model import ARIMA


pd.set_option('future.no_silent_downcasting', True)

# Constants
BASE_URL = 'https://api.blockchain.info/charts/'
BLOCKCHAIN_ENDPOINTS = {
    'hash_rate': 'hash-rate',
    'active_addresses': 'n-unique-addresses',
    'miner_revenue': 'miners-revenue'
}
FINAL_OUTPUT = "../../data/final/new_dataset.csv"

# Helper Functions

def generate_intervals(start_date, days=30):
    """Generate date intervals of a fixed number of days."""
    end_date = datetime.now().date()
    while start_date < end_date:
        interval_end = min(start_date + timedelta(days=days - 1), end_date)
        yield start_date, interval_end
        start_date = interval_end + timedelta(days=1)

# Helper function for ARIMA-based imputation
def arima_impute(series):
    if series.isnull().sum() > 0 and series.notnull().sum() > 5:  # Ensure enough valid points
        model = ARIMA(series, order=(1, 1, 1))  # Adjust (p, d, q) as needed
        fitted_model = model.fit()
        series = series.fillna(fitted_model.fittedvalues)
    return series

def fetch_blockchain_data(start_date):
    """Fetch blockchain metrics."""
    all_data = {metric: [] for metric in BLOCKCHAIN_ENDPOINTS.keys()}
    for metric, endpoint in BLOCKCHAIN_ENDPOINTS.items():
        for start, end in generate_intervals(start_date):
            params = {'start': start.isoformat(), 'end': end.isoformat(), 'format': 'json'}
            url = f"{BASE_URL}{endpoint}"
            response = requests.get(url, params=params)
            if response.status_code == 200:
                all_data[metric].extend(response.json().get('values', []))
            else:
                print(f"Failed to fetch data for {metric} from {start} to {end}.")

    combined_df = pd.DataFrame()
    for metric, data in all_data.items():
        metric_df = pd.DataFrame(data)
        metric_df['metric'] = metric
        combined_df = pd.concat([combined_df, metric_df], ignore_index=True)

    combined_df['x'] = pd.to_datetime(combined_df['x'], unit='s', utc=True)

    # Convert 'x' from Unix timestamp to datetime in UTC
    combined_df['Date'] = pd.to_datetime(combined_df['x'], unit='s', utc=True)

    # Calculate days since the first record
    start_date = combined_df['Date'].min()
    combined_df['days_since_start'] = (combined_df['Date'] - start_date).dt.days

    # Aggregate duplicates by taking the mean of 'y' and keeping the first 'date'
    combined_df = combined_df.groupby(['days_since_start', 'metric'], as_index=False).agg({'y': 'mean', 'Date': 'first'})

    # Pivot the dataset to spread metrics into separate columns
    pivot_df = combined_df.pivot(index='days_since_start', columns='metric', values='y').reset_index()

    # Generate a complete range of days and merge with the pivoted DataFrame
    all_days = pd.DataFrame({'days_since_start': range(combined_df['days_since_start'].max() + 1)})
    pivot_df = all_days.merge(pivot_df, on='days_since_start', how='left')

    # Add the original datetime back to the pivoted DataFrame
    pivot_df['Date'] = start_date + pd.to_timedelta(pivot_df['days_since_start'], unit='d')

    # Drop the 'days_since_start' column and reorder columns to make 'date' first
    pivot_df = pivot_df.drop(columns=['days_since_start']).set_index('Date').reset_index()

    # Reorder columns for consistency
    pivot_df = pivot_df[['Date', 'active_addresses', 'hash_rate', 'miner_revenue']]

    # Optional: Rename columns for clarity
    pivot_df.rename(columns={'active_addresses': 'active_addresses_blockchain', 
                            'hash_rate': 'hash_rate_blockchain', 
                            'miner_revenue': 'miner_revenue_blockchain'}, inplace=True)

    return pivot_df

def fetch_google_trends_data(keyword, start_date):
    """Fetch Google Trends data."""
    pytrends = TrendReq(hl='en-US', tz=360)
    intervals = generate_intervals(start_date)
    all_data = pd.DataFrame()

    for start, end in intervals:
        pytrends.build_payload([keyword], timeframe=f"{start} {end}", geo='')
        data = pytrends.interest_over_time()
        if not data.empty:
            data = data.drop(columns=['isPartial'], errors='ignore')
            all_data = pd.concat([all_data, data])

    all_data.index.name = 'Date'
    all_data.reset_index(inplace=True)
    all_data = all_data.rename(columns={'bitcoin': 'google_trends_bitcoin'})
    return all_data

def fetch_yfinance_data(tickers, start_date):
    """Fetch data from Yahoo Finance."""
    combined_data = pd.DataFrame()
    for ticker, description in tickers.items():
        data = yf.download(ticker, start=start_date, end=datetime.now().date().isoformat())
        data.columns = [f'{description}', 'High', 'Low', 'Open', 'Volume']
        if not data.empty:
            data = data[[f'{description}']]
            combined_data = pd.concat([combined_data, data], axis=1, join='outer')
    
    # Step 1: Create a full date range for the combined data
    full_date_range = pd.date_range(start=combined_data.index.min(), end=combined_data.index.max())
    combined_data = combined_data.reindex(full_date_range)

    # Step 2: Forward Fill for Edge Cases
    combined_data = combined_data.fillna(method="ffill")

    # Step 3: Linear Interpolation for Small Gaps
    combined_data = combined_data.interpolate(method="linear")

    # Step 4: ARIMA-Based Imputation for Large Gaps
    for column in combined_data.columns:
        combined_data[column] = arima_impute(combined_data[column])

    # Step 5: Final Fallback for Remaining Missing Values
    combined_data = combined_data.fillna(combined_data.mean())  # Fallback strategy

    combined_data.index.name = "Date"

    combined_data.reset_index(inplace=True)

    return combined_data

def calculate_technical_indicators(start_date):
    """Fetch historical BTC data and calculate technical indicators."""
    btc_data = yf.download("BTC-USD", start=start_date, end=datetime.now().date().isoformat())
    btc_data = btc_data.reset_index()
    btc_data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    btc_data.set_index('Date', inplace=True)

    data = pd.DataFrame()
    data['Date'] = btc_data.index
    data['btc_close'] = btc_data['Close'].values
    data['btc_sma_14'] = SMAIndicator(close=btc_data['Close'], window=14).sma_indicator().values
    data['btc_ema_14'] = EMAIndicator(close=btc_data['Close'], window=14).ema_indicator().values
    data['btc_rsi_14'] = RSIIndicator(close=btc_data['Close'], window=14).rsi().values
    macd = MACD(close=btc_data['Close'])
    data['btc_macd'] = macd.macd().values
    data['btc_macd_signal'] = macd.macd_signal().values
    data['btc_macd_diff'] = macd.macd_diff().values
    bb = BollingerBands(close=btc_data['Close'], window=20, window_dev=2)
    data['btc_bb_high'] = bb.bollinger_hband().values
    data['btc_bb_low'] = bb.bollinger_lband().values
    data['btc_bb_mid'] = bb.bollinger_mavg().values
    data['btc_bb_width'] = bb.bollinger_wband().values
    data['btc_atr_14'] = AverageTrueRange(high=btc_data['High'], low=btc_data['Low'], close=btc_data['Close'], window=14).average_true_range().values
    data['btc_trading_volume'] = btc_data['Volume'].values
    data['btc_volatility_index'] = (btc_data['High'] - btc_data['Low']).values
    return data

# merge all data
def fetch_new_data(start_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    """Merge data from all sources."""
    google_trends = fetch_google_trends_data("bitcoin", start_date)
    blockchain = fetch_blockchain_data(start_date)
    tickers = {
        "CL=F": "Oil_Close",
        "GC=F": "Gold_Close",
        "^DJI": "DJI",
        "^GSPC": "GSPC",
        "^IXIC": "IXIC",
        "^NYFANG": "NYSE FANG+",
        "ARKK": "ARK Innovation ETF",
        "^VIX": "CBOE Volatility Index",
        "EEM": "iShares MSCI Emerging Markets ETF",
        "000001.SS": "Shanghai Composite Index",
        "DX-Y.NYB": "USD Index (DXY)",
        "EURUSD=X": "EUR to USD Exchange Rate"

    }
    yfinance_data = fetch_yfinance_data(tickers, start_date)
    technical_indicators = calculate_technical_indicators(start_date)

    # Merge all datasets on Date
    all_data = [google_trends, blockchain, yfinance_data, technical_indicators]

    # Ensure all 'Date' columns are timezone-aware in UTC
    for df in all_data:
        df['Date'] = pd.to_datetime(df['Date'])
        if df['Date'].dt.tz is not None:
            # If already timezone-aware, convert to UTC
            df['Date'] = df['Date'].dt.tz_convert('UTC')
        else:
            # If timezone-naive, localize to UTC
            df['Date'] = df['Date'].dt.tz_localize('UTC')

    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), all_data)
    btc_close = merged_df.pop('btc_close')  
    merged_df['btc_close'] = btc_close    
    # Remove the first 35 rows
    merged_df = merged_df.iloc[35:]
    # Impute missing values using forward fill method
    merged_df.ffill(inplace=True)
    # Impute remaining missing values using backward fill method
    merged_df.bfill(inplace=True) 
    return merged_df

# Main Execution
if __name__ == "__main__":
    # start_date_str = input("Enter the start date (YYYY-MM-DD): ")
    start_date = "2024-10-15"
    

    tickers = {
        "^GSPC": "GSPC",
        "^IXIC": "IXIC",
        "^DJI": "DJI",
        "^NYFANG": "NYSE FANG+",
        "ARKK": "ARK Innovation ETF",
        "^VIX": "CBOE Volatility Index",
        "EEM": "iShares MSCI Emerging Markets ETF",
        "000001.SS": "Shanghai Composite Index",
        "DX-Y.NYB": "USD Index (DXY)",
        "EURUSD=X": "EUR to USD Exchange Rate",
        "CL=F": "Oil_Close",
        "GC=F": "Gold_Close"
    }


    # Collect and merge data
    final_data = fetch_new_data(start_date)

    # Save to CSV
    os.makedirs(os.path.dirname(FINAL_OUTPUT), exist_ok=True)
    final_data.to_csv(FINAL_OUTPUT, index=False)
    print(f"Data collection and merging complete. Saved to {FINAL_OUTPUT}.")
