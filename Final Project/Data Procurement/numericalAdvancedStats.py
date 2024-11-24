import pandas as pd
import os
from ta import trend, momentum, volatility, volume

'''
Script that takes in a cleaned csv of stock data and labels its advanced statistics like EMA, SMA, Bollinger Bands, and more.
'''

def read_csv(file_path):
    try:
        column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        dtype_spec = {
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64'
        }
        
        df = pd.read_csv(
            file_path, 
            header=0,
            names=column_names,
            dtype=dtype_spec,
            low_memory=False,
            na_values=['', ' ', 'NA', 'N/A', None]
        )
        
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        if df['datetime'].isnull().any():
            num_invalid = df['datetime'].isnull().sum()
            print(f"Warning: {num_invalid} rows have invalid 'datetime' format and will be dropped.")
            df = df.dropna(subset=['datetime'])
        
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        raise
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise

def calculate_advanced_statistics(df):
    df['time'] = df['datetime'].dt.time
    start_time = pd.to_datetime("08:10:00").time()
    end_time = pd.to_datetime("15:00:00").time()
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    
    df['date'] = df['datetime'].dt.date
    
    results = []
    for _, daily_data in df.groupby('date'):
        daily_data = daily_data.sort_values('datetime').reset_index(drop=True)
        
        # ----- Relative Strength Index (RSI) -----
        rsi_period = 20
        daily_data['RSI_20'] = momentum.RSIIndicator(close=daily_data['close'], window=rsi_period).rsi()
        
        # ----- Bollinger Bands -----
        bollinger = volatility.BollingerBands(close=daily_data['close'], window=20, window_dev=2)
        daily_data['Bollinger_High'] = bollinger.bollinger_hband()
        daily_data['Bollinger_Low'] = bollinger.bollinger_lband()
        daily_data['Bollinger_Middle'] = bollinger.bollinger_mavg()
        
        # ----- On-Balance Volume (OBV) -----
        daily_data['OBV'] = volume.OnBalanceVolumeIndicator(
            close=daily_data['close'], 
            volume=daily_data['volume']
        ).on_balance_volume()
        
        # ----- Average True Range (ATR) -----
        atr_period = 20
        daily_data['ATR_20'] = volatility.AverageTrueRange(
            high=daily_data['high'], 
            low=daily_data['low'], 
            close=daily_data['close'], 
            window=atr_period
        ).average_true_range()
        
        # ----- Stochastic Oscillator -----
        stochastic = momentum.StochasticOscillator(
            high=daily_data['high'], 
            low=daily_data['low'], 
            close=daily_data['close'], 
            window=20, 
            smooth_window=3
        )
        daily_data['Stochastic_%K'] = stochastic.stoch()
        daily_data['Stochastic_%D'] = stochastic.stoch_signal()
        
        # ----- Volume Weighted Average Price (VWAP) -----
        vwap = volume.VolumeWeightedAveragePrice(
            high=daily_data['high'], 
            low=daily_data['low'], 
            close=daily_data['close'], 
            volume=daily_data['volume']
        )
        daily_data['VWAP'] = vwap.volume_weighted_average_price()
        
        # ----- Simple Moving Averages (SMA) -----
        daily_data['SMA_7'] = daily_data['close'].rolling(window=7).mean()
        daily_data['SMA_20'] = daily_data['close'].rolling(window=20).mean()
        
        # ----- Exponential Moving Averages (EMA) -----
        daily_data['EMA_7'] = daily_data['close'].ewm(span=7, adjust=False).mean()
        daily_data['EMA_20'] = daily_data['close'].ewm(span=20, adjust=False).mean()
        
        results.append(daily_data)

    df_processed = pd.concat(results, axis=0)
    
    df_processed = df_processed.drop(columns=['time', 'date'])
    
    return df_processed

def create_new_csv(original_file_path, df, suffix="_advanced"):
    base_name = os.path.basename(original_file_path)
    name, ext = os.path.splitext(base_name)
    
    new_file_name = f"{name}{suffix}{ext}"
    
    output_dir = "../labeling"
    
    os.makedirs(output_dir, exist_ok=True)
    
    new_file_path = os.path.join(output_dir, new_file_name)
    
    try:
        df.to_csv(new_file_path, index=False)
        print(f"Advanced CSV created successfully at: {new_file_path}")
        return new_file_path
    except Exception as e:
        print(f"An error occurred while writing the new CSV: {e}")
        raise

def main():
    original_csv = '../labeling/AAPL_minute_data_cleaned.csv'
    
    # Step 1: Read the original CSV
    df = read_csv(original_csv)
    
    # Step 2: Calculate advanced statistics on the data
    df_advanced = calculate_advanced_statistics(df)
    
    # Step 3: Create the new CSV with advanced statistics
    create_new_csv(original_csv, df_advanced)

if __name__ == "__main__":
    main()