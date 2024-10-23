import pandas as pd
import os
from ta import trend, momentum, volatility, volume

def read_csv(file_path):
    """
    Reads the CSV file into a pandas DataFrame.
    
    Parameters:
        file_path (str): The path to the original CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    try:
        # Parse the 'datetime' column as datetime objects
        df = pd.read_csv(
            file_path, 
            header=None, 
            names=[
                'datetime', 'open', 'high', 'low', 'close', 'volume',
                'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'Bollinger_High', 'Bollinger_Low', 'Bollinger_Middle',
                'OBV', 'ATR_14', 'Stochastic_%K', 'Stochastic_%D',
                'VWAP', 'SMA_3', 'SMA_10', 'EMA_3', 'EMA_10'
            ],
            parse_dates=['datetime']
        )
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        raise
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise

def filter_trading_hours(df, start_time_str="08:30:00", end_time_str="15:00:00"):
    """
    Filters the DataFrame to include only rows within the specified trading hours.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame.
        start_time_str (str): Start time in "HH:MM:SS" format.
        end_time_str (str): End time in "HH:MM:SS" format.
        
    Returns:
        pd.DataFrame: Filtered DataFrame within trading hours.
    """
    # Convert string times to time objects
    start_time = pd.to_datetime(start_time_str).time()
    end_time = pd.to_datetime(end_time_str).time()
    
    # Extract time from 'datetime' column
    df['time'] = df['datetime'].dt.time
    
    # Filter rows within the trading hours
    filtered_df = df[df['time'].between(start_time, end_time)].copy()
    
    # Drop the auxiliary 'time' column
    filtered_df.drop(columns=['time'], inplace=True)
    
    return filtered_df

def calculate_advanced_statistics(df):
    """
    Calculates advanced statistics and adds them as new columns to the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame filtered by trading hours.
        
    Returns:
        pd.DataFrame: DataFrame with additional statistical columns.
    """
    # Ensure the DataFrame is sorted by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # ----- Relative Strength Index (RSI) -----
    rsi_period = 14
    df['RSI_14'] = momentum.RSIIndicator(close=df['close'], window=rsi_period).rsi()
    
    # ----- Moving Average Convergence Divergence (MACD) -----
    macd = trend.MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # ----- Bollinger Bands -----
    bollinger = volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()
    
    # ----- On-Balance Volume (OBV) -----
    df['OBV'] = volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    
    # ----- Average True Range (ATR) -----
    atr_period = 14
    df['ATR_14'] = volatility.AverageTrueRange(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=atr_period
    ).average_true_range()
    
    # ----- Stochastic Oscillator -----
    stochastic = momentum.StochasticOscillator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=14, 
        smooth_window=3
    )
    df['Stochastic_%K'] = stochastic.stoch()
    df['Stochastic_%D'] = stochastic.stoch_signal()
    
    # ----- Volume Weighted Average Price (VWAP) -----
    vwap = volume.VolumeWeightedAveragePrice(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        volume=df['volume']
    )
    df['VWAP'] = vwap.volume_weighted_average_price()
    
    # ----- Simple Moving Averages (Optional) -----
    df['SMA_3'] = df['close'].rolling(window=3).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    
    # ----- Exponential Moving Averages (EMA) -----
    df['EMA_3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    
    return df

def create_new_csv(original_file_path, df, suffix="_advanced"):
    """
    Creates a new CSV file with the advanced statistics.
    
    Parameters:
        original_file_path (str): The path to the original CSV file.
        df (pd.DataFrame): The DataFrame with additional statistics.
        suffix (str): Suffix to append to the original file name.
        
    Returns:
        str: The path to the new CSV file.
    """
    base, ext = os.path.splitext(original_file_path)
    new_file_path = f"{base}{suffix}{ext}"
    
    try:
        df.to_csv(new_file_path, index=False)
        print(f"Advanced CSV created successfully at: {new_file_path}")
        return new_file_path
    except Exception as e:
        print(f"An error occurred while writing the new CSV: {e}")
        raise

def main():
    original_csv = 'D:\\codyb\\COMP6970_Final_Project_Data\\TSLA_minute_data.csv'
    
    # Step 1: Read the original CSV
    df = read_csv(original_csv)
    
    # Step 2: Filter data to include only trading hours (8:30 AM to 3:00 PM)
    df_filtered = filter_trading_hours(df, start_time_str="08:30:00", end_time_str="15:00:00")
    
    # Step 3: Calculate advanced statistics on the filtered data
    df_advanced = calculate_advanced_statistics(df_filtered)
    
    # Step 4: Create the new CSV with advanced statistics
    create_new_csv(original_csv, df_advanced)

if __name__ == "__main__":
    main()
