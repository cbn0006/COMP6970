import pandas as pd
import os
import mplfinance as mpf
from datetime import datetime, timedelta

DATA_DIR = "labeling"

def load_data(filename):
    """
    Load the stock data from a CSV file with headers.
    """
    file_path = os.path.join(DATA_DIR, filename)
    
    # Load the CSV file without specifying column names
    dtype_spec = {
        'datetime': str,
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': float
    }
    df = pd.read_csv(file_path, dtype=dtype_spec)
    
    # Convert the timestamp column to datetime with error handling
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # Drop rows with invalid timestamp parsing
    df.dropna(subset=['datetime'], inplace=True)
    return df

def filter_data_by_time(df, start_time_obj, interval_minutes):
    """
    Filter the dataframe for a specific time range (start time and interval in minutes).
    """
    end_time_obj = start_time_obj + timedelta(minutes=interval_minutes)
    
    # Filter the data to include only rows within the time range
    filtered_df = df[(df['datetime'] >= start_time_obj) & (df['datetime'] < end_time_obj)]
    
    return filtered_df

def plot_candlestick(df, title, save_path):
    """
    Create and save a candlestick chart without the volume sub-plot using mplfinance.
    """
    if 'datetime' not in df.columns:
        raise KeyError("'datetime' column is missing from the DataFrame.")
    
    # Set the index to the datetime for mplfinance
    df.set_index('datetime', inplace=True)
    
    # Save the candlestick chart to a file without volume
    mpf.plot(df, type='candle', title=title, savefig=save_path)
    print(f"Candlestick chart saved to {save_path}")

def generate_20_minute_charts(stock_data, start_date, end_date):
    """
    Generate fixed 20-minute candlestick charts (e.g., 8:30-8:49, 8:50-9:09) between the given start and end dates.
    All images are saved in a single directory without separate folders for each day.
    """
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Define the main directory to save all charts
    SAVE_DIR = os.path.join("D:\\codyb\\COMP6970_Final_Project_Data\\", "charts", "20_minute_intervals")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Define the trading hours for the day
        start_time = datetime.strptime(f"{date_str} 08:30", '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(f"{date_str} 15:00", '%Y-%m-%d %H:%M')
        
        current_time = start_time
        
        # Generate images for every 20-minute fixed interval
        while current_time + timedelta(minutes=20) <= end_time:
            # Filter data for the current 20-minute interval
            filtered_data = filter_data_by_time(stock_data, current_time, 20)
            
            if not filtered_data.empty:
                ohlc_df = filtered_data[['datetime', 'open', 'high', 'low', 'close']].copy()
                
                # Determine the bullish/bearish label from the last row in the interval
                label = filtered_data['Bullish/Bearish'].iloc[-1]
                
                # Define the save path with the label in the filename, saved directly in the main directory
                time_str = current_time.strftime('%Y%m%d_%H%M')
                save_path = os.path.join(SAVE_DIR, f"TSLA_Candlestick_{time_str}_to_{(current_time + timedelta(minutes=20)).strftime('%H%M')}_{label}.png")
                
                # Plot and save the candlestick chart
                plot_candlestick(ohlc_df, f"TSLA Candlestick Chart for {date_str} {time_str} - 20 mins - {label}", save_path)
            
            # Move to the next 20-minute interval
            current_time += timedelta(minutes=20)
        
        # Move to the next day
        current_date += timedelta(days=1)

def main():
    # Load the data from the CSV
    stock_data = load_data("TSLA_minute_data_cleaned_labeled.csv")
    
    # Define the date range
    start_date = "2023-01-01"
    end_date = "2024-02-01"
    
    # Generate overlapping 20-minute candlestick charts between the given dates
    generate_20_minute_charts(stock_data, start_date, end_date)

if __name__ == "__main__":
    main()
