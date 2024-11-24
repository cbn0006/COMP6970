import pandas as pd
import os
import mplfinance as mpf
from datetime import datetime, timedelta

'''
Script that creates 20 minute candlestick charts for Paper 1's implementation.
'''

DATA_DIR = "../labeling"

def load_data(filename):
    file_path = os.path.join(DATA_DIR, filename)
    
    dtype_spec = {
        'datetime': str,
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': float
    }
    df = pd.read_csv(file_path, dtype=dtype_spec)
    
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    df.dropna(subset=['datetime'], inplace=True)
    return df

def filter_data_by_time(df, start_time_obj, interval_minutes):
    end_time_obj = start_time_obj + timedelta(minutes=interval_minutes)

    filtered_df = df[(df['datetime'] >= start_time_obj) & (df['datetime'] < end_time_obj)]
    
    return filtered_df

def plot_candlestick(df, title, save_path):
    if 'datetime' not in df.columns:
        raise KeyError("'datetime' column is missing from the DataFrame.")
    
    df.set_index('datetime', inplace=True)
    
    mpf.plot(df, type='candle', title=title, savefig=save_path)
    print(f"Candlestick chart saved to {save_path}")

def generate_20_minute_charts(stock_data, start_date, end_date, symbol):
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    SAVE_DIR = os.path.join("../labeling/", "20_minute_intervals")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        start_time = datetime.strptime(f"{date_str} 08:30", '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(f"{date_str} 15:00", '%Y-%m-%d %H:%M')
        
        current_time = start_time

        while current_time + timedelta(minutes=20) <= end_time:
            filtered_data = filter_data_by_time(stock_data, current_time, 20)
            
            if not filtered_data.empty:
                ohlc_df = filtered_data[['datetime', 'open', 'high', 'low', 'close']].copy()
                
                label = filtered_data['Bullish/Bearish'].iloc[-1]

                time_str = current_time.strftime('%Y%m%d_%H%M')
                save_path = os.path.join(SAVE_DIR, f"{symbol}_Candlestick_{time_str}_to_{(current_time + timedelta(minutes=20)).strftime('%H%M')}_{label}.png")

                plot_candlestick(ohlc_df, f"{symbol} Candlestick Chart for {date_str} {time_str} - 20 mins - {label}", save_path)

            current_time += timedelta(minutes=20)

        current_date += timedelta(days=1)

def main():
    symbol = 'AAPL'
    # Load the data from the CSV
    stock_data = load_data("AAPL_minute_data_cleaned_labeled.csv")
    
    # Define the date range
    start_date = "2023-01-01"
    end_date = "2023-02-01"
    
    # Generate 20-minute candlestick charts between the given dates
    generate_20_minute_charts(stock_data, start_date, end_date, symbol)

if __name__ == "__main__":
    main()
