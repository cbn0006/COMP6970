import pandas as pd
import os
import mplfinance as mpf
from datetime import datetime, timedelta

DATA_DIR = "D:\\codyb\\COMP6970_Final_Project_Data"

def load_data(filename):
    """
    Load the stock data from a CSV file without headers.
    """
    file_path = f"{DATA_DIR}\\{filename}"
    
    # Load the CSV file and specify the column names
    df = pd.read_csv(file_path, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert the timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def filter_data_by_time(df, date, start_time, interval_minutes):
    """
    Filter the dataframe for a specific time range (start time and interval in minutes).
    """
    start_time_obj = datetime.strptime(f"{date} {start_time}:00", '%Y-%m-%d %H:%M:%S')
    end_time_obj = start_time_obj + timedelta(minutes=interval_minutes)
    
    # Filter the data to only include rows within the time range for the specified interval
    filtered_df = df[(df['timestamp'] >= start_time_obj) & (df['timestamp'] < end_time_obj)]
    
    return filtered_df

def plot_candlestick(df, title, save_path):
    """
    Create and save a candlestick chart without the volume sub-plot using mplfinance.
    """
    # Ensure 'timestamp' is the index for mplfinance
    if 'timestamp' not in df.columns:
        raise KeyError("'timestamp' column is missing from the DataFrame.")
    
    # Set the index to the timestamp for mplfinance
    df.set_index('timestamp', inplace=True)
    
    # Save the candlestick chart to a file without volume
    mpf.plot(df, type='candle', title=title, savefig=save_path)
    print(f"Candlestick chart saved to {save_path}")

def generate_hourly_charts(stock_data, start_date, end_date):
    """
    Generate hourly candlestick charts from 8:30 AM to 1:30 PM (1-hour intervals)
    and from 1:30 PM to 3:00 PM (1.5-hour interval) between the given start and end dates.
    """
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    SAVE_DIR = "D:\\codyb\\COMP6970_Final_Project_Data\\charts\\TSLA"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Loop through each day between the start and end dates
    while current_date <= end_date:
        # Generate charts for every 1-hour interval from 8:30 AM to 1:30 PM
        hourly_intervals = [('08:30', 60), ('09:30', 60), ('10:30', 60), ('11:30', 60), ('12:30', 60), ('13:30', 90)]
        
        for time_interval, interval_minutes in hourly_intervals:
            # Format the date and time for file naming and filtering
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Filter data for the current time interval
            filtered_data = filter_data_by_time(stock_data, date_str, time_interval, interval_minutes)
            
            if filtered_data.empty:
                print(f"No data found for {date_str} {time_interval}.")
                continue
            
            # Prepare the data in the OHLC format (Open, High, Low, Close)
            ohlc_df = filtered_data[['timestamp', 'open', 'high', 'low', 'close']].copy()
            
            # Define where to save the chart
            save_path = os.path.join(SAVE_DIR, f"TSLA_Candlestick_{date_str}_{time_interval.replace(':', '')}.png")
            
            # Plot and save the candlestick chart
            plot_candlestick(ohlc_df, f"TSLA Candlestick Chart for {date_str} {time_interval} - {interval_minutes} mins", save_path)
        
        # Move to the next day
        current_date += timedelta(days=1)

def main():
    # Load the data from the CSV
    stock_data = load_data("TSLA_minute_data.csv")
    
    # Define the date range
    start_date = "2023-01-01"
    end_date = "2023-04-01"
    
    # Generate hourly candlestick charts between the given dates
    generate_hourly_charts(stock_data, start_date, end_date)

if __name__ == "__main__":
    main()
