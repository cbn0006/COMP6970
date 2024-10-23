import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime

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

def filter_data_by_date(df, date):
    """
    Filter the dataframe for the specified date between 9:00 AM and 4:00 PM.
    """
    start_time = datetime.strptime(date + " 08:30:00", '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(date + " 15:00:00", '%Y-%m-%d %H:%M:%S')
    
    # Filter the data to only include rows within the time range for the specified date
    filtered_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    
    return filtered_df

def plot_candlestick(df, title):
    """
    Create and display a candlestick chart using mplfinance.
    """
    # Ensure 'timestamp' is the index for mplfinance
    if 'timestamp' not in df.columns:
        raise KeyError("'timestamp' column is missing from the DataFrame.")
    
    # Set the index to the timestamp for mplfinance
    df.set_index('timestamp', inplace=True)
    
    # Plot the candlestick chart
    mpf.plot(df, type='candle', volume=True, title=title)

def main():
    # Load the data from the CSV
    stock_data = load_data("AAPL_minute_data.csv")
    
    # Ask the user for the date to plot
    date = input("Enter the date (YYYY-MM-DD) to generate the candlestick chart: ")
    
    # Filter the data for that date and between 9:00 AM and 4:00 PM
    filtered_data = filter_data_by_date(stock_data, date)
    
    if filtered_data.empty:
        print(f"No data found for {date} between 9:00 AM and 4:00 PM.")
    else:
        # Check if 'timestamp' column exists in filtered data
        if 'timestamp' not in filtered_data.columns:
            print("'timestamp' column is missing in the filtered data.")
            return
        
        # Prepare the data in the OHLCV format (Open, High, Low, Close, Volume)
        ohlcv_df = filtered_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Plot the candlestick chart
        plot_candlestick(ohlcv_df, f"AAPL Candlestick Chart for {date} (9:00 AM - 4:00 PM)")

if __name__ == "__main__":
    main()
