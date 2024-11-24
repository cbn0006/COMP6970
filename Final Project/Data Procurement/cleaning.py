import os
import pandas as pd
from datetime import datetime, timedelta

'''
Script that takes in the raw stock data pulled from Polygon and cleans it to be in desired time window.
'''

def filter_trading_hours(input_file, start_time_str="08:10:00", end_time_str="15:00:00"):
    try:
        df = pd.read_csv(
            input_file, 
            header=None, 
            skiprows=1,
            names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            dtype={
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            },
            parse_dates=['datetime'],
            date_format="%Y-%m-%d %H:%M:%S"
        )

        df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

        df.dropna(subset=['datetime'], inplace=True)

        start_time = pd.to_datetime(start_time_str).time()
        end_time = pd.to_datetime(end_time_str).time()

        df['time'] = df['datetime'].dt.time

        filtered_df = df[df['time'].between(start_time, end_time)].copy()

        filtered_df.drop(columns=['time'], inplace=True)

        output_file = input_file.replace("_raw", "_cleaned")

        filtered_df.to_csv(output_file, index=False)
        print(f"Filtered data saved to {output_file}")

        return output_file

    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
        raise
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        raise

if __name__ == "__main__":
    SAVE_DIR = "../labeling"
    symbol = "AAPL"
    input_filename = os.path.join(SAVE_DIR, f"{symbol}_minute_data_raw.csv")

    print(f"Filtering trading hours in {input_filename}...")
    output_filename = filter_trading_hours(input_filename)
    print(f"Filtered file created: {output_filename}")
