import os
import pandas as pd
from datetime import datetime, timedelta

def filter_trading_hours(input_file, start_time_str="08:30:00", end_time_str="15:00:00"):
    try:
        # Read the input CSV file
        df = pd.read_csv(
            input_file, 
            header=None, 
            names=['datetime', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Explicitly parse 'datetime' column in the exact format of the data
        df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        
        # Drop any rows where datetime conversion failed
        df.dropna(subset=['datetime'], inplace=True)

        # Convert string times to time objects
        start_time = pd.to_datetime(start_time_str).time()
        end_time = pd.to_datetime(end_time_str).time()

        # Extract time from 'datetime' column
        df['time'] = df['datetime'].dt.time

        # Filter rows within the trading hours
        filtered_df = df[df['time'].between(start_time, end_time)].copy()

        # Drop the auxiliary 'time' column
        filtered_df.drop(columns=['time'], inplace=True)

        # Create output filename by replacing "_raw" with "_cleaned"
        output_file = input_file.replace("_raw", "_cleaned")

        # Save the filtered data to the new CSV file
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
    SAVE_DIR = "D:\\codyb\\COMP6970_Final_Project_Data"
    symbol = "TQQQ"
    input_filename = os.path.join(SAVE_DIR, f"{symbol}_minute_data_raw.csv")

    print(f"Filtering trading hours in {input_filename}...")
    output_filename = filter_trading_hours(input_filename)
    print(f"Filtered file created: {output_filename}")
