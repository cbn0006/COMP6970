import pandas as pd
import os

def read_csv(file_path):
    """
    Reads the advanced CSV file into a pandas DataFrame.
    
    Parameters:
        file_path (str): The path to the advanced CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the CSV data with headers.
    """
    try:
        # Read the CSV with headers
        df = pd.read_csv(
            file_path,
            header=0,  # Assuming the advanced.csv has headers
            parse_dates=['datetime']
        )
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        raise
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise

def is_doji(row, threshold=0.1):
    """
    Identifies a Doji candlestick pattern.
    
    Parameters:
        row (pd.Series): A row of the DataFrame.
        threshold (float): The maximum difference between open and close to be considered a Doji.
        
    Returns:
        bool: True if Doji pattern is identified, else False.
    """
    return abs(row['close'] - row['open']) <= threshold * (row['high'] - row['low'])

def is_hammer(row):
    """
    Identifies a Hammer candlestick pattern.
    
    Parameters:
        row (pd.Series): A row of the DataFrame.
        
    Returns:
        bool: True if Hammer pattern is identified, else False.
    """
    body = abs(row['close'] - row['open'])
    lower_shadow = row['open'] - row['low'] if row['close'] > row['open'] else row['close'] - row['low']
    return lower_shadow > 2 * body and row['close'] > row['open']

def is_shooting_star(row):
    """
    Identifies a Shooting Star candlestick pattern.
    
    Parameters:
        row (pd.Series): A row of the DataFrame.
        
    Returns:
        bool: True if Shooting Star pattern is identified, else False.
    """
    body = abs(row['close'] - row['open'])
    upper_shadow = row['high'] - row['close'] if row['close'] > row['open'] else row['high'] - row['open']
    return upper_shadow > 2 * body and row['close'] < row['open']

# There is strong and weak. Look into this
# Also, look into 5 minute, 10 minute downward patterns to engulfing candlestick.
def is_bullish_engulfing(prev_row, current_row):
    """
    Identifies a Bullish Engulfing candlestick pattern.
    
    Parameters:
        prev_row (pd.Series): The previous row of the DataFrame.
        current_row (pd.Series): The current row of the DataFrame.
        
    Returns:
        bool: True if Bullish Engulfing pattern is identified, else False.
    """
    return (prev_row['close'] < prev_row['open']) and \
           (current_row['close'] > current_row['open']) and \
           (current_row['open'] < prev_row['close']) and \
           (current_row['close'] > prev_row['open'])

def is_bearish_engulfing(prev_row, current_row):
    """
    Identifies a Bearish Engulfing candlestick pattern.
    
    Parameters:
        prev_row (pd.Series): The previous row of the DataFrame.
        current_row (pd.Series): The current row of the DataFrame.
        
    Returns:
        bool: True if Bearish Engulfing pattern is identified, else False.
    """
    return (prev_row['close'] > prev_row['open']) and \
           (current_row['close'] < current_row['open']) and \
           (current_row['open'] > prev_row['close']) and \
           (current_row['close'] < prev_row['open'])

def is_double_top(df, index, lookback=10, tolerance=0.005):
    """
    Identifies a Double Top chart pattern.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing stock data.
        index (int): The current index in the DataFrame.
        lookback (int): Number of periods to look back for the first top.
        tolerance (float): Price tolerance for the two tops to be considered the same.
        
    Returns:
        bool: True if Double Top pattern is identified, else False.
    """
    if index < lookback * 2:
        return False
    first_peak_idx = df['high'].iloc[index - lookback:index].idxmax()
    first_peak = df['high'].iloc[first_peak_idx]
    
    second_peak_window = df['high'].iloc[first_peak_idx + 1:index + 1]
    if second_peak_window.empty:
        return False
    second_peak_idx = second_peak_window.idxmax()
    second_peak = df['high'].iloc[second_peak_idx]
    
    return abs(first_peak - second_peak) / first_peak <= tolerance

def is_double_bottom(df, index, lookback=20, tolerance=0.005):
    """
    Identifies a Double Bottom chart pattern.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing stock data.
        index (int): The current index in the DataFrame.
        lookback (int): Number of periods to look back for the first trough.
        tolerance (float): Price tolerance for the two troughs to be considered the same.
        
    Returns:
        bool: True if Double Bottom pattern is identified, else False.
    """
    if index < lookback * 2:
        return False
    first_trough_idx = df['low'].iloc[index - lookback:index].idxmin()
    first_trough = df['low'].iloc[first_trough_idx]
    
    second_trough_window = df['low'].iloc[first_trough_idx + 1:index + 1]
    if second_trough_window.empty:
        return False
    second_trough_idx = second_trough_window.idxmin()
    second_trough = df['low'].iloc[second_trough_idx]
    
    # Check if the two troughs are within the tolerance
    return abs(first_trough - second_trough) / first_trough <= tolerance

def is_head_and_shoulders(df, index, lookback=30, tolerance=0.005):
    """
    Identifies a Head and Shoulders chart pattern.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing stock data.
        index (int): The current index in the DataFrame.
        lookback (int): Number of periods to look back for the pattern.
        tolerance (float): Price tolerance for the peaks.
        
    Returns:
        bool: True if Head and Shoulders pattern is identified, else False.
    """
    if index < lookback * 3:
        return False
    window = df.iloc[index - lookback:index + 1]
    peaks = window['high'].nlargest(3)
    if len(peaks) < 3:
        return False
    sorted_peaks = peaks.sort_values(ascending=False)
    head, shoulder1, shoulder2 = sorted_peaks.values
    # Check if shoulders are approximately equal within tolerance
    return abs(shoulder1 - shoulder2) / head <= tolerance

def is_inverse_head_and_shoulders(df, index, lookback=30, tolerance=0.005):
    """
    Identifies an Inverse Head and Shoulders chart pattern.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing stock data.
        index (int): The current index in the DataFrame.
        lookback (int): Number of periods to look back for the pattern.
        tolerance (float): Price tolerance for the troughs.
        
    Returns:
        bool: True if Inverse Head and Shoulders pattern is identified, else False.
    """
    if index < lookback * 3:
        return False
    window = df.iloc[index - lookback:index + 1]
    troughs = window['low'].nsmallest(3)
    if len(troughs) < 3:
        return False
    sorted_troughs = troughs.sort_values()
    head, shoulder1, shoulder2 = sorted_troughs.values
    # Check if shoulders are approximately equal within tolerance
    return abs(shoulder1 - shoulder2) / head <= tolerance

def label_patterns(df):
    """
    Labels each row in the DataFrame with identified technical analysis patterns.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing stock data.
        
    Returns:
        pd.DataFrame: DataFrame with additional 'Candlestick Pattern' and 'Chart Pattern' columns.
    """
    candlestick_patterns = []
    chart_patterns = []
    df = df.reset_index(drop=True)
    
    for i in range(len(df)):
        # Initialize patterns
        candlestick_pattern = 'None'
        chart_pattern = 'None'
        current_row = df.loc[i]
        
        # ----- Identify Candlestick Patterns -----
        # Doji
        if is_doji(current_row):
            candlestick_pattern = 'Doji'
        # Hammer
        elif is_hammer(current_row):
            candlestick_pattern = 'Hammer'
        # Shooting Star
        elif is_shooting_star(current_row):
            candlestick_pattern = 'Shooting Star'
        
        # Bullish/Bearish Engulfing Patterns
        if i > 0:
            prev_row = df.loc[i - 1]
            if is_bullish_engulfing(prev_row, current_row):
                candlestick_pattern = 'Bullish Engulfing'
            elif is_bearish_engulfing(prev_row, current_row):
                candlestick_pattern = 'Bearish Engulfing'
        
        candlestick_patterns.append(candlestick_pattern)
        
        # ----- Identify Chart Patterns -----
        # Double Top
        if is_double_top(df, i):
            chart_pattern = 'Double Top'
        # Double Bottom
        elif is_double_bottom(df, i):
            chart_pattern = 'Double Bottom'
        # Head and Shoulders
        elif is_head_and_shoulders(df, i):
            chart_pattern = 'Head and Shoulders'
        # Inverse Head and Shoulders
        elif is_inverse_head_and_shoulders(df, i):
            chart_pattern = 'Inverse Head and Shoulders'
        
        chart_patterns.append(chart_pattern)
    
    df['Candlestick Pattern'] = candlestick_patterns
    df['Chart Pattern'] = chart_patterns
    return df

def create_new_csv(original_file_path, df, suffix="_labeled"):
    """
    Creates a new CSV file with the labeled patterns.
    
    Parameters:
        original_file_path (str): The path to the advanced CSV file.
        df (pd.DataFrame): The DataFrame with labeled patterns.
        suffix (str): Suffix to append to the original file name.
        
    Returns:
        str: The path to the new CSV file.
    """
    base, ext = os.path.splitext(original_file_path)
    new_file_path = f"{base}{suffix}{ext}"
    
    try:
        df.to_csv(new_file_path, index=False)
        print(f"Labeled CSV created successfully at: {new_file_path}")
        return new_file_path
    except Exception as e:
        print(f"An error occurred while writing the new CSV: {e}")
        raise

def main():
    # Specify the path to your advanced CSV file
    original_csv = 'D:\\codyb\\COMP6970_Final_Project_Data\\AAPL_minute_data_advanced.csv'
    
    # Step 1: Read the advanced CSV
    df = read_csv(original_csv)
    
    # Step 2: Label the data with candlestick and chart patterns
    df_labeled = label_patterns(df)
    
    # Step 3: Create the new labeled CSV
    create_new_csv(original_csv, df_labeled)

if __name__ == "__main__":
    main()
