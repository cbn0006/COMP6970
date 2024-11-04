import pandas as pd
import os
from ta.trend import ADXIndicator
from ta import trend, momentum, volatility, volume
import numpy as np

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

# Also, look into 5 minute, 10 minute downward patterns to engulfing candlestick.
def is_bullish_engulfing_strong(prev_row, current_row):
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

def is_bullish_engulfing_weak(prev_row, current_row):
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
           (current_row['open'] > prev_row['close']) and \
           (current_row['close'] > prev_row['open']) and \
           (current_row['low'] < prev_row['low'])

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

def is_double_top(df, index, lookback=60, tolerance=0.01):
    # Initialize the 'Double Top' column if it does not already exist
    if 'Double Top' not in df.columns:
        df['Double Top'] = 'None'
        
    if index < lookback or index >= len(df) - lookback:
        return False

    window = df.iloc[index - lookback:index]

    peaks = window[window['Peak/Trough'] == 'Peak']
    if peaks.empty:
        return False
    first_peak_idx = peaks.index[0]
    first_peak_price = window.loc[first_peak_idx, 'close']

    troughs = window[(window.index > first_peak_idx) & (window['Peak/Trough'] == 'Trough')]
    if troughs.empty:
        return False
    trough_idx = troughs.index[0]

    second_peaks = window[(window.index > trough_idx) & 
                          (window['Peak/Trough'] == 'Peak')]
    if second_peaks.empty:
        return False
    second_peak_idx = second_peaks.index[0]

    if second_peak_idx + 1 < len(df):
        df.loc[first_peak_idx:second_peak_idx, 'Double Top'] = 'Double Top'
        # print(f"Double Top detected from index {first_peak_idx} to {second_peak_idx}")
        return True

    return False


def is_double_bottom(df, index, lookback=25, tolerance=0.005):
    print("X")

def is_head_and_shoulders(df, index, lookback=30, tolerance=0.005):
    print("X")

def is_inverse_head_and_shoulders(df, index, lookback=30, tolerance=0.005):
    print("X")

def label_singular_trends(df):
    trends = []
    current_trend = None
    trend_count = 1
    
    # Loop through each row to determine the trend label
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            # Upward trend
            if current_trend == 'U':
                trend_count += 1
            else:
                current_trend = 'U'
                trend_count = 1
            trends.append(f"{trend_count}{current_trend}")
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            # Downward trend
            if current_trend == 'D':
                trend_count += 1
            else:
                current_trend = 'D'
                trend_count = 1
            trends.append(f"{trend_count}{current_trend}")
        else:
            # No change in close price, keep the previous trend
            trends.append(trends[-1] if trends else "1N")
    
    # Insert None for the first point (no previous data to compare)
    trends.insert(0, 'None')
    df['Singular Trend'] = trends
    return df

def detect_support_lines(df):
    support_lines = ['None'] * len(df)

    for i in range(3, len(df) - 1):
        # Check for a valley or trough followed by an upward trend
        if df['Peak/Trough'].iloc[i] in ['Trough']:
            # Check if there is an upward trend immediately after the valley/trough
            if 'U' in df['Singular Trend'].iloc[i + 1] and int(df['Singular Trend'].iloc[i + 1][0]) >= 1:
                # Set the support line at the closing price of the valley/trough
                support_price = df['close'].iloc[i]
                
                # Mark this support line in the column
                support_lines[i] = support_price

    # Add the support line levels to the DataFrame
    df['Support Line'] = support_lines
    return df

def label_plateaus_valleys(df, tolerance=0.0005):
    plateau_valley_labels = ['None'] * len(df)

    for i in range(len(df) - 5):
        # Check for a Plateau
        if i >= 3:
            if 'U' in df['Singular Trend'].iloc[i - 1] and int(df['Singular Trend'].iloc[i - 1][0]) >= 3:
                close_prices = df['close'].iloc[i:i + 5].reset_index(drop=True)
                
                # Ensure all five closing prices are within tolerance
                if all(abs((close_prices[j] - close_prices[j - 1]) / close_prices[j - 1]) <= tolerance for j in range(1, 5)):
                    # Check if the 5th price is within tolerance of the 1st price
                    if abs((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) <= tolerance:
                        # Label each of these five points as 'Plateau'
                        for j in range(i, i + 5):
                            plateau_valley_labels[j] = 'Plateau'

        # Check for a Valley
        if i >= 3:
            if 'D' in df['Singular Trend'].iloc[i - 1] and int(df['Singular Trend'].iloc[i - 1][0]) >= 3:
                close_prices = df['close'].iloc[i:i + 5].reset_index(drop=True)
                
                # Ensure all five closing prices are within tolerance
                if all(abs((close_prices[j] - close_prices[j - 1]) / close_prices[j - 1]) <= tolerance for j in range(1, 5)):
                    # Check if the 5th price is within tolerance of the 1st price
                    if abs((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) <= tolerance:
                        # Label each of these five points as 'Valley'
                        for j in range(i, i + 5):
                            plateau_valley_labels[j] = 'Valley'

    # Add the labels to the DataFrame
    df['Plateau/Valley'] = plateau_valley_labels
    return df

# Edit peaks and troughs with more conditional logic. Extrapolate out more stats for advanced trend analysis and add that to labeling decisions.
def label_peaks_troughs(df):
    peak_trough_labels = []
    
    for i in range(len(df)):
        label = 'None'

        # Check for a peak
        if i >= 3 and i <= len(df) - 4:
            if ('U' in df['Singular Trend'].iloc[i - 1] and int(df['Singular Trend'].iloc[i - 1][0]) >= 3 and
                'D' in df['Singular Trend'].iloc[i + 1] and int(df['Singular Trend'].iloc[i + 3][0]) == 3):
                label = 'Peak'
            # Make these elif statements robust for better labeling
            elif ('U' in df['Singular Trend'].iloc[i - 1] and int(df['Singular Trend'].iloc[i - 1][0]) >= 3 and
                'D' in df['Singular Trend'].iloc[i + 1] and int(df['Singular Trend'].iloc[i + 3][0]) == 3):
                label = 'Peak'
            elif ('U' in df['Singular Trend'].iloc[i - 1] and int(df['Singular Trend'].iloc[i - 1][0]) >= 3 and
                'D' in df['Singular Trend'].iloc[i + 1] and int(df['Singular Trend'].iloc[i + 3][0]) == 3):
                label = 'Peak'
            elif ('U' in df['Singular Trend'].iloc[i - 1] and int(df['Singular Trend'].iloc[i - 1][0]) >= 3 and
                'D' in df['Singular Trend'].iloc[i + 1] and int(df['Singular Trend'].iloc[i + 3][0]) == 3):
                label = 'Peak'
        
        if i >= 3 and i <= len(df) - 4:
            if ('D' in df['Singular Trend'].iloc[i - 1] and int(df['Singular Trend'].iloc[i - 1][0]) >= 3 and
                'U' in df['Singular Trend'].iloc[i + 1] and int(df['Singular Trend'].iloc[i + 3][0]) == 3):
                label = 'Trough'
        
        peak_trough_labels.append(label)
    
    # Add the labels to the DataFrame
    df['Peak/Trough'] = peak_trough_labels
    return df

def label_candles(df):
    """
    Labels each row in the DataFrame with identified individual candlestick patterns.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing stock data.
        
    Returns:
        pd.DataFrame: DataFrame with an additional 'Candlestick Pattern' column.
    """
    candlestick_patterns = []
    df = df.reset_index(drop=True)
    
    for i in range(len(df)):
        candlestick_pattern = 'None'  # Default pattern
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
            if is_bullish_engulfing_strong(prev_row, current_row):
                candlestick_pattern = 'Strong Bullish Engulfing'
            elif is_bullish_engulfing_weak(prev_row, current_row):
                candlestick_pattern = 'Weak Bullish Engulfing'
            elif is_bearish_engulfing(prev_row, current_row):
                candlestick_pattern = 'Bearish Engulfing'
        
        candlestick_patterns.append(candlestick_pattern)
    
    df['Candlestick Pattern'] = candlestick_patterns
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

def label_advanced_trends(df, threshold=0.005, short_window=3, long_window=10, rsi_threshold=70, adx_threshold=20):
    trends = []

    # Compute Short and Long Moving Averages
    df['SMA_Short'] = df['close'].rolling(window=short_window).mean().round(3)
    df['SMA_Long'] = df['close'].rolling(window=long_window).mean().round(3)
    
    # Calculate RSI
    # df['RSI'] = momentum.RSIIndicator(close=df['close'], window=14).rsi().round(3)
    
    # Calculate ADX
    # adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    # df['ADX'] = adx_indicator.adx().round(3)
    
    for i in range(1, len(df)):
        # Moving Average Crossover
        if df['SMA_Short'].iloc[i] > df['SMA_Long'].iloc[i]:
            ma_trend = 'U'
        elif df['SMA_Short'].iloc[i] < df['SMA_Long'].iloc[i]:
            ma_trend = 'D'
        else:
            ma_trend = 'N'

        # RSI-based Trend
        if df['RSI'].iloc[i] > rsi_threshold:
            rsi_trend = 'D'
        elif df['RSI'].iloc[i] < (100 - rsi_threshold):
            rsi_trend = 'U'
        else:
            rsi_trend = 'N'

        # ADX for Trend Strength
        adx_strong = df['ADX'].iloc[i] > adx_threshold

        # Minute-by-Minute Trend
        price_change = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
        if price_change > threshold:
            min_trend = 'U'
        elif price_change < -threshold:
            min_trend = 'D'
        else:
            min_trend = 'N'

        # Advanced Trend Decision: Combine Indicators
        if ma_trend == rsi_trend == min_trend:
            trend = f"{ma_trend} Strong" if adx_strong else f"{ma_trend} Weak"
        elif ma_trend == rsi_trend:
            trend = f"{ma_trend} Moderate" if adx_strong else f"{ma_trend} Weak"
        else:
            trend = f"{min_trend} Moderate" if adx_strong else min_trend

        trends.append(trend)

    # Pad the trends list to match DataFrame length
    trends.insert(0, 'None')
    df['Advanced Trend'] = trends
    return df

def main():
    # Specify the path to your advanced CSV file
    original_csv = 'D:\\codyb\\COMP6970_Final_Project_Data\\TSLA_minute_data_cleaned.csv'
    
    # Step 1: Read the advanced CSV
    df = read_csv(original_csv)

    # Step 2.1: Label the candlesticks
    df = label_candles(df)
    
    # Step 2.2: Label the data with trends
    df = label_singular_trends(df)

    # Step 2.3: Label peaks and troughs based on trends
    df = label_peaks_troughs(df)
    df = label_plateaus_valleys(df)
    df = detect_support_lines(df)
    df = label_advanced_trends(df)
    
    # Step 2.4: Label the data with candlestick and chart patterns
    df['Double Top'] = None
    for index in range(len(df)):
        is_double_top(df, index)
    
    # Step 3: Create the new labeled CSV
    create_new_csv(original_csv, df)

if __name__ == "__main__":
    main()
