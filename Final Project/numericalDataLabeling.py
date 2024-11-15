import pandas as pd
import os
from ta.trend import ADXIndicator
from ta import trend, momentum, volatility, volume
import numpy as np
from sklearn.linear_model import LinearRegression

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

# def is_doji(row, threshold=0.1):
#     return abs(row['close'] - row['open']) <= threshold * (row['high'] - row['low'])

# Good
def is_hammer(row):
    body = abs(row['close'] - row['open'])
    lower_shadow = abs(row['low'] - min(row['open'], row['close']))
    upper_shadow = abs(row['high'] - max(row['open'], row['close']))
    return (
        lower_shadow > 2 * body and 
        upper_shadow < 0.1 * body and
        row['close'] > row['open']
    )

# Good
def is_inverted_hammer(row):
    body = abs(row['close'] - row['open'])
    lower_shadow = abs(row['low'] - min(row['open'], row['close']))
    upper_shadow = abs(row['high'] - max(row['open'], row['close']))
    return (
        upper_shadow > 2 * body and
        lower_shadow < 0.1 * body and
        row['close'] > row['open']
    )

# Good
def is_shooting_star(row):
    body = abs(row['close'] - row['open'])
    lower_shadow = abs(row['low'] - min(row['open'], row['close']))
    upper_shadow = abs(row['high'] - max(row['open'], row['close']))
    return (
        upper_shadow > 2 * body and
        lower_shadow < 0.1 * body and
        row['close'] < row['open']
    )  

# Good (Technically)
def is_bullish_engulfing(prev_row, current_row):
    return (prev_row['close'] < prev_row['open']) and \
           (current_row['close'] > current_row['open']) and \
           (current_row['open'] < prev_row['close']) and \
           (current_row['close'] > prev_row['open'])

'''def is_bullish_engulfing_strong(prev_row, current_row):
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
           (current_row['low'] < prev_row['low'])'''

# Good (Technically)
def is_bearish_engulfing(prev_row, current_row):
    return (prev_row['close'] > prev_row['open']) and \
           (current_row['close'] < current_row['open']) and \
           (current_row['open'] > prev_row['close']) and \
           (current_row['close'] < prev_row['open'])

# Good
def is_morning_star(prev_row, mid_row, curr_row, threshold=0.1):
    return (
        prev_row['close'] < prev_row['open'] and
        curr_row['open'] < curr_row['close'] and
        abs(mid_row['open'] - mid_row['close']) < threshold * (prev_row['high'] - prev_row['low']) and
        curr_row['close'] > prev_row['open'] - (threshold * abs(prev_row['close'] - prev_row['open']))
    )

# Good
def is_evening_star(prev_row, mid_row, curr_row, threshold=0.1):
    return (
        prev_row['close'] > prev_row['open'] and
        curr_row['open'] > curr_row['close'] and
        abs(mid_row['open'] - mid_row['close']) < threshold * (prev_row['high'] - prev_row['low']) and
        curr_row['close'] < prev_row['open'] + (threshold * abs(prev_row['close'] - prev_row['open']))
    )

# Good
def is_hanging_man(row):
    body = abs(row['close'] - row['open'])
    lower_shadow = abs(row['low'] - min(row['open'], row['close']))
    upper_shadow = abs(row['high'] - max(row['open'], row['close']))
    return (
        lower_shadow > 2 * body and
        upper_shadow < 0.1 * body and
        row['close'] < row['open']
    )

# Good
def is_bullish_harami(prev_row, curr_row):
    return (
        prev_row['close'] < prev_row['open'] and
        curr_row['close'] > curr_row['open'] and
        curr_row['close'] < prev_row['open'] and
        curr_row['open'] > prev_row['close']
    )

# Paper 1's Research
def label_candles(df):
    candlestick_patterns = ['None'] * len(df)
    df = df.reset_index(drop=True)
    
    for i in range(len(df)):
        candlestick_pattern = 'None'
        current_row = df.loc[i]
        
        # if is_doji(current_row):
        #     candlestick_pattern = 'Doji'
        if is_hammer(current_row):
            candlestick_pattern = 'Hammer'
        elif is_inverted_hammer(current_row):
            candlestick_pattern = 'Inverted Hammer'
        elif is_shooting_star(current_row):
            candlestick_pattern = 'Shooting Star'
        
        if i > 0:
            prev_row = df.loc[i - 1]
            if is_bullish_engulfing(prev_row, current_row):
                candlestick_pattern = 'Bullish Engulfing'
            # if is_bullish_engulfing_strong(prev_row, current_row):
            #     candlestick_pattern = 'Strong Bullish Engulfing'
            # elif is_bullish_engulfing_weak(prev_row, current_row):
            #     candlestick_pattern = 'Weak Bullish Engulfing'
            elif is_bearish_engulfing(prev_row, current_row):
                candlestick_pattern = 'Bearish Engulfing'
            elif is_bullish_harami(prev_row, current_row):
                candlestick_pattern = 'Bullish Harami'
            elif is_hanging_man(current_row):
                candlestick_pattern = 'Hanging Man'
        
        if i > 1:
            prev_row = df.loc[i - 2]
            mid_row = df.loc[i - 1]
            if is_morning_star(prev_row, mid_row, current_row):
                candlestick_patterns[i - 1] = 'Morning Star'
            elif is_evening_star(prev_row, mid_row, current_row):
                candlestick_patterns[i - 1] = 'Evening Star'

        if candlestick_patterns[i] == 'None':
            candlestick_patterns[i] = candlestick_pattern
    
    df['Candlestick Pattern'] = candlestick_patterns
    return df

def label_bullish_bearish(df, window=20):
    labels = ["None"] * len(df)
    
    # Loop through the DataFrame in steps of 20 minutes
    for i in range(0, len(df) - window + 1, window):
        # Select the current 20-minute window
        window_df = df.iloc[i:i + window]
        
        # Calculate the midpoint (average) of each candle's open and close prices in the window
        midpoints = (window_df['open'] + window_df['close']) / 2
        
        # Prepare data for linear regression
        X = np.arange(window).reshape(-1, 1)  # Time steps as input
        y = midpoints.values.reshape(-1, 1)  # Midpoints as output
        
        # Perform linear regression to get the slope
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0][0]  # Extract the slope
        
        # Label the last row in this window as 'Bullish' or 'Bearish' based on slope
        if slope > 0:
            labels[i + window - 1] = "Bullish"
        elif slope < 0:
            labels[i + window - 1] = "Bearish"
    
    # Add the labels to the DataFrame
    df['Bullish/Bearish'] = labels
    return df

'''
Above is the labeling for candlesticks. This is Wu's research. (Paper 1)
Below is labeling for chart patterns. This is my own research/contribution.
Other research includes: Using RL and extremely complex architectures to perform trades (Paper 2).
Using Object recoginition systems (pattern recognition) to find buy/sell signals in technical charts.
'''

# Labeling for my own research
def detect_double_top(df):
    double_top_labels = ['None'] * len(df)

    for i in range(1, len(df) - 1):
        # Check for a 0Support0 line
        if df['Support Line'].iloc[i] == '0Support0':
            # Check if the associated trough meets the {X}Trough{Y} conditions
            trough_label = df['Peak/Trough'].iloc[i]
            
            # Ensure the label follows the format "{X}Trough{Y}"
            if 'Trough' in trough_label:
                try:
                    # Extract X and Y values from the label
                    X, Y = map(int, trough_label.replace("Trough", " ").split())
                    
                    # Check if the trough meets the double top conditions
                    if abs(X - Y) <= 10 and max(X, Y) <= 50:
                        if i - (X + 1) >= 0 and i + Y + 1 <= len(df) - 1:
                            double_top_labels[i] = "DoubleTop"
                except ValueError:
                    # Skip if the label format does not match "{X}Trough{Y}"
                    continue

    # Add the double top labels to the DataFrame
    df['Double Top'] = double_top_labels
    return df

# Labeling for my own research
def detect_double_bottom(df):
    double_bottom_labels = ['None'] * len(df)

    for i in range(1, len(df) - 1):
        # Check for a 0Ceiling0 line
        if df['Ceiling Line'].iloc[i] == '0Ceiling0':
            # Check if the associated peak meets the {X}Peak{Y} conditions
            peak_label = df['Peak/Trough'].iloc[i]
            
            # Ensure the label follows the format "{X}Peak{Y}"
            if 'Peak' in peak_label:
                try:
                    # Extract X and Y values from the label
                    X, Y = map(int, peak_label.replace("Peak", " ").split())
                    
                    # Check if the peak meets the double bottom conditions
                    if abs(X - Y) <= 10 and max(X, Y) <= 50:
                        # Ensure the range falls within the dataset boundaries
                        if i - (X + 1) >= 0 and i + Y + 1 <= len(df) - 1:
                            double_bottom_labels[i] = "DoubleBottom"
                except ValueError:
                    # Skip if the label format does not match "{X}Peak{Y}"
                    continue

    # Add the double bottom labels to the DataFrame
    df['Double Bottom'] = double_bottom_labels
    return df

# Labeling for my own research
def detect_head_and_shoulders(df):
    head_and_shoulders_labels = ['None'] * len(df)

    i = 0
    while i < len(df) - 1:
        if df['Support Line'].iloc[i] == '0Support1':
            start_support_index = i

            j = i + 1
            while j < len(df):
                if "Support" in df['Support Line'].iloc[j]:
                    if df['Support Line'].iloc[j] == '1Support0':
                        end_support_index = j

                        peaks_in_range = df['Peak/Trough'].iloc[start_support_index + 1:end_support_index]
                        peak_indices = [k for k, label in peaks_in_range.items() if 'Peak' in label or 'Anomaly' in label]

                        if len(peak_indices) == 1:
                            peak_index = peak_indices[0] + start_support_index + 1

                            if 0 <= peak_index < len(df) and 'Peak' in df['Peak/Trough'].iloc[peak_index]:
                                head_peak_height = df['high'].iloc[peak_index]

                                left_peak_height = df['high'].iloc[start_support_index - 1] if start_support_index - 1 >= 0 else -float('inf')
                                right_peak_height = df['high'].iloc[end_support_index + 1] if end_support_index + 1 < len(df) else -float('inf')

                                if head_peak_height > left_peak_height and head_peak_height > right_peak_height:
                                    head_and_shoulders_labels[peak_index] = "HeadAndShoulders"
                        break
                    else:
                        break
                j += 1
        i += 1

    df['Head And Shoulders'] = head_and_shoulders_labels
    return df

# Labeling for my own research
def is_inverse_head_and_shoulders(df, index, lookback=30, tolerance=0.005):
    print("X")

# Labeling for my own research
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

# Labeling for my own research
def detect_ceiling_lines(df, threshold=0.001):
    ceiling_lines = ['None'] * len(df)

    for i in range(1, len(df) - 1):
        # Check if the point is labeled as a peak and if it's the highest within the next 7 candles
        if 'Peak' in df['Peak/Trough'].iloc[i] and (i + 7 < len(df)) and df['high'].iloc[i] >= df['high'].iloc[i + 1:i + 5].max():
            ceiling_price = df['high'].iloc[i]
            test_count = 0

            # Count how often the ceiling is tested without breaking
            for j in range(i + 1, len(df)):
                high_price = df['high'].iloc[j]
                is_peak = 'Peak' in df['Peak/Trough'].iloc[j]  # Check if this point is also a peak

                # Increment test count if within threshold bounds and it's a peak
                if ceiling_price * (1 - threshold) <= high_price <= ceiling_price * (1 + threshold) and is_peak:
                    test_count += 1
                elif high_price > ceiling_price * (1 + threshold):
                    # Ceiling is broken if it rises above the upper threshold
                    break

            # Label the ceiling line at the peak point with the test count
            ceiling_lines[i] = f"Ceiling{test_count}"

    df['Ceiling Line'] = ceiling_lines
    return df

# Labeling for my own research
def reverse_support_lines(df, threshold=0.001):
    for i in range(len(df) - 2, 0, -1):
        # Check if the point is labeled as a trough and if it's the lowest within the previous 7 candles
        if 'Trough' in df['Peak/Trough'].iloc[i] and (i - 5 >= 0) and df['low'].iloc[i] <= df['low'].iloc[max(i - 5, 0):i].min():
            support_price = df['low'].iloc[i]
            reverse_test_count = 0

            # Count how often the support is tested without breaking, looking backward
            for j in range(i - 1, -1, -1):
                low_price = df['low'].iloc[j]
                is_trough = 'Trough' in df['Peak/Trough'].iloc[j]  # Check if this point is also a trough

                # Increment test count if within threshold bounds and it's a trough
                if support_price * (1 - threshold) <= low_price <= support_price * (1 + threshold) and is_trough:
                    reverse_test_count += 1
                elif low_price < support_price * (1 - threshold):
                    # Support is broken if it drops below the threshold
                    break

            # Prepend the reverse test count to the existing support line label
            if df.loc[i, 'Support Line'] != 'None':
                df.loc[i, 'Support Line'] = f"{reverse_test_count}" + df.loc[i, 'Support Line']

    return df

# Labeling for my own research
def reverse_ceiling_lines(df, threshold=0.001):
    for i in range(len(df) - 2, 0, -1):
        # Check if the point is labeled as a peak and if it's the highest within the previous 7 candles
        if 'Peak' in df['Peak/Trough'].iloc[i] and (i - 5 >= 0) and df['high'].iloc[i] >= df['high'].iloc[max(i - 5, 0):i].max():
            ceiling_price = df['high'].iloc[i]
            reverse_test_count = 0

            # Count how often the ceiling is tested without breaking, looking backward
            for j in range(i - 1, -1, -1):
                high_price = df['high'].iloc[j]
                is_peak = 'Peak' in df['Peak/Trough'].iloc[j]  # Check if this point is also a peak

                # Increment test count if within threshold bounds and it's a peak
                if ceiling_price * (1 - threshold) <= high_price <= ceiling_price * (1 + threshold) and is_peak:
                    reverse_test_count += 1
                elif high_price > ceiling_price * (1 + threshold):
                    # Ceiling is broken if it rises above the upper threshold
                    break

            # Prepend the reverse test count to the existing ceiling line label
            if df.loc[i, 'Ceiling Line'] != 'None':
                df.loc[i, 'Ceiling Line'] = f"{reverse_test_count}" + df.loc[i, 'Ceiling Line']

    return df

# Labeling for my own research
def detect_support_lines(df, threshold=0.001):
    support_lines = ['None'] * len(df)

    for i in range(1, len(df) - 1):
        if 'Trough' in df['Peak/Trough'].iloc[i] and (i + 7 < len(df)) and df['low'].iloc[i] <= df['low'].iloc[i + 1:i + 6].min():
            support_price = df['low'].iloc[i]
            test_count = 0

            for j in range(i + 1, len(df)):
                low_price = df['low'].iloc[j]
                is_trough = 'Trough' in df['Peak/Trough'].iloc[j]
                
                # If low price is within threshold of support line and it is a trough, increment support tested
                if support_price * (1 - threshold) <= low_price <= support_price * (1 + threshold) and is_trough:
                    test_count += 1
                elif low_price < support_price * (1 - threshold):
                    break

            support_lines[i] = f"Support{test_count}"

    df['Support Line'] = support_lines
    return df

# Labeling for my own research
def label_plateaus_valleys(df, lookback=5, tolerance=0.0005):
    plateau_valley_labels = ['None'] * len(df)

    i = 0
    while i < len(df) - 5:
        if i >= lookback:
            trend = df['close'].iloc[i] - df['close'].iloc[i - lookback]
            close_prices = df['close'].iloc[i:i + 5].reset_index(drop=True)
            
            if all(abs((close_prices[j] - close_prices[j - 1]) / close_prices[j - 1]) <= tolerance for j in range(1, 5)):
                if abs((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) <= tolerance:
                    region_length = 5
                    j = i + 5

                    while j < len(df) and abs((df['close'].iloc[j] - df['close'].iloc[j - 1]) / df['close'].iloc[j - 1]) <= tolerance:
                        region_length += 1
                        j += 1

                    label_type = 'Plateau' if trend > 0 else 'Valley'
                    for k in range(region_length):
                        plateau_valley_labels[i + k] = f"{region_length}{label_type}{region_length - k}"

                    i += region_length - 1
        i += 1

    df['Plateau/Valley'] = plateau_valley_labels
    return df

# Labeling for my own research
def label_peaks_troughs(df):
    peak_trough_labels = ['None'] * len(df)

    for i in range(1, len(df) - 1):
        prev_high = df['high'].iloc[i - 1]
        curr_high = df['high'].iloc[i]
        next_high = df['high'].iloc[i + 1]
        
        prev_low = df['low'].iloc[i - 1]
        curr_low = df['low'].iloc[i]
        next_low = df['low'].iloc[i + 1]

        if curr_high > prev_high and curr_high > next_high and curr_low < prev_low and curr_low < next_low:
            X = 0
            # Count previous values for anomaly detection
            for j in range(i - 1, -1, -1):
                if df['high'].iloc[j] < curr_high and df['low'].iloc[j] > curr_low:
                    X += 1
                else:
                    # Breaks anomaly; set X to negative if broken by low, positive if broken by high
                    if df['high'].iloc[j] >= curr_high:
                        break
                    elif df['low'].iloc[j] <= curr_low:
                        X = -X if X > 0 else X  # Make X negative if broken by a low
                        break

            Y = 0
            # Count subsequent values for anomaly detection
            for j in range(i + 1, len(df)):
                if df['high'].iloc[j] < curr_high and df['low'].iloc[j] > curr_low:
                    Y += 1
                else:
                    # Breaks anomaly; set Y to negative if broken by low, positive if broken by high
                    if df['high'].iloc[j] >= curr_high:
                        break
                    elif df['low'].iloc[j] <= curr_low:
                        Y = -Y if Y > 0 else Y  # Make Y negative if broken by a low
                        break

            # Label the anomaly
            peak_trough_labels[i] = f"{X}Anomaly{Y}"
            continue

        elif curr_high > prev_high and curr_high > next_high:
            X = 0
            for j in range(i - 1, -1, -1):
                if df['high'].iloc[j] < curr_high:
                    X += 1
                else:
                    break

            Y = 0
            for j in range(i + 1, len(df)):
                if df['high'].iloc[j] < curr_high:
                    Y += 1
                else:
                    break

            peak_trough_labels[i] = f"{X}Peak{Y}"

        elif curr_low < prev_low and curr_low < next_low:
            X = 0
            for j in range(i - 1, -1, -1):
                if df['low'].iloc[j] > curr_low:
                    X += 1
                else:
                    break

            Y = 0
            for j in range(i + 1, len(df)):
                if df['low'].iloc[j] > curr_low:
                    Y += 1
                else:
                    break

            peak_trough_labels[i] = f"{X}Trough{Y}"

    df['Peak/Trough'] = peak_trough_labels
    return df

# Good
def create_new_csv(original_file_path, df, suffix="_labeled", directory="labeling"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    base, ext = os.path.splitext(os.path.basename(original_file_path))
    new_file_path = os.path.join(directory, f"{base}{suffix}.csv")

    try:
        df.to_csv(new_file_path, index=False)
        print(f"Labeled CSV created successfully at: {new_file_path}")
        return new_file_path
    except Exception as e:
        print(f"An error occurred while writing the new CSV: {e}")
        raise

def main():
    original_csv = 'D:\\codyb\\COMP6970_Final_Project_Data\\TSLA_minute_data_cleaned.csv'
    
    # Step 1: Read the advanced CSV
    df = read_csv(original_csv)

    # Extract unique dates to process data on a day-by-day basis
    unique_dates = df['datetime'].dt.date.unique()
    daily_results = []

    # Process each date individually
    for date in unique_dates:
        # Filter the data for the current day
        day_df = df[df['datetime'].dt.date == date].copy()
        
        # Step 2.1: Label the candlesticks for the day
        day_df = label_candles(day_df)
        
        # Step 2.2: Label the data with trends for the day
        day_df = label_singular_trends(day_df)

        # Step 2.3: Label peaks and troughs, plateaus and valleys, and support lines for the day
        day_df = label_peaks_troughs(day_df)
        day_df = label_plateaus_valleys(day_df)
        day_df = detect_support_lines(day_df)
        day_df = detect_ceiling_lines(day_df)
        day_df = reverse_ceiling_lines(day_df)
        day_df = reverse_support_lines(day_df)
        
        # Step 2.4: Detect double top patterns for the day
        day_df = detect_double_top(day_df)
        day_df = detect_double_bottom(day_df)
        # day_df = detect_head_and_shoulders(day_df)

        # Step 2.5: Label Bullish/Bearish
        day_df = label_bullish_bearish(day_df)

        # Append the processed day data to the results list
        daily_results.append(day_df)
    
    # Concatenate all daily results back into a single DataFrame
    result_df = pd.concat(daily_results, ignore_index=True)
    
    # Step 3: Create the new labeled CSV with all days combined
    create_new_csv(original_csv, result_df)

if __name__ == "__main__":
    main()
