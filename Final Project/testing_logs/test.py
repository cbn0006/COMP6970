import pandas as pd
import os

# Path to the directory containing trade log CSVs
# Path to the TQQQ data CSV
tqqq_csv_path = "../labeling/SPXL_minute_data_cleaned.csv"
# Output CSV file
output_csv = "results2.csv"

# Load the TQQQ data
tqqq_data = pd.read_csv(tqqq_csv_path)
tqqq_data['datetime'] = pd.to_datetime(tqqq_data['datetime'])  # Ensure datetime is parsed
tqqq_data.set_index('datetime', inplace=True)

# Prepare results
results = []

total1, total2 = 500000,500000

# Process each trade log CSV
for i in range(1, 92):
    # Load the trade log
    trade_log_file = f"./trade_log_day_{i}.csv"
    trade_log = pd.read_csv(trade_log_file)
    
    # Extract the last row's total_asset_value and compute the percentage change
    last_row = trade_log.iloc[-1]
    total_asset_value = last_row['total_asset_value']
    percent_change = (total_asset_value - 500_000) / 500_000

    # Extract the datetime from the last row
    datetime = pd.to_datetime(last_row['datetime'])

    # Find the same datetime day in TQQQ data
    same_day = datetime.date()
    tqqq_same_day = tqqq_data.loc[tqqq_data.index.date == same_day]
    
    if not tqqq_same_day.empty:
        try:
            # Extract closing prices at 8:30 and 15:00
            close_8_30 = tqqq_same_day.at[tqqq_same_day.index[tqqq_same_day.index.time == pd.Timestamp("08:30").time()][0], 'close']
            close_15_00 = tqqq_same_day.at[tqqq_same_day.index[tqqq_same_day.index.time == pd.Timestamp("15:00").time()][0], 'close']
            
            # Compute the percentage change
            percent_change_2 = (close_15_00 - close_8_30) / close_8_30

            diff = percent_change - percent_change_2

            total1 += total1 * percent_change
            total2 += total2 * percent_change_2
            # Append the results
            results.append([datetime, percent_change, percent_change_2, diff, total1, total2])
        except IndexError:
            print(f"Data for 8:30 or 15:00 missing on {same_day}. Skipping.")

# Create a DataFrame for results
results_df = pd.DataFrame(results, columns=['datetime', 'RL', 'B&H', 'diff', 'tot1', 'tot2'])

# Save to a CSV file
results_df.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}.")

cum_change = 0
for i in range(0,91):
    cum_change += results_df['diff'].iloc[i]

print("Cumulative Change: " + str(cum_change))

print("Positive means better. Negative means worse.")
