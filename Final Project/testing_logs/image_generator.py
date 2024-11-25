import pandas as pd
import matplotlib.pyplot as plt

# Input CSV file
input_csv = "1.csv"  # Replace with your file path

# Read the CSV file
df = pd.read_csv(input_csv)

# Initialize the starting value for all series
initial_value = 500000

# Adjust each column to start at 500,000
df['TQQQ5_adjusted'] = initial_value + (df['TQQQ5'] - df['TQQQ5'].iloc[0])
df['TQQQ_B&H_adjusted'] = initial_value + (df['TQQQ_B&H'] - df['TQQQ_B&H'].iloc[0])
df['TQQQ18_adjusted'] = initial_value + (df['TQQQ18'] - df['TQQQ18'].iloc[0])

# Calculate the dynamic y-axis range
min_value = min(df[['TQQQ5_adjusted', 'TQQQ_B&H_adjusted', 'TQQQ18_adjusted']].min())
max_value = max(df[['TQQQ5_adjusted', 'TQQQ_B&H_adjusted', 'TQQQ18_adjusted']].max())
y_margin = 0.05 * (max_value - min_value)  # 5% margin on both sides
y_min = max(0, min_value - y_margin)
y_max = max_value + y_margin

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['TQQQ5_adjusted'], label='TQQQ5', marker='o')
plt.plot(df['TQQQ_B&H_adjusted'], label='TQQQ_B&H', marker='o')
plt.plot(df['TQQQ18_adjusted'], label='TQQQ18', marker='o')

# Customize the plot
plt.title("Value Changes Over Time")
plt.xlabel("Day")
plt.ylabel("Value")
plt.ylim(y_min, y_max)  # Dynamically adjust y-axis
plt.legend()
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()
