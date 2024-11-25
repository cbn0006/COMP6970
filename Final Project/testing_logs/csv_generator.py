import pandas as pd

# Input CSV file
input_csv = "results2.csv"  # Replace with your file path

# Output CSV file
output_csv = "1.csv"

# Define new column names
new_col1_name = "SPXL18"  # Set your desired column name
new_col2_name = "SPXL_B&H"  # Set your desired column name

# Read the input CSV file
df = pd.read_csv(input_csv)

# Extract the last two columns
last_two_cols = df.iloc[:, -2:].copy()

# Rename the columns
last_two_cols.columns = [new_col1_name, new_col2_name]

# Append to the output CSV file
try:
    # Check if the file already exists
    existing_data = pd.read_csv(output_csv)

    # Filter out columns that already exist in the output file
    for col in last_two_cols.columns:
        if col in existing_data.columns:
            print(f"Column '{col}' already exists in {output_csv}. Skipping...")
            last_two_cols.drop(columns=[col], inplace=True)

    # Append only the new columns
    if not last_two_cols.empty:
        combined_data = pd.concat([existing_data, last_two_cols], axis=1)
        combined_data.to_csv(output_csv, index=False)
        print(f"Appended new columns to {output_csv}: {', '.join(last_two_cols.columns)}")
    else:
        print("No new columns to append. All columns already exist.")
except FileNotFoundError:
    # If the file doesn't exist, create it
    last_two_cols.to_csv(output_csv, index=False)
    print(f"Created {output_csv} and saved data.")
