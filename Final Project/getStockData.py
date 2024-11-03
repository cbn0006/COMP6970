import os
import requests
import csv
import time
from datetime import datetime, timedelta

def get_polygon_data(symbol, start_date, end_date):
    url = POLYGON_BASE_URL.format(
        ticker=symbol,
        multiplier=1,
        timespan="minute",
        from_date=start_date,
        to_date=end_date
    )
    
    params = {"apiKey": POLYGON_API_KEY}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        print(f"Polygon.io Error: {response.status_code}")
        return []

def save_to_csv(stock_data, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline='') as csvfile:
        fieldnames = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Write header if file doesn't exist
            
        for entry in stock_data:
            writer.writerow(entry)

def format_polygon_data(data):
    formatted_data = []
    for result in data:
        formatted_data.append({
            'datetime': datetime.fromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'open': result['o'],
            'high': result['h'],
            'low': result['l'],
            'close': result['c'],
            'volume': result['v']
        })
    return formatted_data

def fetch_and_save_data(symbol, start_date, end_date, filename):
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    api_call_count = 0
    
    while current_date < end_date_dt:
        next_week = current_date + timedelta(days=7)
        if next_week > end_date_dt:
            next_week = end_date_dt
        
        current_str = current_date.strftime('%Y-%m-%d')
        next_week_str = next_week.strftime('%Y-%m-%d')
        
        polygon_data = get_polygon_data(symbol, current_str, next_week_str)
        if polygon_data:
            formatted_data = format_polygon_data(polygon_data)
            save_to_csv(formatted_data, filename)
        
        api_call_count += 1
        if api_call_count >= 5:
            print("Reached 5 API calls, sleeping for 60 seconds...")
            time.sleep(60)
            api_call_count = 0
        
        current_date = next_week

if __name__ == "__main__":

    POLYGON_API_KEY = "nFz8hqkcTGmJY9iXsAx9wtcNrw_pst7x"
    POLYGON_BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

    SAVE_DIR = "D:\\codyb\\COMP6970_Final_Project_Data"

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    symbols = ["TSLA"]
    start_date = "2023-01-01"
    end_date = "2023-04-01"
    
    for symbol in symbols:
        filename = os.path.join(SAVE_DIR, f"{symbol}_minute_data_raw.csv")
        print(f"Fetching data for {symbol} and saving to {filename}")
        fetch_and_save_data(symbol, start_date, end_date, filename)
        print(f"Data for {symbol} saved to {filename}")
