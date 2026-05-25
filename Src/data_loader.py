import os
import yaml
import yfinance as yf
import pandas as pd

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def fetch_gold_data():
    print("Reading project configuration...")
    config = load_config()
    
    ticker = config['data']['ticker']
    start = config['data']['start_date']
    end = config['data']['end_date']
    output_path = config['data']['raw_output_path']
    
    print(f"Downloading historical data for {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end)
    
    if data.empty:
        raise ValueError("Failed to download data. Check internet connection or ticker symbol.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)
    
    print(f"Raw data successfully saved to: {output_path}")
    print(f"Total rows loaded: {len(data)}")

if __name__ == "__main__":
    fetch_gold_data()