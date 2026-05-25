import pandas as pd
import numpy as np
import os

def calculate_features(input_path="data/raw/xauusd_raw.csv", output_path="data/processed/xauusd_features.csv"):
    print("Loading raw data for feature engineering...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # Technical Indicators
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # Volatility Metrics
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (rolling_std * 4) / df['Close'].rolling(window=20).mean()
    
    # Momentum and Lags
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Lag_1'] = df['Log_Return'].shift(1)
    df['Lag_2'] = df['Log_Return'].shift(2)
    
    # Target Variable: 1 if next day's return is positive, else 0
    df['Target'] = (df['Log_Return'].shift(-1) > 0).astype(int)
    
    df.dropna(inplace=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    
    print(f"Feature engineering complete. Processed data saved to: {output_path}")
    print(f"Total features generated: {df.shape[1] - 1}")

if __name__ == "__main__":
    calculate_features()