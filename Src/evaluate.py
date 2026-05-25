import os
import yaml
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def evaluate_and_backtest(input_path="data/processed/xauusd_features.csv", model_path="models/xgboost_model.pkl"):
    print("Loading model and data for evaluation...")
    config = load_config()
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # Split data yang sama persis dengan train.py
    X = df.drop(columns=['Target', 'Log_Return'])
    test_size = config['model']['test_size']
    split_idx = int(len(df) * (1 - test_size))
    
    # Ambil subset data test beserta harga aslinya untuk simulasi trading
    test_df = df.iloc[split_idx:].copy()
    X_test = X.iloc[split_idx:]
    
    # Predict arah market
    test_df['Signal'] = model.predict(X_test)
    # Ubah signal 0 (turun) menjadi -1 untuk posisi Short/Sell
    test_df['Position'] = np.where(test_df['Signal'] == 1, 1, -1)
    
    # Hitung Strategy Return (Posisi hari ini dikali dengan return besok)
    test_df['Strategy_Return'] = test_df['Position'] * test_df['Log_Return']
    
    # Hitung akumulasi return (Equity Curve)
    test_df['Cum_Buy_Hold'] = np.exp(test_df['Log_Return'].cumsum()) - 1
    test_df['Cum_Strategy'] = np.exp(test_df['Strategy_Return'].cumsum()) - 1
    
    # Buat folder output gambar
    os.makedirs("docs/images", exist_ok=True)
    
    # 1. Plot Equity Curve
    print("Generating Equity Curve plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['Cum_Buy_Hold'] * 100, label='Buy & Hold (Gold Asli)', color='gray', linestyle='--')
    plt.plot(test_df.index, test_df['Cum_Strategy'] * 100, label='XGBoost Strategy', color='gold', linewidth=2)
    plt.title('XAUUSD Algorithmic Trading Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('docs/images/equity_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot Feature Importance
    print("Generating Feature Importance plot...")
    importance = model.feature_importances_
    feat_names = X_test.columns
    feat_imp = pd.Series(importance, index=feat_names).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Relative Importance Score')
    plt.savefig('docs/images/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n--- EVALUATION COMPLETE ---")
    print("Graph 1 saved: docs/images/equity_curve.png")
    print("Graph 2 saved: docs/images/feature_importance.png")

if __name__ == "__main__":
    evaluate_and_backtest()