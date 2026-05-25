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

def evaluate_and_backtest(input_path="data/processed/xauusd_features.csv", model_path="models/xgboost_model_tuned.pkl"):
    print("Loading model and data for evaluation...")
    config = load_config()
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    X = df.drop(columns=['Target', 'Log_Return'])
    test_size = config['model']['test_size']
    split_idx = int(len(df) * (1 - test_size))
    
    test_df = df.iloc[split_idx:].copy()
    X_test = X.iloc[split_idx:]
    
    test_df['Signal'] = model.predict(X_test)
    
    # --- PERUBAHAN: Strategi Long & Flat ---
    # Jika 1 = Beli (1), Jika 0 = Pegang Uang Tunai (0)
    test_df['Position'] = np.where(test_df['Signal'] == 1, 1, 0)
    
    test_df['Strategy_Return'] = test_df['Position'] * test_df['Log_Return']
    
    test_df['Cum_Buy_Hold'] = np.exp(test_df['Log_Return'].cumsum()) - 1
    test_df['Cum_Strategy'] = np.exp(test_df['Strategy_Return'].cumsum()) - 1
    
    os.makedirs("docs/images", exist_ok=True)
    
    print("Generating Equity Curve plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['Cum_Buy_Hold'] * 100, label='Buy & Hold (Gold Asli)', color='gray', linestyle='--')
    plt.plot(test_df.index, test_df['Cum_Strategy'] * 100, label='XGBoost Long & Flat', color='gold', linewidth=2)
    plt.title('XAUUSD Algorithmic Trading (Long & Flat Strategy)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('docs/images/equity_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
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

if __name__ == "__main__":
    evaluate_and_backtest()