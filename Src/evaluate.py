import os
import yaml
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# =========================================================
# 🗄️ DATABASE ENGINE: MODEL VAULT LAYERS
# =========================================================
def init_model_vault_db():
    os.makedirs("models", exist_ok=True)
    conn = sqlite3.connect('models/gold_model_vault.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS xgboost_performance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_name TEXT,
            accuracy REAL,
            precision REAL,
            strategy_cumulative_return REAL,
            buy_and_hold_return REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_metrics_to_sql(model_name, accuracy, precision, strat_ret, bnh_ret):
    conn = sqlite3.connect('models/gold_model_vault.db')
    cursor = conn.cursor()
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO xgboost_performance_logs (timestamp, model_name, accuracy, precision, strategy_cumulative_return, buy_and_hold_return)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (waktu_sekarang, model_name, float(accuracy), float(precision), float(strat_ret), float(bnh_ret)))
    conn.commit()
    conn.close()
    print(f"[SQL AUDIT] Performa model resmi dikunci ke 'models/gold_model_vault.db' 🔒")

# =========================================================
# CORE EVALUATION & BACKTEST PIPELINE
# =========================================================
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
    
    # Generate Prediksi Isyarat Trading
    test_df['Signal'] = model.predict(X_test)
    
    # --- Strategi Long & Flat ---
    test_df['Position'] = np.where(test_df['Signal'] == 1, 1, 0)
    test_df['Strategy_Return'] = test_df['Position'] * test_df['Log_Return']
    
    test_df['Cum_Buy_Hold'] = np.exp(test_df['Log_Return'].cumsum()) - 1
    test_df['Cum_Strategy'] = np.exp(test_df['Strategy_Return'].cumsum()) - 1
    
    # 📑 SUNTIKAN BARU: Ekstraksi Metrik Riil dari Hasil Backtest
    accuracy = accuracy_score(test_df['Target'], test_df['Signal'])
    precision = precision_score(test_df['Target'], test_df['Signal'], zero_division=0)
    final_strat_ret = test_df['Cum_Strategy'].iloc[-1]
    final_bnh_ret = test_df['Cum_Buy_Hold'].iloc[-1]
    
    # Pembuatan Folder Output Chart
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
    
    # 💾 SUNTIKAN BARU: Commit Riwayat Evaluasi Model ke Lapisan Database SQL
    init_model_vault_db()
    log_metrics_to_sql('XGBoost Classifier', accuracy, precision, final_strat_ret, final_bnh_ret)
    
    print("\n--- EVALUATION COMPLETE ---")
    print(f"📈 Hasil Akhir  -> Akurasi: {round(accuracy * 100, 2)}% | Presisi: {round(precision * 100, 2)}%")
    print(f"💰 Profitabilitas -> Strategi: {round(final_strat_ret * 100, 2)}% vs Market: {round(final_bnh_ret * 100, 2)}%\n")

if __name__ == "__main__":
    evaluate_and_backtest()