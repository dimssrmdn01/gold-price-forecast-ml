import os
import yaml
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def train_model(input_path="data/processed/xauusd_features.csv", model_path="models/xgboost_model.pkl"):
    print("Reading configuration and loading processed data...")
    config = load_config()
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # Memisahkan Fitur (X) dan Target (y)
    # Kolom 'Target' dan 'Log_Return' didrop agar tidak terjadi kebocoran data (data leakage)
    X = df.drop(columns=['Target', 'Log_Return'])
    y = df['Target']
    
    # Time-Series Split (Penting: Data time-series tidak boleh di-shuffle/diacak)
    test_size = config['model']['test_size']
    split_idx = int(len(df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training data: {len(X_train)} rows | Testing data: {len(X_test)} rows")
    
    # Inisialisasi dan Pelatihan Model XGBoost
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=config['model']['n_estimators'],
        learning_rate=config['model']['learning_rate'],
        random_state=config['model']['random_state'],
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluasi Performa Model
    print("Evaluating model...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"\n--- RESULTS ---")
    print(f"Accuracy on Test Set: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    # Menyimpan Model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\nModel successfully saved to: {model_path}")

if __name__ == "__main__":
    train_model()