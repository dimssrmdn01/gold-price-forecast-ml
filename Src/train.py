import os
import yaml
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_config(config_path="config.yaml"):
    # Load config
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def train_model(input_path="data/processed/xauusd_features.csv", model_path="models/xgboost_model.pkl"):
    print("Loading processed data...")
    config = load_config()
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # Split features
    X = df.drop(columns=['Target', 'Log_Return'])
    y = df['Target']
    
    # Time-series split
    test_size = config['model']['test_size']
    split_idx = int(len(df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Init model
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=config['model']['n_estimators'],
        learning_rate=config['model']['learning_rate'],
        random_state=config['model']['random_state'],
        eval_metric='logloss'
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print("\n--- RESULTS ---")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("Classification Report:\n")
    print(classification_report(y_test, predictions))
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\nModel saved: {model_path}")

if __name__ == "__main__":
    train_model()