import os
import yaml
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def tune_hyperparameters(input_path="data/processed/xauusd_features.csv", model_path="models/xgboost_model_tuned.pkl"):
    print("Loading processed data for hyperparameter tuning...")
    config = load_config()
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    X = df.drop(columns=['Target', 'Log_Return'])
    y = df['Target']
    
    # Split train/test
    test_size = config['model']['test_size']
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Mencegah model malas: Hitung rasio untuk menyeimbangkan kelas Up (1) dan Down (0)
    ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Perluas rentang pencarian parameter
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.5, 1, 2],
        'min_child_weight': [1, 3, 5]
    }
    
    base_model = XGBClassifier(
        eval_metric='logloss', 
        random_state=config['model']['random_state'],
        scale_pos_weight=ratio # <--- Memaksa model adil terhadap posisi Sell
    )
    
    print("Starting Randomized Search Cross-Validation...")
    print("Optimizing for MACRO F1-SCORE to force active trading (Long & Short)...")
    
    # Ganti scoring menjadi f1_macro
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=tscv,
        scoring='f1_macro', 
        random_state=config['model']['random_state'],
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print("\n--- TUNING RESULTS ---")
    print(f"Best Parameters Found: {random_search.best_params_}")
    
    best_model = random_search.best_estimator_
    test_preds = best_model.predict(X_test)
    
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, test_preds))
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    print(f"\nActive trading model successfully saved to: {model_path}")

if __name__ == "__main__":
    tune_hyperparameters()