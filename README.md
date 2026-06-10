#  Gold Price (XAUUSD) Directional Forecasting using XGBoost (Pro Version)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey)
![Status](https://img.shields.io/badge/Status-Completed-success)

An end-to-end Machine Learning pipeline and model vault logging architecture designed to forecast the daily directional movement of Gold (XAUUSD). This repository demonstrates modular data engineering, sequential time-series feature extraction, out-of-sample backtesting, and automated SQL model-performance tracking.

---

## 🚀 Live Demo
Access the live web application here: https://dimas-xauusd-engine.streamlit.app/

---

##  Project Structure

```text
gold-price-forecast-ml/
│
├── data/
│   ├── raw/                  # Raw historical OHLCV data via yfinance
│   └── processed/            # Cleaned data with engineered technical indicators
├── docs/
│   └── images/               # Generated charts (Equity Curve & Feature Importance)
├── models/
│   ├── gold_model_vault.db   # SQL Database tracking production metrics history
│   └── serialized_model.pkl  # Serialized XGBoost model artifacts
├── Src/
│   ├── data_loader.py        # Automated data ingestion script
│   ├── features.py           # Technical indicator and feature engineering pipeline
│   ├── train.py              # XGBoost model training configuration
│   └── evaluate.py           # Equity curve backtesting & SQL audit tracking layer
├── config.yaml               # Global parameters and hyperparameters
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

##  Data & Feature Engineering

Raw daily Close/Volume data alone is insufficient for gradient boosting frameworks. The processing pipeline expands the mathematical feature space into:
* **Trend Indicators:** Simple Moving Averages (SMA 10, SMA 30) to establish baseline support.
* **Volatility Metrics:** Bollinger Bands Width gauging market expansion and contraction regimes.
* **Momentum & Statistical Lags:** Dynamic log-returns and historical daily sequential return parameters to map auto-correlation structures.

---

##  Models & Performance Tracking

The core pipeline utilizes **XGBoost (Extreme Gradient Boosting)** to isolate non-linear relationships without assuming historical norm distributions. 

To guarantee production safety, every iteration run commits validation parameters directly into an integrated relasionali database (`gold_model_vault.db`):

| SQL Column | Storage Class | Analytical Focus |
| :--- | :--- | :--- |
| **timestamp** | TEXT | Records execution times for drift and version monitoring. |
| **accuracy / precision** | REAL | Measures validation boundaries to prevent classification errors. |
| **strategy_cumulative_return** | REAL | Monitors directional profitability against market benchmarks. |
| **buy_and_hold_return** | REAL | Baseline evaluation index tracking underlying asset performance. |

---

##  How to Run (Local Setup)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/dimssrmdn01/gold-price-forecast-ml.git](https://github.com/dimssrmdn01/gold-price-forecast-ml.git)
   cd gold-price-forecast-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the pipeline sequentially:**
   ```bash
   python Src/data_loader.py
   python Src/features.py
   python Src/train.py
   python Src/evaluate.py
   ```
