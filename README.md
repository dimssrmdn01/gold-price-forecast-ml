# Gold Price (XAUUSD) Directional Forecasting using XGBoost

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning pipeline designed to forecast the daily directional movement of Gold (XAUUSD) using historical market data, technical indicators, and gradient boosting algorithms. 

This repository demonstrates data engineering, time-series feature extraction, out-of-sample model evaluation, and simulated algorithmic trading performance.

---

## 🚀 Live Demo
Access the live web application here: 

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Data & Feature Engineering](#data--feature-engineering)
4. [Models & Performance](#models--performance)
5. [Results & Visualizations](#results--visualizations)
6. [How to Run (Local Setup)](#how-to-run-local-setup)
7. [Disclaimer](#disclaimer)

---

## Project Overview

The goal of this project is to predict whether the closing price of XAUUSD will be strictly higher (1) or lower (0) on the following trading day. By treating market forecasting as a binary classification problem, this pipeline builds a foundational quantitative trading model that can be expanded into automated algorithmic trading systems.

## Project Structure

```text
gold-price-forecast-ml/
│
├── data/
│   ├── raw/                  # Raw historical OHLCV data
│   └── processed/            # Cleaned data with engineered features
├── docs/
│   └── images/               # Generated charts and visual results
├── models/                   # Serialized ML models (e.g., .pkl)
├── Src/
│   ├── data_loader.py        # Automated data ingestion script
│   ├── features.py           # Technical indicator and feature engineering
│   ├── train.py              # Model training and evaluation script
│   └── evaluate.py           # Equity curve simulation and feature importance
├── config.yaml               # Global parameters and hyperparameters
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Data & Feature Engineering

Historical daily data is retrieved automatically via `yfinance`. Raw OHLCV (Open, High, Low, Close, Volume) data alone is often insufficient for robust machine learning models. This pipeline extracts several technical and statistical features to capture market microstructure and momentum:

* **Trend Indicators:** Simple Moving Averages (SMA 10, SMA 30).
* **Volatility Metrics:** Bollinger Bands Width (gauging market expansion/contraction).
* **Momentum & Statistical Lags:** Log-returns and lagged sequential returns to capture auto-correlation.
* **Target Variable:** Shifted binary outcome (1 for Up, 0 for Down) predicting the next period's movement.

## Models & Performance

The core predictive model is built using **XGBoost (Extreme Gradient Boosting)**, chosen for its efficiency with tabular data and ability to capture non-linear relationships without heavy feature scaling.

**Out-of-Sample Evaluation:**
Because this is a time-series problem, the dataset is split sequentially (no random shuffling) to prevent future data leakage. 
* The model is evaluated not just on raw accuracy, but on its ability to generate a profitable **Equity Curve** compared to a standard Buy & Hold strategy.

## Results & Visualizations

*(Note: The strategy return assumes a simple 1:1 position sizing without transaction costs/slippage for baseline comparison).*

### 1. Simulated Equity Curve
The chart below illustrates the cumulative return of the XGBoost predictions vs. the underlying Gold asset.
![Equity Curve](docs/images/equity_curve.png)

### 2. Feature Importance
This plot shows which technical features drove the model's decision-making process the most.
![Feature Importance](docs/images/feature_importance.png)

## How to Run (Local Setup)

Want to test or improve the model on your local machine? Follow these steps:

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
   # 1. Download raw data
   python Src/data_loader.py
   
   # 2. Engineer features
   python Src/features.py
   
   # 3. Train the XGBoost model
   python Src/train.py
   
   # 4. Generate backtest and visualizations
   python Src/evaluate.py
   ```

## Disclaimer

**This project is strictly for educational, portfolio, and research purposes.** The machine learning models, backtest results, and any generated trading signals do not constitute financial advice. Trading financial markets especially leveraged instruments like XAUUSD carries a high level of risk and may not be suitable for all investors. Past performance is not indicative of future results.

## Contact

**[Dimas Arya Ramadhan / ThomasFx]**
* Email: [dimasaryaramdhan6@gmail.com]
* GitHub: [@dimssrmdn01](https://github.com/dimssrmdn01)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
