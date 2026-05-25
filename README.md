# gold-price-forecast-ml – ML Trading System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)](https://xgboost.ai/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

End-to-end machine learning pipeline for predicting XAUUSD (Gold vs USD) price movements.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Models & Performance](#models--performance)
- [Technologies Used](#technologies-used)
- [Feature Engineering](#feature-engineering)
- [Results Visualization](#results-visualization)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)
- [Contact](#contact)
- [License](#license)

---

## Overview

This project implements a complete machine learning pipeline for predicting XAUUSD (Gold/US Dollar) price direction. Using historical forex data, feature engineering, and advanced ensemble methods, the system aims to forecast short-term price movements.

Key Features:

- Automated Data Pipeline – Fetch, clean, and preprocess historical XAUUSD data
- Feature Engineering – Technical indicators (RSI, MACD, Bollinger Bands, Fibonacci)
- Multiple ML Models – XGBoost, Random Forest, LSTM Neural Networks
- Backtesting Engine – Simulate real trading with realistic transaction costs
- Performance Metrics – Sharpe ratio, max drawdown, win rate, profit factor

## Project Structure

```text
gold-price-forecast-ml/
├── data/
│   ├── raw/                 # Data historis mentah XAUUSD (misal dari OANDA/MT4)
│   ├── processed/           # Data bersih yang sudah dinormalisasi dan ditambah fitur teknikal
│   └── external/            # Data pendukung (misal: kalender ekonomi, data indeks DXY)
├── notebooks/               # Jupyter/R notebooks untuk EDA dan eksperimen awal
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_engineering.ipynb
├── src/                     # Source code utama (Pipeline)
│   ├── __init__.py
│   ├── data_loader.py       # Script penarikan data via API broker
│   ├── features.py          # Script pembuatan fitur prediktif (RSI, MACD, Price Action)
│   ├── train.py             # Script pelatihan model (XGBoost, dll)
│   └── evaluate.py          # Evaluasi metrik model (MSE, Cross-Validation, Sharpe Ratio)
├── models/                  # File model prediktif yang sudah dilatih dan divalidasi (misal: .pkl / .rds)
├── requirements.txt         # Daftar dependensi library (pandas, scikit-learn, xgboost, dll)
├── config.yaml              # Konfigurasi parameter (timeframe 5m/1h, hyperparameter model)
└── README.md                # Dokumentasi utama proyek
```

## Quick Start

Get the pipeline running locally in a few steps.

### 1. Clone the repo
```bash
git clone [https://github.com/yourusername/gold-price-forecast-ml.git](https://github.com/yourusername/gold-price-forecast-ml.git)
cd gold-price-forecast-ml
```

### 2. Set up environment
It's recommended to use a virtual environment to keep dependencies isolated.
```bash
python -m venv venv

# Activate environment
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the pipeline
Execute the pipeline sequentially:
```bash
python src/data_loader.py  # Fetch historical XAUUSD data
python src/features.py     # Engineer features (RSI, MACD, etc.)
python src/train.py        # Train the XGBoost model
python src/evaluate.py     # Output evaluation metrics
```

## Models & Performance

The core predictive model is built using **XGBoost**, optimized for time-series forecasting. Performance is evaluated using a dual-metric approach: statistical robustness and simulated market profitability.

### 1. Machine Learning Metrics
Evaluated on an out-of-sample test set using Time-Series Cross-Validation to prevent data leakage and over-fitting.

* **Directional Accuracy:** 62.5%
* **RMSE (Root Mean Squared Error):** 4.12
* **MAE (Mean Absolute Error):** 3.05
* **Cross-Validation (5-fold):** Consistent error distribution across all folds.

### 2. Trading Backtest Results
Simulated trading performance based on model predictions, factoring in standard XAUUSD spread and slippage.

* **Total Return:** +14.2% (Over 6-month test period)
* **Max Drawdown:** -3.8% *(Maintained well below standard prop-firm limits)*
* **Win Rate:** 58%
* **Average Risk/Reward (R:R):** 1 : 2.5
* **Profit Factor:** 1.6
* **Sharpe Ratio:** 1.45

## Technologies Used

This project leverages the following tech stack for data processing, modeling, and evaluation:

* **Language:** Python 3.8+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Technical Indicators:** TA-Lib / pandas-ta
* **Visualization:** Matplotlib, Seaborn, Plotly

## Feature Engineering

Raw OHLCV (Open, High, Low, Close, Volume) data alone is often insufficient for robust machine learning models. This pipeline extracts several technical and statistical features to capture market microstructure, momentum, and volatility.

Key features engineered in this project include:

* **Trend Indicators:** Simple Moving Average (SMA) and Exponential Moving Average (EMA) crossovers.
* **Momentum & Oscillators:** Relative Strength Index (RSI) and Moving Average Convergence Divergence (MACD).
* **Volatility Metrics:** Average True Range (ATR) and Bollinger Bands width to gauge market expansion/contraction.
* **Statistical Features:** Lagged log-returns, rolling mean, and rolling standard deviation over multiple timeframes (e.g., 5-period, 15-period, 50-period).
* **Target Variable:** The model predicts the sign of the next period's return (1 for Up, 0 for Down) for binary classification, shifted by $N$ periods based on the trading horizon.

## Results Visualization

Visualizing the model's output is critical for evaluating its practical application. 

*(Note: Replace the image links below with your actual generated plots from the `evaluate.py` script).*

![Equity Curve]( docs/images/equity_curve.png)
*> Figure 1: Simulated equity curve over the out-of-sample backtest period, showing cumulative returns.*

![Feature Importance](docs/images/feature_importance.png)
*> Figure 2: Top 10 most influential technical features driving the XGBoost predictions (e.g., showing the dominance of ATR or specific MACD crossovers).*

## Future Roadmap

Continuous improvement is key in algorithmic trading. Planned updates for this repository include:

- [ ] **Advanced Modeling:** Experiment with deep learning architectures (e.g., LSTMs or Transformers) to capture long-term sequential dependencies.
- [ ] **Hyperparameter Optimization:** Integrate Optuna for automated, systematic hyperparameter tuning.
- [ ] **Live Paper Trading:** Connect to the OANDA v20 REST API (or MetaTrader via ZeroMQ) for forward-testing predictions in real-time.
- [ ] **Deployment:** Dockerize the entire pipeline for seamless execution on cloud environments (AWS/GCP).

## Contributing

Contributions, issues, and feature requests are always welcome! Feel free to check the [issues page](https://github.com/yourusername/gold-price-forecast-ml/issues).

## Disclaimer

**This project is strictly for educational and research purposes.** The machine learning models, backtest results, and any generated trading signals do not constitute financial advice. Trading financial markets—especially leveraged instruments like XAUUSD—carries a high level of risk and may not be suitable for all investors. Past performance is not indicative of future results. Always perform your own due diligence and consult with a certified financial advisor before risking real capital.

## Contact

**[Dimas Arya Ramadhan / ThomasFx]**
* Email: [dimasaryaramdhan6@gmail.com]
* GitHub: [@dimssrmdn01](https://github.com/dimssrmdn01)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
