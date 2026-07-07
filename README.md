# 🏛️ Institutional Asset Quantitative & Predictive Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

An institutional-grade, real-time quantitative dashboard designed to analyze and forecast financial asset volatility (default: XAUUSD / Bitcoin). This application merges algorithmic risk management, Natural Language Processing (NLP) sentiment analysis, and advanced machine learning architectures (Lasso & PyTorch LSTM) into a single unified terminal.

---

## 🚀 Live Terminal
*(https://dimas-xauusd-engine.streamlit.app/)*

---

## 🧠 Core Engine Architectures

This terminal operates on four distinct analytical layers:

### 1. Algorithmic Execution & Risk Management
* **Dynamic Position Sizing:** Automatically calculates capital allocation based on Average True Range (ATR) and defined risk percentage to preserve fund principal.
* **Crossover Logic:** Executes simulated Long/Short markers based on Fast and Slow Moving Average convergences.
* **Drawdown Matrix:** Tracks cumulative algorithmic returns against benchmark holding returns and monitors Maximum Drawdown metrics.

### 2. Real-Time NLP Sentiment Radar
* Scrapes live financial headlines using the Yahoo Finance API.
* Processes text through a pre-trained TF-IDF vectorizer and machine learning classification model to output real-time institutional market bias (**Bullish, Bearish, or Neutral**).

### 3. Predictive Machine Learning (Lasso Regression)
* Extracts 2 years of historical data and engineers lagged features (Lag 1, Lag 2, SMA 10, SMA 30).
* Employs Lasso Regression (L1 Regularization) to force optimal feature selection, aggressively penalizing irrelevant market noise to project the next day's closing price.

### 4. Deep Learning Forecaster (PyTorch LSTM)
* **Sequential Memory:** Utilizes a Long Short-Term Memory (LSTM) neural network to capture long-term non-linear dependencies in market volatility.
* **Tensor Computation:** Normalizes real-time market data, processes it through multi-layered LSTM gates, and performs out-of-sample tensor projections for future price movement.

---

## 🛠️ Tech Stack
* **Frontend:** Streamlit, Plotly, Seaborn (Bloomberg Terminal-inspired UI/UX)
* **Data Ingestion:** `yfinance`, Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Joblib
* **Deep Learning:** PyTorch (`torch`, `torch.nn`)

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/dimssrmdn01/gold-price-forecast-ml.git](https://github.com/dimssrmdn01/gold-price-forecast-ml.git)
   cd gold-price-forecast-ml
   
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
