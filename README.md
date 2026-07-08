## Institutional Asset Quantitative & Predictive Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

> **Empowering financial decisions with machine learning and real-time sentiment analysis.**
> 
> An institutional-grade, real-time quantitative dashboard designed to analyze and forecast financial asset volatility (default: **XAU/USD** and **Bitcoin**). This application merges algorithmic risk management, Natural Language Processing (NLP) sentiment analysis, and advanced machine learning architectures into a single unified terminal.

---

## Dashboard Preview

Here is a detailed look at the Institutional Quant Engine terminal interfaces:

### 1. Main Terminal Overview & Risk Management
<div align="center">
  <img src="https://github.com/user-attachments/assets/6b12f42e-eb23-4df4-b951-7cc7710c5067" alt="Main Terminal Overview">
  <br>
  <em>Figure 1: Main quantitative engine terminal displaying real-time asset prices, market volatility (ATR), and key risk/return metrics.</em>
</div>

---

### 2. Quantitative Execution History Chart
<div align="center">
  <img src="https://github.com/user-attachments/assets/f650cae2-a530-4c54-963b-7406657aae85" alt="Quantitative Execution Chart">
  <br>
  <em>Figure 2: Detailed historical price action chart with moving average crossovers and dynamic Long/Short execution markers.</em>
</div>

---

### 3. Global Macro Radar & Correlation Matrix
<div align="center">
  <img src="https://github.com/user-attachments/assets/6685dc36-a4b7-4999-9ab9-58a8d4d0bd5b" alt="Global Macro Radar">
  <br>
  <em>Figure 3: Global Macro Radar section showing asset correlation heatmaps and comparative performance analysis against major indices.</em>
</div>

---

### 4. Predictive Architectures (Lasso & LSTM)
<div align="center">
  <img src="https://github.com/user-attachments/assets/0cc9aa97-0e5b-48ac-82f5-2c43deddd972" alt="Predictive Architectures">
  <br>
  <em>Figure 4: Interface for machine learning models, displaying Lasso Regression feature selection and PyTorch LSTM deep learning price projections.</em>
</div>

---

### 5. Strategy Backtesting Results
<div align="center">
  <img src="https://github.com/user-attachments/assets/5bc999ec-4b3f-44bd-bd02-cd542b5e09fa" alt="Strategy Backtesting">
  <br>
  <em>Figure 5: 5-Year strategy backtest terminal showing equity curves, drawdown analysis, and trade-by-trade performance metrics.</em>
</div>

---

## Core Engine Architectures

This terminal operates on four distinct analytical layers, ensuring a comprehensive approach to market analysis:

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

## Project Structure
```
gold-price-forecast-ml/
├── Src/
│   ├── data_loader.py    # Ingests live XAUUSD/Crypto data via yfinance
│   ├── features.py       # Handles technical indicators & NLP sentiment logic
│   ├── train.py          # Trains the Lasso and PyTorch LSTM models
│   └── evaluate.py       # Backtesting execution and metrics evaluation
├── app.py                # Main Streamlit dashboard application
├── requirements.txt      # Project dependencies and libraries
└── README.md             # Project documentation
```
## Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Frontend / UI** | Streamlit, Plotly, Seaborn |
| **Data Ingestion** | `yfinance`, Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (Lasso Regression), Joblib |
| **Deep Learning** | PyTorch (`torch`, `torch.nn`), LSTM Architectures |
| **Natural Language Processing** | NLTK, TF-IDF Vectorization |

---

## Local Execution Guide

Follow these steps to deploy the terminal on your local machine:

**1. Clone the repository:**

```bash
git clone [https://github.com/dimssrmdn01/gold-price-forecast-ml.git]
cd gold-price-forecast-ml
```

**2. Initialize Virtual Environment (Recommended):**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Execute the pipeline sequentially:**

```bash
python Src/data_loader.py
python Src/features.py
python Src/train.py
python Src/evaluate.py
```

**5. Launch the Dashboard:**

```bash
streamlit run app.py
```

---
*Developed for advanced quantitative research and algorithmic trading simulations.*
