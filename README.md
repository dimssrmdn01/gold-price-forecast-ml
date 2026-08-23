<div align="center">

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=24&pause=1000&color=EE4C2C&center=true&vCenter=true&width=700&lines=Institutional+Quant+Engine;Powered+by+PyTorch+%26+Lasso;Walk-Forward+Backtesting+%26+Agentic+AI;Real-Time+XAU%2FUSD+Forecasting)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white)
![vectorbt](https://img.shields.io/badge/vectorbt-000000?style=for-the-badge&logoColor=white)

</div>

<br>

> Empowering financial decisions through hybrid machine learning, robust backtesting, and Agentic AI.
>
> An institutional-grade quantitative dashboard for analyzing and forecasting financial asset volatility (default: XAU/USD, extensible to any Yahoo Finance ticker). This system merges strict machine learning validation, walk-forward algorithmic backtesting, and an autonomous LLM agent capable of real-time tool calling for fundamental and technical market analysis.

##  Key Architectural Upgrades

- **Agentic LLM Integration.** Groq API (LLaMA-3.3-70B) with strict function-calling: the agent can pull live ML projections, backtest metrics, and real-time market news (via DuckDuckGo Search) on its own, with an explicit guardrail against fabricating numbers when a tool hasn't been run yet.
- **Walk-Forward Validation.** The `vectorbt` backtest engine chunks 5 years of historical data into rolling 1-year windows, reporting win rate, return, and max drawdown per period so strategy robustness is checked across bull, bear, and sideways regimes, not just one lucky window.
- **Data Leakage Prevention.** Both the Lasso and PyTorch LSTM pipelines use strict train/test splits the LSTM's `MinMaxScaler` is fit only on the training slice, never on the full dataset, eliminating forward-looking bias.
- **Confidence Intervals & Model Persistence.** Lasso predictions ship with a 95% confidence interval (Z = 1.96) derived from test-set RMSE. Trained models are cached to disk via `joblib` to skip redundant retraining.
- **Live Signal Alerts.** A toast notification fires the moment a new MA crossover (Buy/Sell) is detected on the latest candle.
- **Explainability Dashboard.** Lasso feature-importance coefficients and a head-to-head RMSE comparison (Lasso vs. LSTM) are rendered directly in the UI.

## Dashboard Preview

<div align="center">
  <img src="https://github.com/user-attachments/assets/6b12f42e-eb23-4df4-b951-7cc7710c5067" alt="Main Terminal Overview">
  <br><em>Figure 1: Main terminal real-time price, volatility (ATR), and risk/return metrics.</em>
</div>

<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/0cc9aa97-0e5b-48ac-82f5-2c43deddd972" alt="Predictive Architectures">
  <br><em>Figure 2: Lasso feature selection alongside PyTorch LSTM projections, both with train/test RMSE.</em>
</div>

<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/5bc999ec-4b3f-44bd-bd02-cd542b5e09fa" alt="Walk-Forward Backtesting">
  <br><em>Figure 3: Walk-forward backtest equity curve plus a year-by-year robustness matrix.</em>
</div>

<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/6685dc36-a4b7-4999-9ab9-58a8d4d0bd5b" alt="Global Macro Radar">
  <br><em>Figure 4: Cross-asset correlation heatmap and normalized performance vs. major indices.</em>
</div>

> Screenshots reflect the Cyberpunk Red theme. Update these image links once you capture fresh screenshots of the current UI (Neural Agent panel included).

## Core Engine Architectures

**1. Algorithmic Execution & Risk Management**
Dynamic position sizing from ATR and a configurable risk percentage. Fast/Slow MA crossover logic drives simulated Long/Short markers, with a live toast alert the moment a new signal fires on the latest candle.

**2. Walk-Forward Backtesting (vectorbt)**
Goes beyond a single static backtest: five years of history are sliced into rolling annual windows, each independently scored on win rate, return, and max drawdown, so the strategy's robustness across different market regimes is visible at a glance.

**3. Predictive Machine Learning (Lasso Regression)**
Lagged features (Lag 1, Lag 2, SMA 10, SMA 30) feed an L1-regularized Lasso model with a proper train/test split. Outputs a next-day price projection with a 95% confidence interval, RMSE, and per-feature coefficient importance. Trained models persist to disk via `joblib`.

**4. Deep Learning Forecaster (PyTorch LSTM)**
A sequential LSTM network captures non-linear dependencies in price action. The scaler is fit strictly on the training split to prevent leakage, and test-set RMSE is reported alongside the Lasso model for direct comparison.

**5. Neural Agent Interface (LLM)**
LLaMA-3.3-70B via Groq, with three callable tools: `get_predictions` (latest Lasso/LSTM output), `get_backtest` (walk-forward portfolio metrics), and `get_market_news` (live headlines via DuckDuckGo Search). The system prompt explicitly forbids the agent from inventing numbers when a tool hasn't been run yet.

**6. Stochastic Risk Simulation (Monte Carlo)**
Runs 500 simulated price paths 30 days ahead from historical volatility, reporting 95% and 99% confidence bounds on the projected price.

## Project Structure

````text
gold-price-forecast-ml/
├── app.py              # Main Streamlit app — data ingestion, ML training, backtesting, and the LLM agent all live here
├── monte_carlo.py       # Monte Carlo simulation + risk metric helpers
├── requirements.txt
└── README.md
````

`models/` is created automatically at runtime to cache trained Lasso models via `joblib` it's not part of the repo and should stay in `.gitignore`.

## Technology Stack

| Category | Technologies |
| :--- | :--- |
| Frontend / Visualization | Streamlit, Plotly (`plotly_dark` theme) |
| Data Ingestion | `yfinance`, Pandas, NumPy |
| Quantitative Backtesting | `vectorbt` (walk-forward validation) |
| Machine Learning | Scikit-Learn (Lasso Regression), `joblib` (model persistence) |
| Deep Learning | PyTorch (`torch`, `torch.nn`), LSTM |
| Agentic AI & Search | Groq API (LLaMA-3.3-70B), `duckduckgo-search` |

## Local Execution Guide

**1. Clone the repository**
````bash
git clone https://github.com/dimssrmdn01/gold-price-forecast-ml.git
cd gold-price-forecast-ml
````

**2. Initialize a virtual environment (recommended)**
````bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate
````

**3. Install dependencies**
````bash
pip install -r requirements.txt
````

**4. Configure the AI Agent**

A Groq API key is required to power the Neural Agent Interface. Enter it directly into the Streamlit sidebar after launching the app no `.env` file needed.

**5. Launch the dashboard**
````bash
streamlit run app.py
````

*Developed for advanced quantitative research, machine learning architecture validation, and algorithmic trading simulations.*
