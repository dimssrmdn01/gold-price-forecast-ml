# XAUUSD Price Prediction – ML Trading System

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

---

## Project Structure
xauusd-price-prediction/
│
├── data/ # Raw & processed datasets
│ ├── raw/ # Original downloaded data
│ └── processed/ # Cleaned & feature-engineered data
│
├── notebooks/ # Jupyter notebooks for EDA & experimentation
│ ├── 01_EDA.ipynb # Exploratory Data Analysis
│ ├── 02_Feature_Engineering.ipynb
│ └── 03_Model_Training.ipynb
│
├── src/ # Production-ready source code
│ ├── data_pipeline/ # Data fetching & preprocessing
│ ├── features/ # Technical indicators
│ ├── models/ # ML model definitions
│ └── backtest/ # Trading simulation engine
│
├── models/ # Saved trained models
│ ├── xgboost_model.pkl
│ └── lstm_weights.h5
│
├── results/ # Output & visualizations
│ ├── plots/ # Performance charts
│ ├── logs/ # Training logs
│ └── predictions.csv # Model predictions
│
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
└── README.md # This file

Disclaimer
This project is for educational and research purposes only.
Trading financial markets carries substantial risk. The authors assume no responsibility for financial losses incurred using this system.

Contact
Author: Dimas Ramadhan
GitHub: @dimssrmdn01
Project Link: https://github.com/dimssrmdn01/xauusd-ml-prediction


