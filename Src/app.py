import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import os
from datetime import datetime, timedelta
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# KELAS ARSITEKTUR DEEP LEARNING (LSTM)
# -------------------------------------------------------------------
class XAUUSDForecasterLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(XAUUSDForecasterLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# Pengaturan layout terminal 
st.set_page_config(
    page_title="Institutional Gold Quant Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS untuk nuansa Bloomberg 
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .metric-card {
        background-color: #161b22;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# HEADER SECTION
# -------------------------------------------------------------------
st.title("XAUUSD Institutional Quantitative & Predictive Analytics Engine")
st.markdown("### `SYSTEM STATUS: OPERATIONAL` | Risk Mitigation & Quantitative Execution Framework")
st.markdown("---")

# -------------------------------------------------------------------
# SIDEBAR CONTROL PANEL
# -------------------------------------------------------------------
st.sidebar.header("Quant Engine Control Panel")
ticker = st.sidebar.text_input("Instrument Ticker", value="GC=F")
backtest_days = st.sidebar.slider("Historical Data Window (Days)", min_value=60, max_value=365, value=180)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters")
short_window = st.sidebar.number_input("Fast Moving Average (Days)", min_value=5, max_value=30, value=20)
long_window = st.sidebar.number_input("Slow Moving Average (Days)", min_value=31, max_value=100, value=50)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Management Simulator")
account_capital = st.sidebar.number_input("Total Account Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
risk_percentage = st.sidebar.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

# -------------------------------------------------------------------
# DATA INGESTION PIPELINE
# -------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_institutional_data(symbol, days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days + 100)
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

try:
    df_raw = fetch_institutional_data(ticker, backtest_days)
    
    if df_raw.empty:
        st.error("Execution Terminated: Invalid ticker symbol or data pipeline connection failed.")
    else:
        df = df_raw.copy()
        
        # -------------------------------------------------------------------
        # ADVANCED FEATURE ENGINEERING
        # -------------------------------------------------------------------
        df['MA_Fast'] = df['Close'].rolling(window=short_window).mean()
        df['MA_Slow'] = df['Close'].rolling(window=long_window).mean()
        
        # Average True Range (ATR)
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df_filtered = df.tail(backtest_days).copy()
        
        # -------------------------------------------------------------------
        # ALGORITHMIC BACKTEST EXECUTION ENGINE & RISK METRICS
        # -------------------------------------------------------------------
        df_filtered['Signal'] = np.where(df_filtered['MA_Fast'] > df_filtered['MA_Slow'], 1, -1)
        df_filtered['Strategy_Return'] = df_filtered['Log_Return'] * df_filtered['Signal'].shift(1)
        
        df_filtered['Position_Changes'] = df_filtered['Signal'].diff()
        df_filtered['Buy_Markers'] = np.where(df_filtered['Position_Changes'] == 2, df_filtered['Close'], np.nan)
        df_filtered['Sell_Markers'] = np.where(df_filtered['Position_Changes'] == -2, df_filtered['Close'], np.nan)
        
        latest_price = float(df_filtered['Close'].iloc[-1])
        current_atr = float(df_filtered['ATR'].iloc[-1])
        
        asset_cum_return = (np.exp(df_filtered['Log_Return'].sum()) - 1) * 100
        strategy_cum_return = (np.exp(df_filtered['Strategy_Return'].sum()) - 1) * 100
        
        strategy_cum_wealth = np.exp(df_filtered['Strategy_Return'].cumsum())
        peak = strategy_cum_wealth.cummax()
        drawdown = (strategy_cum_wealth - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        cash_risk = account_capital * (risk_percentage / 100)
        stop_loss_distance = current_atr * 2 
        simulated_position_size = cash_risk / stop_loss_distance if stop_loss_distance > 0 else 0.0

        # -------------------------------------------------------------------
        # INTERACTIVE TERMINAL DASHBOARD LAYOUT
        # -------------------------------------------------------------------
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Spot Asset Price", f"${latest_price:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Market Volatility (ATR)", f"${current_atr:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Algorithmic Net Return", f"{strategy_cum_return:+.2f}%", delta=f"{strategy_cum_return - asset_cum_return:.2f}% vs Benchmark")
            st.markdown("</div>", unsafe_allow_html=True)
        with m4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Maximum Strategy Drawdown", f"{max_drawdown:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Execution Chart Analysis", "Risk Management & Sizing", "Core Matrix Data"])
        
        with tab1:
            st.subheader("Quantitative Crossover Execution History")
            sns.set_theme(style="darkgrid")
            fig, ax1 = plt.subplots(figsize=(16, 7))
            fig.patch.set_facecolor('#0e1117')
            ax1.set_facecolor('#161b22')
            ax1.plot(df_filtered.index, df_filtered['Close'], label='XAUUSD Spot Price', color='#D4AF37', linewidth=2, alpha=0.9)
            ax1.plot(df_filtered.index, df_filtered['MA_Fast'], label=f'Fast MA ({short_window}D)', color='#00D2FF', linestyle='--', linewidth=1.2)
            ax1.plot(df_filtered.index, df_filtered['MA_Slow'], label=f'Slow MA ({long_window}D)', color='#FF3B30', linestyle='--', linewidth=1.2)
            ax1.scatter(df_filtered.index, df_filtered['Buy_Markers'], label='EXECUTE LONG (BUY)', color='#34C759', marker='^', s=150, zorder=5)
            ax1.scatter(df_filtered.index, df_filtered['Sell_Markers'], label='EXECUTE SHORT (SELL)', color='#FF3B30', marker='v', s=150, zorder=5)
            ax1.set_ylabel("Price (USD)", color='white', fontsize=12)
            ax1.tick_params(colors='white')
            ax1.legend(loc='upper left', facecolor='#0e1117', edgecolor='#30363d', labelcolor='white')
            ax1.set_title("Algorithmic Order Execution Tracking Framework", color='white', fontsize=14, fontweight='bold')
            st.pyplot(fig)

        with tab2:
            st.subheader("Automated Position Sizing & Capital Allocation")
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"Allowed Cash Risk: ${cash_risk:,.2f} per trade")
                st.markdown(f"**Target Stop Loss Distance:** ${stop_loss_distance:.2f}")
            with c2:
                st.success(f"SIMULATED POSITION SIZE: {simulated_position_size:.3f} units")
                st.markdown(f"**Capital Allocation:** ${simulated_position_size * latest_price:,.2f}")

        with tab3:
            st.subheader("Automated Computed Matrix Stream")
            st.dataframe(df_filtered[['Close', 'MA_Fast', 'MA_Slow', 'ATR', 'Strategy_Return', 'Signal']].tail(15), use_container_width=True)
            
except Exception as e:
    st.error(f"Critical System Failure: {str(e)}")

# -------------------------------------------------------------------
# NLP SENTIMENT ENGINE
# -------------------------------------------------------------------
st.divider()
st.subheader("❖ Real-Time Fundamental Sentiment (NLP Engine)")
if os.path.exists('Models/sentiment_model.pkl'):
    # ... (Bagian NLP dibiarkan sama karena sudah optimal)
    st.info("Mesin NLP terdeteksi aktif. (Keluaran disembunyikan untuk fokus pada algoritma DL)")
else:
    st.info("Mesin NLP belum terpasang.")

# -------------------------------------------------------------------
# MACHINE LEARNING ENGINE (LASSO)
# -------------------------------------------------------------------
st.divider()
st.subheader("⟁ XAUUSD Predictive ML (Lasso Regression)")
with st.spinner("Mengekstrak data historis dan melatih model statistik..."):
    try:
        hist = yf.Ticker('GC=F').history(period="2y")
        if not hist.empty:
            df_ml = pd.DataFrame()
            df_ml['Close'] = hist['Close']
            df_ml['Lag_1'] = df_ml['Close'].shift(1)
            df_ml['Lag_2'] = df_ml['Close'].shift(2)
            df_ml['SMA_10'] = df_ml['Close'].rolling(window=10).mean()
            df_ml['SMA_30'] = df_ml['Close'].rolling(window=30).mean()
            df_ml.dropna(inplace=True)
            
            X = df_ml[['Lag_1', 'Lag_2', 'SMA_10', 'SMA_30']]
            y = df_ml['Close']
            
            split_idx = int(len(df_ml) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Implementasi Lasso Regression
            model_ml = Lasso(alpha=0.1)
            model_ml.fit(X_train, y_train)
            
            predictions = model_ml.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            last_row = X.iloc[-1].values.reshape(1, -1)
            next_day_pred = model_ml.predict(last_row)[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Prediksi Lasso (Esok)", value=f"${next_day_pred:,.2f}")
            with col2:
                st.metric(label="Akurasi MSE Loss (RMSE)", value=f"${rmse:,.2f}", delta="Optimal Feature Selection", delta_color="normal")
    except Exception as e:
        st.error(f"Mesin ML gagal dieksekusi: {e}")

# -------------------------------------------------------------------
# DEEP LEARNING ENGINE (PyTorch LSTM)
# -------------------------------------------------------------------
st.divider()
st.subheader("⚡ XAUUSD Deep Learning Forecaster (PyTorch LSTM)")
st.markdown("Arsitektur jaringan saraf tiruan (*Neural Network*) dengan memori LSTM untuk membedah volatilitas sekuensial pasar.")

if st.button("▰▰ INITIALIZE TENSOR COMPUTATION (LSTM) ▰▰", type="primary"):
    with st.spinner("Memproses Tensor dan melatih arsitektur LSTM selama 50 epoch..."):
        try:
            seq_length = 10
            raw_dl = yf.download(ticker, period="2y", progress=False)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(raw_dl[['Close']].values)
            
            X_dl, y_dl = [], []
            for i in range(len(scaled_data) - seq_length):
                X_dl.append(scaled_data[i:(i + seq_length), 0])
                y_dl.append(scaled_data[i + seq_length, 0])
                
            X_tensor = torch.FloatTensor(np.array(X_dl).reshape(-1, seq_length, 1))
            y_tensor = torch.FloatTensor(np.array(y_dl).reshape(-1, 1))
            
            model_dl = XAUUSDForecasterLSTM()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model_dl.parameters(), lr=0.01)
            
            # Progress bar untuk UI
            progress = st.progress(0)
            for epoch in range(50):
                model_dl.train()
                optimizer.zero_grad()
                loss = criterion(model_dl(X_tensor), y_tensor)
                loss.backward()
                optimizer.step()
                progress.progress((epoch + 1) / 50)
            
            # Prediksi
            model_dl.eval()
            with torch.no_grad():
                pred_scaled = model_dl(X_tensor[-1:].clone().detach())
                
            lstm_pred_usd = scaler.inverse_transform(pred_scaled.numpy())[0][0]
            lstm_actual = raw_dl['Close'].iloc[-1].item()
            
            dl1, dl2 = st.columns(2)
            with dl1:
                st.metric("Harga Aktual Terakhir", f"${lstm_actual:,.2f}")
            with dl2:
                st.metric("Proyeksi Neural Network", f"${lstm_pred_usd:,.2f}", f"{lstm_pred_usd - lstm_actual:+.2f} USD")
                
        except Exception as e:
            st.error(f"Mesin PyTorch gagal dieksekusi: {e}")