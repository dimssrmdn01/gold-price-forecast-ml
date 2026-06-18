import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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
# CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(page_title="Quant Engine", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-base: #050507;
    --panel-bg: rgba(15, 18, 25, 0.8);
    --gold: #D4AF37;
    --gold-glow: rgba(212, 175, 55, 0.15);
    --cyan: #00d2ff;
    --red: #ff3b30;
    --green: #34c759;
    --border: rgba(255, 255, 255, 0.08);
}

.stApp {
    background-color: var(--bg-base);
    background-image: radial-gradient(circle at 50% 0%, var(--gold-glow), transparent 40%);
    color: #e0e0e0;
}

.terminal-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    color: var(--gold);
    letter-spacing: 2px;
    margin-bottom: 0;
    text-shadow: 0 0 15px var(--gold-glow);
}

.terminal-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--cyan);
    letter-spacing: 1px;
    margin-bottom: 2rem;
}

div[data-testid="stMetric"] {
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-top: 3px solid var(--gold);
    padding: 1rem 1.5rem;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    transition: transform 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
}

div[data-testid="stMetric"] label {
    font-family: 'Space Mono', monospace;
    color: #8b949e;
    font-size: 0.8rem;
    text-transform: uppercase;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif;
    color: white;
    font-size: 2.5rem;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0a0c10;
    border-right: 1px solid var(--border);
}

hr {
    border-color: var(--border);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# MODEL LSTM
# -------------------------------------------------------------------
class XAUUSDForecasterLSTM(nn.Module):
    # Inisialisasi layer
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(XAUUSDForecasterLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    # Forward pass
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# -------------------------------------------------------------------
# SIDEBAR CONTROL
# -------------------------------------------------------------------
st.sidebar.markdown("<h2 style='font-family: Bebas Neue; color: #D4AF37;'> QUANT TERMINAL</h2>", unsafe_allow_html=True)
ticker = st.sidebar.text_input("Instrument Ticker", value="GC=F")
backtest_days = st.sidebar.slider("Historical Data (Days)", min_value=60, max_value=365, value=180)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters")
short_window = st.sidebar.number_input("Fast MA (Days)", min_value=5, max_value=30, value=20)
long_window = st.sidebar.number_input("Slow MA (Days)", min_value=31, max_value=100, value=50)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Manager")
account_capital = st.sidebar.number_input("Total Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
risk_percentage = st.sidebar.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
st.markdown(f'<h1 class="terminal-header">{ticker} QUANTITATIVE ENGINE</h1>', unsafe_allow_html=True)
st.markdown('<div class="terminal-sub"> SYSTEM STATUS: OPERATIONAL | PREDICTIVE ANALYTICS & EXECUTION FRAMEWORK </div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# DATA INGESTION
# -------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_institutional_data(symbol, days):
    # Tarik data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days + 100)
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Flatten kolom
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

try:
    df_raw = fetch_institutional_data(ticker, backtest_days)
    
    if df_raw.empty:
        # Error validasi
        st.error("Execution Terminated: Invalid ticker!")
    else:
        df = df_raw.copy()
        
        # Ekstrak fitur
        df['MA_Fast'] = df['Close'].rolling(window=short_window).mean()
        df['MA_Slow'] = df['Close'].rolling(window=long_window).mean()
        
        # Hitung volatilitas
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Hitung log-return
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df_filtered = df.tail(backtest_days).copy()
        
        # Algoritma crossover
        df_filtered['Signal'] = np.where(df_filtered['MA_Fast'] > df_filtered['MA_Slow'], 1, -1)
        df_filtered['Strategy_Return'] = df_filtered['Log_Return'] * df_filtered['Signal'].shift(1)
        
        # Marker sinyal
        df_filtered['Position_Changes'] = df_filtered['Signal'].diff()
        df_filtered['Buy_Markers'] = np.where(df_filtered['Position_Changes'] == 2, df_filtered['Close'], np.nan)
        df_filtered['Sell_Markers'] = np.where(df_filtered['Position_Changes'] == -2, df_filtered['Close'], np.nan)
        
        # Hitung metrik
        latest_price = float(df_filtered['Close'].iloc[-1])
        current_atr = float(df_filtered['ATR'].iloc[-1])
        asset_cum_return = (np.exp(df_filtered['Log_Return'].sum()) - 1) * 100
        strategy_cum_return = (np.exp(df_filtered['Strategy_Return'].sum()) - 1) * 100
        
        # Hitung drawdown
        strategy_cum_wealth = np.exp(df_filtered['Strategy_Return'].cumsum())
        peak = strategy_cum_wealth.cummax()
        drawdown = (strategy_cum_wealth - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Risk management
        cash_risk = account_capital * (risk_percentage / 100)
        stop_loss_distance = current_atr * 2 
        simulated_position_size = cash_risk / stop_loss_distance if stop_loss_distance > 0 else 0.0

        # UI Metrik
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spot Asset Price", f"${latest_price:,.2f}")
        m2.metric("Market Volatility (ATR)", f"${current_atr:.2f}")
        m3.metric("Algo Net Return", f"{strategy_cum_return:+.2f}%", delta=f"{strategy_cum_return - asset_cum_return:.2f}% vs Market")
        m4.metric("Max Algo Drawdown", f"{max_drawdown:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs([" Execution Chart", " Risk Sizing", " Raw Matrix"])
        
        with tab1:
            # Upgrade Plotly
            st.markdown("<h3 style='font-family: Bebas Neue; color: white;'>Quantitative Execution History</h3>", unsafe_allow_html=True)
            
            fig = go.Figure()
            # Plot harga
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Close'], name='Spot Price', line=dict(color='#D4AF37', width=2)))
            # Plot MA
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MA_Fast'], name=f'Fast MA', line=dict(color='#00D2FF', dash='dot')))
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MA_Slow'], name=f'Slow MA', line=dict(color='#FF3B30', dash='dot')))
            # Plot Sinyal
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Buy_Markers'], mode='markers', name='LONG', marker=dict(symbol='triangle-up', size=14, color='#34C759', line=dict(width=1, color='white'))))
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Sell_Markers'], mode='markers', name='SHORT', marker=dict(symbol='triangle-down', size=14, color='#FF3B30', line=dict(width=1, color='white'))))
            
            # Styling grafik
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#8b949e',
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Info risiko
            c1, c2 = st.columns(2)
            with c1:
                st.info(f" **Allowed Cash Risk:** ${cash_risk:,.2f} per trade")
                st.markdown(f"**Target Stop Loss:** ${stop_loss_distance:.2f} (2x ATR)")
            with c2:
                st.success(f" **SIMULATED SIZE:** {simulated_position_size:.3f} units")
                st.markdown(f"**Capital Allocation:** ${simulated_position_size * latest_price:,.2f}")

        with tab3:
            # Data raw
            st.dataframe(df_filtered[['Close', 'MA_Fast', 'MA_Slow', 'ATR', 'Strategy_Return', 'Signal']].tail(15), use_container_width=True)
            
except Exception as e:
    st.error(f"System Failure: {str(e)}")

# -------------------------------------------------------------------
# MACRO RADAR
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: white;'>❖ Global Macro Radar</h2>", unsafe_allow_html=True)

with st.spinner("Sinkronisasi data makro..."):
    try:
        # Kamus instrumen
        macro_basket = {
            "Primary": ticker,
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "USD Index": "DX-Y.NYB"
        }
        
        macro_data = pd.DataFrame()
        # Looping tarikan
        for name, sym in macro_basket.items():
            temp_df = yf.download(sym, period=f"{backtest_days}d", progress=False)
            if not temp_df.empty and 'Close' in temp_df.columns:
                macro_data[name] = temp_df['Close'].squeeze()
            
        macro_data.dropna(inplace=True)
        corr_matrix = macro_data.corr()
        
        col_macro1, col_macro2 = st.columns([1, 2], gap="large")
        
        with col_macro1:
            st.markdown("<span style='font-family: Space Mono; color: #8b949e;'>MATRIKS KORELASI</span>", unsafe_allow_html=True)
            # Heatmap korelasi
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}", showscale=False
            ))
            fig_corr.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_macro2:
            st.markdown("<span style='font-family: Space Mono; color: #8b949e;'>PERBANDINGAN KINERJA (BASE 100)</span>", unsafe_allow_html=True)
            normalized_data = (macro_data / macro_data.iloc[0]) * 100
            fig_line = go.Figure()
            
            # Plot garis
            for col in normalized_data.columns:
                width = 3 if col == "Primary" else 1.5
                dash = 'solid' if col == "Primary" else 'dot'
                fig_line.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data[col], mode='lines', name=col, line=dict(width=width, dash=dash)))
                
            fig_line.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig_line, use_container_width=True)
            
    except Exception as e:
        st.error(f"Gagal memuat radar: {str(e)}")

# -------------------------------------------------------------------
# ML & DL ENGINES
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: white;'>❖ Predictive Architectures</h2>", unsafe_allow_html=True)

col_ai1, col_ai2 = st.columns(2, gap="large")

with col_ai1:
    # Kontainer Lasso
    with st.container(border=True):
        st.markdown("<h3 style='font-family: Space Mono; color: #00d2ff;'>⟁ Lasso Regression</h3>", unsafe_allow_html=True)
        with st.spinner("Konfigurasi ML..."):
            try:
                hist = yf.Ticker(ticker).history(period="2y")
                if not hist.empty:
                    df_ml = pd.DataFrame()
                    df_ml['Close'] = hist['Close']
                    df_ml['Lag_1'] = df_ml['Close'].shift(1)
                    df_ml['Lag_2'] = df_ml['Close'].shift(2)
                    df_ml['SMA_10'] = df_ml['Close'].rolling(window=10).mean()
                    df_ml['SMA_30'] = df_ml['Close'].rolling(window=30).mean()
                    df_ml.dropna(inplace=True)
                    
                    # Split data
                    X = df_ml[['Lag_1', 'Lag_2', 'SMA_10', 'SMA_30']]
                    y = df_ml['Close']
                    split_idx = int(len(df_ml) * 0.8)
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    
                    # Train model
                    model_ml = Lasso(alpha=0.1)
                    model_ml.fit(X_train, y_train)
                    
                    # Evaluasi
                    predictions = model_ml.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    next_day_pred = model_ml.predict(X.iloc[-1].values.reshape(1, -1))[0]
                    
                    l1, l2 = st.columns(2)
                    l1.metric("Prediksi (T+1)", f"${next_day_pred:,.2f}")
                    l2.metric("RMSE Akurasi", f"${rmse:,.2f}")
            except Exception as e:
                st.error(f"ML Error: {e}")

with col_ai2:
    # Kontainer PyTorch
    with st.container(border=True):
        st.markdown("<h3 style='font-family: Space Mono; color: #b026ff;'>⚡ PyTorch LSTM Core</h3>", unsafe_allow_html=True)
        if st.button("INITIALIZE TENSOR COMPUTATION", use_container_width=True):
            with st.spinner("Iterasi pelatihan..."):
                try:
                    seq_length = 10
                    raw_dl = yf.download(ticker, period="2y", progress=False)
                    
                    # Skala data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(raw_dl[['Close']].values)
                    
                    # Buat sekuens
                    X_dl, y_dl = [], []
                    for i in range(len(scaled_data) - seq_length):
                        X_dl.append(scaled_data[i:(i + seq_length), 0])
                        y_dl.append(scaled_data[i + seq_length, 0]) 
                        
                    # Konversi tensor
                    X_tensor = torch.FloatTensor(np.array(X_dl).reshape(-1, seq_length, 1))
                    y_tensor = torch.FloatTensor(np.array(y_dl).reshape(-1, 1))
                    
                    # Train LSTM
                    model_dl = XAUUSDForecasterLSTM()
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model_dl.parameters(), lr=0.01)
                    
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
                        
                    lstm_pred = scaler.inverse_transform(pred_scaled.numpy())[0][0]
                    lstm_actual = raw_dl['Close'].iloc[-1].item()
                    
                    d1, d2 = st.columns(2)
                    d1.metric("Harga Aktual", f"${lstm_actual:,.2f}")
                    d2.metric("Proyeksi LSTM", f"${lstm_pred:,.2f}", f"{lstm_pred - lstm_actual:+.2f} USD")
                except Exception as e:
                    st.error(f"PyTorch Error: {e}")

# -------------------------------------------------------------------
# BACKTESTING (VECTORBT)
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: white;'>❖ 5-Year Strategy Backtest</h2>", unsafe_allow_html=True)

with st.spinner("Komputasi historis..."):
    try:
        import vectorbt as vbt
        
        # Tarik data
        bt_data = yf.download(ticker, period="5y", progress=False)
        if isinstance(bt_data.columns, pd.MultiIndex):
            bt_data.columns = bt_data.columns.droplevel(1)
        price_series = bt_data['Close']
            
        # Sinyal vektor
        fast_ma = vbt.MA.run(price_series, short_window)
        slow_ma = vbt.MA.run(price_series, long_window)
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Eksekusi simulasi
        portfolio = vbt.Portfolio.from_signals(price_series, entries, exits, init_cash=account_capital, fees=0.001)
        
        # Metrik
        col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
        col_bt1.metric("Total Kuantitatif", f"{portfolio.total_return() * 100:.2f}%")
        col_bt2.metric("Net Profit", f"${portfolio.total_profit():,.2f}")
        col_bt3.metric("Win Rate", f"{portfolio.trades.win_rate() * 100:.2f}%")
        col_bt4.metric("Max Drawdown", f"{portfolio.max_drawdown() * 100:.2f}%")
        
        # Plot
        fig_bt = portfolio.plot()
        fig_bt.update_layout(height=500, template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bt, use_container_width=True)
        
    except ImportError:
        st.warning("Install 'vectorbt' untuk fitur ini!")
    except Exception as e:
        st.error(f"Backtest Error: {e}")
