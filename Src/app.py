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

# Import module
from monte_carlo import run_monte_carlo, calculate_risk_metrics

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# SETUP CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="Quant Engine | Cyberpunk Ed.", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (CYBERPUNK RED THEME)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-base: #08080a;
    --panel-bg: rgba(18, 5, 5, 0.75);
    --neon-red: #FF003C;
    --red-glow: rgba(255, 0, 60, 0.35);
    --neon-yellow: #FCEE0A;
    --cyan: #00F0FF;
    --green: #00FF66;
    --border: rgba(255, 0, 60, 0.4);
}

.stApp {
    background-color: var(--bg-base);
    background-image: 
        linear-gradient(rgba(255, 0, 60, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 0, 60, 0.04) 1px, transparent 1px);
    background-size: 35px 35px;
    color: #e0e0e0;
}

.terminal-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    color: var(--neon-red);
    letter-spacing: 5px;
    margin-bottom: 0;
    text-shadow: 3px 0px 0px var(--cyan), -3px 0px 0px var(--neon-yellow), 0 0 25px var(--red-glow);
    text-transform: uppercase;
}

.terminal-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    color: var(--neon-yellow);
    letter-spacing: 3px;
    margin-bottom: 2rem;
    border-left: 4px solid var(--neon-red);
    padding-left: 12px;
    text-shadow: 0 0 8px rgba(252, 238, 10, 0.5);
    background: linear-gradient(90deg, rgba(255,0,60,0.15) 0%, transparent 100%);
}

div[data-testid="stMetric"] {
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-top: 4px solid var(--neon-red);
    padding: 1rem 1.5rem;
    border-radius: 2px;
    box-shadow: 0 0 15px rgba(255, 0, 60, 0.1);
    transition: all 0.3s ease;
    backdrop-filter: blur(4px);
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 25px var(--red-glow);
    border-color: var(--neon-red);
}

div[data-testid="stMetric"] label {
    font-family: 'Space Mono', monospace;
    color: #ff7597;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif;
    color: white;
    font-size: 2.8rem;
    text-shadow: 0 0 12px rgba(255, 255, 255, 0.4);
}

[data-testid="stSidebar"] {
    background-color: #050505;
    border-right: 1px solid var(--border);
}

hr {
    border-color: var(--border);
}

.stTabs [data-baseweb="tab-list"] { gap: 24px; }
.stTabs [data-baseweb="tab"] { color: #ff7597; font-family: 'Space Mono', monospace; }
.stTabs [aria-selected="true"] {
    color: var(--neon-red) !important;
    border-bottom-color: var(--neon-red) !important;
    text-shadow: 0 0 10px var(--red-glow);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# MODEL LSTM
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

# -------------------------------------------------------------------
# SIDEBAR CONTROL
# -------------------------------------------------------------------
st.sidebar.markdown("<h2 style='font-family: Bebas Neue; color: #FF003C; letter-spacing: 2px; text-shadow: 0 0 15px rgba(255,0,60,0.6);'> ✦ SYS_CONTROL</h2>", unsafe_allow_html=True)
ticker = st.sidebar.text_input("Instrument Ticker", value="GC=F")
backtest_days = st.sidebar.slider("Historical Data (Days)", min_value=60, max_value=365, value=180)

st.sidebar.markdown("---")
st.sidebar.markdown("<span style='font-family: Space Mono; color: #FCEE0A;'>⚙️ PARAMETERS</span>", unsafe_allow_html=True)
short_window = st.sidebar.number_input("Fast MA", min_value=5, max_value=30, value=20)
long_window = st.sidebar.number_input("Slow MA", min_value=31, max_value=100, value=50)

st.sidebar.markdown("---")
st.sidebar.markdown("<span style='font-family: Space Mono; color: #FF003C;'>🛡️ SECURITY / RISK</span>", unsafe_allow_html=True)
account_capital = st.sidebar.number_input("Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
risk_percentage = st.sidebar.slider("Risk (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

# -------------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------------
st.markdown(f'<h1 class="terminal-header">{ticker} QUANT_ENGINE</h1>', unsafe_allow_html=True)
st.markdown('<div class="terminal-sub"> [SYS.ONLINE] ✦ PREDICTIVE NEURAL NETWORK ACTIVATED </div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# DATA INGESTION
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
        st.error("Execution Terminated: No Data Found!")
    else:
        df = df_raw.copy()
        
        df['MA_Fast'] = df['Close'].rolling(window=short_window).mean()
        df['MA_Slow'] = df['Close'].rolling(window=long_window).mean()
        
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df_filtered = df.tail(backtest_days).copy()
        
        df_filtered['Signal'] = np.where(df_filtered['MA_Fast'] > df_filtered['MA_Slow'], 1, -1)
        df_filtered['Strategy_Return'] = df_filtered['Log_Return'] * df_filtered['Signal'].shift(1)
        
        df_filtered['Position_Changes'] = df_filtered['Signal'].diff()
        df_filtered['Buy_Markers'] = np.where(df_filtered['Position_Changes'] == 2, df_filtered['Close'], np.nan)
        df_filtered['Sell_Markers'] = np.where(df_filtered['Position_Changes'] == -2, df_filtered['Close'], np.nan)
        
        latest_price = float(df_filtered['Close'].iloc[-1])
        current_atr = float(df_filtered['ATR'].iloc[-1])
        
        # --- 🚨 LIVE ALERT SYSTEM ---
        last_signal_change = df_filtered['Position_Changes'].iloc[-1]
        if last_signal_change == 2:
            st.toast(f"ALGO DETECTED: STRONG BUY Signal at ${latest_price:,.2f}!", icon="🟢")
        elif last_signal_change == -2:
            st.toast(f"ALGO DETECTED: STRONG SELL Signal at ${latest_price:,.2f}!", icon="🔴")
        # ----------------------------

        asset_cum_return = (np.exp(df_filtered['Log_Return'].sum()) - 1) * 100
        strategy_cum_return = (np.exp(df_filtered['Strategy_Return'].sum()) - 1) * 100
        
        strategy_cum_wealth = np.exp(df_filtered['Strategy_Return'].cumsum())
        peak = strategy_cum_wealth.cummax()
        drawdown = (strategy_cum_wealth - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        cash_risk = account_capital * (risk_percentage / 100)
        stop_loss_distance = current_atr * 2 
        simulated_position_size = cash_risk / stop_loss_distance if stop_loss_distance > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spot Price", f"${latest_price:,.2f}")
        m2.metric("Volatility (ATR)", f"${current_atr:.2f}")
        m3.metric("Algo Return", f"{strategy_cum_return:+.2f}%", delta=f"{strategy_cum_return - asset_cum_return:.2f}%")
        m4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["[CHART_DATA]", "[RISK_SIZE]", "[RAW_MATRIX]"])
        
        with tab1:
            st.markdown("<h3 style='font-family: Bebas Neue; color: white;'>EXECUTION HISTORY</h3>", unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Close'], name='Spot', line=dict(color='#FF003C', width=2.5)))
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MA_Fast'], name='Fast MA', line=dict(color='#FCEE0A', dash='dot')))
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MA_Slow'], name='Slow MA', line=dict(color='#00F0FF', dash='dot')))
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Buy_Markers'], mode='markers', name='LONG', marker=dict(symbol='triangle-up', size=14, color='#00FF66', line=dict(width=1, color='white'))))
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Sell_Markers'], mode='markers', name='SHORT', marker=dict(symbol='triangle-down', size=14, color='#FF003C', line=dict(width=1, color='white'))))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,0,60,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,0,60,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.info(f" ✦ Cash Risk: ${cash_risk:,.2f}")
                st.markdown(f"✦ Stop Loss: ${stop_loss_distance:.2f}")
            with c2:
                st.success(f" ✦ SIMULATED: {simulated_position_size:.3f} units")
                st.markdown(f"✦ Allocation: ${simulated_position_size * latest_price:,.2f}")

        with tab3:
            st.dataframe(df_filtered[['Close', 'MA_Fast', 'MA_Slow', 'ATR', 'Strategy_Return', 'Signal']].tail(15), use_container_width=True)
            
except Exception as e:
    st.error(f"System Failure: {str(e)}")

# -------------------------------------------------------------------
# MACRO RADAR
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: #FF003C; text-shadow: 0 0 10px rgba(255,0,60,0.5);'> GLOBAL MACRO RADAR</h2>", unsafe_allow_html=True)

with st.spinner("Sinkronisasi data..."):
    try:
        macro_basket = {
            "Primary": ticker,
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "USD Index": "DX-Y.NYB"
        }
        
        macro_data = pd.DataFrame()
        for name, sym in macro_basket.items():
            temp_df = yf.download(sym, period=f"{backtest_days}d", progress=False)
            if not temp_df.empty and 'Close' in temp_df.columns:
                macro_data[name] = temp_df['Close'].squeeze()
            
        macro_data.dropna(inplace=True)
        corr_matrix = macro_data.corr()
        
        col_macro1, col_macro2 = st.columns([1, 2], gap="large")
        
        with col_macro1:
            st.markdown("<span style='font-family: Space Mono; color: #00F0FF;'>[MATRIKS KORELASI]</span>", unsafe_allow_html=True)
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}", showscale=False
            ))
            fig_corr.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_macro2:
            st.markdown("<span style='font-family: Space Mono; color: #FCEE0A;'>[KINERJA BASE 100]</span>", unsafe_allow_html=True)
            normalized_data = (macro_data / macro_data.iloc[0]) * 100
            fig_line = go.Figure()
            
            colors = ["#FF003C", "#00F0FF", "#FCEE0A", "#00FF66"]
            for i, col in enumerate(normalized_data.columns):
                width = 3.5 if col == "Primary" else 1.5
                dash = 'solid' if col == "Primary" else 'dot'
                fig_line.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data[col], mode='lines', name=col, line=dict(width=width, dash=dash, color=colors[i%len(colors)])))
                
            fig_line.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', xaxis=dict(showgrid=True, gridcolor='rgba(255,0,60,0.1)'), yaxis=dict(showgrid=True, gridcolor='rgba(255,0,60,0.1)'))
            st.plotly_chart(fig_line, use_container_width=True)
            
    except Exception as e:
        st.error(f"Gagal radar: {str(e)}")

# -------------------------------------------------------------------
# ML ENGINES & CACHING
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: white;'> PREDICTIVE ARCHITECTURES</h2>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def train_lasso_model(ticker_symbol):
    hist = yf.Ticker(ticker_symbol).history(period="2y")
    if hist.empty: return None, None, None
    
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
    
    model_ml = Lasso(alpha=0.1)
    model_ml.fit(X_train, y_train)
    
    predictions = model_ml.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    next_day_pred = model_ml.predict(X.iloc[-1].values.reshape(1, -1))[0]
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model_ml.coef_
    })
    
    return next_day_pred, rmse, feature_importance

@st.cache_data(ttl=3600)
def train_lstm_model(ticker_symbol):
    seq_length = 10
    raw_dl = yf.download(ticker_symbol, period="2y", progress=False)
    
    split_idx = int(len(raw_dl) * 0.8)
    train_raw = raw_dl[['Close']].iloc[:split_idx].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw) 
    scaled_data = scaler.transform(raw_dl[['Close']].values)
    
    X_dl, y_dl = [], []
    for i in range(len(scaled_data) - seq_length):
        X_dl.append(scaled_data[i:(i + seq_length), 0])
        y_dl.append(scaled_data[i + seq_length, 0]) 
        
    X_tensor = torch.FloatTensor(np.array(X_dl).reshape(-1, seq_length, 1))
    y_tensor = torch.FloatTensor(np.array(y_dl).reshape(-1, 1))
    
    X_train_tensor = X_tensor[:split_idx - seq_length]
    y_train_tensor = y_tensor[:split_idx - seq_length]
    X_test_tensor = X_tensor[split_idx - seq_length:]
    y_test_tensor = y_tensor[split_idx - seq_length:]
    
    model_dl = XAUUSDForecasterLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_dl.parameters(), lr=0.01)
    
    for epoch in range(50):
        model_dl.train()
        optimizer.zero_grad()
        loss = criterion(model_dl(X_train_tensor), y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model_dl.eval()
    with torch.no_grad():
        test_preds = model_dl(X_test_tensor)
        test_preds_inv = scaler.inverse_transform(test_preds.numpy())
        y_test_inv = scaler.inverse_transform(y_test_tensor.numpy())
        lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, test_preds_inv))
        
        pred_scaled = model_dl(X_tensor[-1:].clone().detach())
        
    lstm_pred = scaler.inverse_transform(pred_scaled.numpy())[0][0]
    lstm_actual = raw_dl['Close'].iloc[-1].item()
    
    return lstm_pred, lstm_actual, lstm_rmse

col_ai1, col_ai2 = st.columns(2, gap="large")

with col_ai1:
    with st.container(border=True):
        st.markdown("<h3 style='font-family: Space Mono; color: #00F0FF; text-shadow: 0 0 10px rgba(0,240,255,0.5);'>:: Lasso Regression</h3>", unsafe_allow_html=True)
        with st.spinner("Load model statistik..."):
            try:
                lasso_pred, lasso_rmse, lasso_fi = train_lasso_model(ticker)
                if lasso_pred:
                    st.session_state['next_day_pred'] = lasso_pred
                    st.session_state['lasso_rmse'] = lasso_rmse
                    st.session_state['lasso_importance'] = lasso_fi
                    
                    l1, l2 = st.columns(2)
                    l1.metric("Prediksi", f"${lasso_pred:,.2f}")
                    l2.metric("RMSE (Test Data)", f"${lasso_rmse:,.2f}")
            except Exception as e:
                st.error(f"ML Error: {e}")

with col_ai2:
    with st.container(border=True):
        st.markdown("<h3 style='font-family: Space Mono; color: #FCEE0A; text-shadow: 0 0 10px rgba(252,238,10,0.5);'>:: PyTorch LSTM</h3>", unsafe_allow_html=True)
        if st.button("INITIALIZE TENSOR", use_container_width=True):
            with st.spinner("Komputasi Deep Learning..."):
                try:
                    lstm_pred, lstm_actual, lstm_rmse = train_lstm_model(ticker)
                    st.session_state['lstm_pred'] = lstm_pred
                    st.session_state['lstm_rmse'] = lstm_rmse
                    
                    d1, d2 = st.columns(2)
                    d1.metric("Proyeksi", f"${lstm_pred:,.2f}", f"{lstm_pred - lstm_actual:+.2f}")
                    d2.metric("RMSE (Test Data)", f"${lstm_rmse:,.2f}")
                except Exception as e:
                    st.error(f"PyTorch Error: {e}")

# -------------------------------------------------------------------
# MODEL EVALUATION & EXPLAINABILITY DASHBOARD
# -------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 AI/ML MODEL METRICS & EXPLAINABILITY", expanded=True):
    c_eval1, c_eval2 = st.columns([1, 1], gap="large")
    
    with c_eval1:
        st.markdown("<span style='font-family: Space Mono; color: #00F0FF;'>[LASSO FEATURE IMPORTANCE]</span>", unsafe_allow_html=True)
        if 'lasso_importance' in st.session_state:
            fi_df = st.session_state['lasso_importance']
            fig_fi = go.Figure(go.Bar(
                x=fi_df['Coefficient'],
                y=fi_df['Feature'],
                orientation='h',
                marker=dict(color=np.where(fi_df['Coefficient'] > 0, '#00FF66', '#FF003C'))
            ))
            fig_fi.update_layout(height=250, margin=dict(l=0, r=20, t=10, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Sistem menunggu inisialisasi model Lasso...")
            
    with c_eval2:
        st.markdown("<span style='font-family: Space Mono; color: #FCEE0A;'>[MODEL EVALUATION (RMSE)]</span>", unsafe_allow_html=True)
        lasso_err = st.session_state.get('lasso_rmse', 0)
        lstm_err = st.session_state.get('lstm_rmse', 0)
        
        if lasso_err and lstm_err:
            fig_comp = go.Figure(go.Bar(
                x=['Lasso (Statistical)', 'LSTM (Deep Learning)'],
                y=[lasso_err, lstm_err],
                marker=dict(color=['#00F0FF', '#FCEE0A'])
            ))
            fig_comp.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', yaxis_title="Error Margin (USD)")
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Jalankan komputasi PyTorch LSTM untuk mengaktifkan komparasi metrik.")

# -------------------------------------------------------------------
# VECTORBT BACKTEST
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: #FF003C; text-shadow: 0 0 10px rgba(255,0,60,0.5);'> HISTORICAL SIMULATION (5Y)</h2>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def run_cached_backtest(ticker_sym, sw, lw, capital):
    import vectorbt as vbt
    bt_data = yf.download(ticker_sym, period="5y", progress=False)
    if isinstance(bt_data.columns, pd.MultiIndex):
        bt_data.columns = bt_data.columns.droplevel(1)
    price_series = bt_data['Close']
        
    fast_ma = vbt.MA.run(price_series, sw)
    slow_ma = vbt.MA.run(price_series, lw)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    port = vbt.Portfolio.from_signals(price_series, entries, exits, init_cash=capital, fees=0.001)
    
    ret = port.total_return() * 100
    prof = port.total_profit()
    win = port.trades.win_rate() * 100
    dd = port.max_drawdown() * 100
    fig = port.plot()
    
    return ret, prof, win, dd, fig

with st.spinner("Komputasi historis..."):
    try:
        vbt_ret, vbt_prof, vbt_win, vbt_dd, fig_bt = run_cached_backtest(ticker, short_window, long_window, account_capital)
        
        st.session_state['vbt_return'] = vbt_ret
        st.session_state['vbt_profit'] = vbt_prof
        st.session_state['vbt_drawdown'] = vbt_dd
        
        col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
        col_bt1.metric("Return", f"{vbt_ret:.2f}%")
        col_bt2.metric("Profit", f"${vbt_prof:,.2f}")
        col_bt3.metric("Win Rate", f"{vbt_win:.2f}%")
        col_bt4.metric("Drawdown", f"{vbt_dd:.2f}%")
        
        fig_bt.update_layout(height=500, template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bt, use_container_width=True)
        
    except ImportError:
        st.warning("Install vectorbt!")
    except Exception as e:
        st.error(f"Backtest Error: {e}")

# -------------------------------------------------------------------
# MONTE CARLO
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: white;'> STOCHASTIC MONTE CARLO</h2>", unsafe_allow_html=True)

with st.spinner("Menjalankan simulasi..."):
    try:
        if 'df' in locals() and not df.empty:
            historical_closes = df['Close']
            
            sim_df = run_monte_carlo(historical_closes, days_ahead=30, simulations=500)
            v95, v99 = calculate_risk_metrics(sim_df)
            
            col_mc1, col_mc2 = st.columns(2)
            col_mc1.metric("95% Confidence", f"${v95:,.2f}")
            col_mc2.metric("99% Confidence", f"${v99:,.2f}")
            
            fig_mc = go.Figure()
            for col in sim_df.columns[:100]:
                fig_mc.add_trace(go.Scatter(
                    y=sim_df[col], mode='lines', 
                    line=dict(width=1, color='rgba(255, 0, 60, 0.15)'), 
                    showlegend=False
                ))
                
            fig_mc.update_layout(
                height=450,
                xaxis_title="Days Ahead",
                yaxis_title="Projected Price",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,0,60,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,0,60,0.1)')
            )
            st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.warning("Data kosong.")
            
    except Exception as e:
        st.error(f"Monte Carlo: {e}")

# -------------------------------------------------------------------
# AI AGENT
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: #00F0FF; text-shadow: 0 0 10px rgba(0,240,255,0.5);'> ✧ NEURAL_AGENT INTERFACE</h2>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("<span style='font-family: Space Mono; color: #00F0FF;'>🤖 AGENT UPLINK</span>", unsafe_allow_html=True)
groq_api_key = st.sidebar.text_input("Groq Key:", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system", 
            "content": "Kamu adalah AI Quant Agent. Jawablah layaknya analis finansial profesional. ATURAN SUPER PENTING: Jika kamu butuh memanggil tools/fungsi (seperti get_predictions atau get_backtest), panggil tools tersebut secara langsung TANPA menambahkan teks penjelasan apapun. Jangan pernah merender tag HTML/XML seperti <function> secara manual."
        },
        {
            "role": "assistant", 
            "content": "SYSTEM ONLINE. Quant Agent siap menerima instruksi analisis pasar."
        }
    ]

for msg in st.session_state.messages:
    role = msg.get("role") if isinstance(msg, dict) else msg.role
    content = msg.get("content") if isinstance(msg, dict) else msg.content
    
    if role not in ["system", "tool"] and content:
        with st.chat_message(role):
            st.markdown(content)

if prompt := st.chat_input("Ketik instruksi atau parameter analisis..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not groq_api_key:
        st.error("UPLINK FAILED: Masukkan API Key Groq.")
    else:
        from groq import Groq
        import json
        client = Groq(api_key=groq_api_key)
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_predictions",
                    "description": "Ambil hasil prediksi Lasso dan LSTM terbaru.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_backtest",
                    "description": "Ambil metrik backtest (Return, Profit, Drawdown).",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            try:
                response = client.chat.completions.create(
                    messages=st.session_state.messages,
                    model="llama-3.3-70b-versatile",
                    tools=tools,
                    tool_choice="auto"
                )
                
                response_msg = response.choices[0].message
                
                if response_msg.tool_calls:
                    st.session_state.messages.append(response_msg.model_dump())
                    
                    for tool_call in response_msg.tool_calls:
                        func_name = tool_call.function.name
                        
                        if func_name == "get_predictions":
                            val_lasso = st.session_state.get('next_day_pred', 'Belum dikomputasi')
                            val_lstm = st.session_state.get('lstm_pred', 'Belum dikomputasi')
                            result = f"Prediksi Lasso: {val_lasso}, Prediksi PyTorch LSTM: {val_lstm}"
                        elif func_name == "get_backtest":
                            ret = st.session_state.get('vbt_return', None)
                            prof = st.session_state.get('vbt_profit', None)
                            dd = st.session_state.get('vbt_drawdown', None)
                            if ret is not None:
                                result = f"Return: {ret:.2f}%, Profit: ${prof:.2f}, Max Drawdown: {dd:.2f}%"
                            else:
                                result = "Data backtest kosong."
                        else:
                            result = "Fungsi invalid."
                            
                        st.session_state.messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": func_name,
                            "content": result
                        })
                        
                    final_response = client.chat.completions.create(
                        messages=st.session_state.messages,
                        model="llama-3.3-70b-versatile"
                    )
                    final_reply = final_response.choices[0].message.content
                    msg_placeholder.markdown(final_reply)
                    st.session_state.messages.append({"role": "assistant", "content": final_reply})
                    
                else:
                    final_reply = response_msg.content
                    msg_placeholder.markdown(final_reply)
                    st.session_state.messages.append({"role": "assistant", "content": final_reply})

            except Exception as e:
                st.error(f"Error AI: {e}")