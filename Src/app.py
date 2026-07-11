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
    # Init layer
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
st.sidebar.markdown("<h2 style='font-family: Bebas Neue; color: #D4AF37;'> ✦ TERMINAL</h2>", unsafe_allow_html=True)
ticker = st.sidebar.text_input("Instrument Ticker", value="GC=F")
backtest_days = st.sidebar.slider("Historical Data (Days)", min_value=60, max_value=365, value=180)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parameters")
short_window = st.sidebar.number_input("Fast MA", min_value=5, max_value=30, value=20)
long_window = st.sidebar.number_input("Slow MA", min_value=31, max_value=100, value=50)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Risk")
account_capital = st.sidebar.number_input("Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
risk_percentage = st.sidebar.slider("Risk (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

# -------------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------------
st.markdown(f'<h1 class="terminal-header">{ticker} QUANTITATIVE ENGINE</h1>', unsafe_allow_html=True)
st.markdown('<div class="terminal-sub"> ✦ OPERATIONAL | PREDICTIVE ANALYTICS </div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# DATA INGESTION
# -------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_institutional_data(symbol, days):
    # Fetch data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days + 100)
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

try:
    df_raw = fetch_institutional_data(ticker, backtest_days)
    
    if df_raw.empty:
        # Error check
        st.error("Execution Terminated!")
    else:
        df = df_raw.copy()
        
        # Extract features
        df['MA_Fast'] = df['Close'].rolling(window=short_window).mean()
        df['MA_Slow'] = df['Close'].rolling(window=long_window).mean()
        
        # Calc volatility
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Calc return
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df_filtered = df.tail(backtest_days).copy()
        
        # Crossover logic
        df_filtered['Signal'] = np.where(df_filtered['MA_Fast'] > df_filtered['MA_Slow'], 1, -1)
        df_filtered['Strategy_Return'] = df_filtered['Log_Return'] * df_filtered['Signal'].shift(1)
        
        # Signal marker
        df_filtered['Position_Changes'] = df_filtered['Signal'].diff()
        df_filtered['Buy_Markers'] = np.where(df_filtered['Position_Changes'] == 2, df_filtered['Close'], np.nan)
        df_filtered['Sell_Markers'] = np.where(df_filtered['Position_Changes'] == -2, df_filtered['Close'], np.nan)
        
        # Calc metrics
        latest_price = float(df_filtered['Close'].iloc[-1])
        current_atr = float(df_filtered['ATR'].iloc[-1])
        asset_cum_return = (np.exp(df_filtered['Log_Return'].sum()) - 1) * 100
        strategy_cum_return = (np.exp(df_filtered['Strategy_Return'].sum()) - 1) * 100
        
        # Calc drawdown
        strategy_cum_wealth = np.exp(df_filtered['Strategy_Return'].cumsum())
        peak = strategy_cum_wealth.cummax()
        drawdown = (strategy_cum_wealth - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Risk management
        cash_risk = account_capital * (risk_percentage / 100)
        stop_loss_distance = current_atr * 2 
        simulated_position_size = cash_risk / stop_loss_distance if stop_loss_distance > 0 else 0.0

        # UI metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spot Price", f"${latest_price:,.2f}")
        m2.metric("Volatility (ATR)", f"${current_atr:.2f}")
        m3.metric("Algo Return", f"{strategy_cum_return:+.2f}%", delta=f"{strategy_cum_return - asset_cum_return:.2f}%")
        m4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs([" Chart", " Size", " Matrix"])
        
        with tab1:
            # Execution chart
            st.markdown("<h3 style='font-family: Bebas Neue; color: white;'>Execution History</h3>", unsafe_allow_html=True)
            
            fig = go.Figure()
            # Plot price
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Close'], name='Spot', line=dict(color='#D4AF37', width=2)))
            # Plot MA
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MA_Fast'], name='Fast MA', line=dict(color='#00D2FF', dash='dot')))
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MA_Slow'], name='Slow MA', line=dict(color='#FF3B30', dash='dot')))
            # Plot signal
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Buy_Markers'], mode='markers', name='LONG', marker=dict(symbol='triangle-up', size=14, color='#34C759', line=dict(width=1, color='white'))))
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Sell_Markers'], mode='markers', name='SHORT', marker=dict(symbol='triangle-down', size=14, color='#FF3B30', line=dict(width=1, color='white'))))
            
            # Style chart
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
            # Risk info
            c1, c2 = st.columns(2)
            with c1:
                st.info(f" ✦ Cash Risk: ${cash_risk:,.2f}")
                st.markdown(f"✦ Stop Loss: ${stop_loss_distance:.2f}")
            with c2:
                st.success(f" ✦ SIMULATED: {simulated_position_size:.3f} units")
                st.markdown(f"✦ Allocation: ${simulated_position_size * latest_price:,.2f}")

        with tab3:
            # Raw data
            st.dataframe(df_filtered[['Close', 'MA_Fast', 'MA_Slow', 'ATR', 'Strategy_Return', 'Signal']].tail(15), use_container_width=True)
            
except Exception as e:
    st.error(f"System Failure: {str(e)}")

# -------------------------------------------------------------------
# MACRO RADAR
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: white;'> Macro Radar</h2>", unsafe_allow_html=True)

with st.spinner("Sinkronisasi data..."):
    try:
        # Instrument dict
        macro_basket = {
            "Primary": ticker,
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "USD Index": "DX-Y.NYB"
        }
        
        macro_data = pd.DataFrame()
        # Fetch loop
        for name, sym in macro_basket.items():
            temp_df = yf.download(sym, period=f"{backtest_days}d", progress=False)
            if not temp_df.empty and 'Close' in temp_df.columns:
                macro_data[name] = temp_df['Close'].squeeze()
            
        macro_data.dropna(inplace=True)
        corr_matrix = macro_data.corr()
        
        col_macro1, col_macro2 = st.columns([1, 2], gap="large")
        
        with col_macro1:
            st.markdown("<span style='font-family: Space Mono; color: #8b949e;'>MATRIKS KORELASI</span>", unsafe_allow_html=True)
            # Correlation heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}", showscale=False
            ))
            fig_corr.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_macro2:
            st.markdown("<span style='font-family: Space Mono; color: #8b949e;'>KINERJA (BASE 100)</span>", unsafe_allow_html=True)
            normalized_data = (macro_data / macro_data.iloc[0]) * 100
            fig_line = go.Figure()
            
            # Line plot
            for col in normalized_data.columns:
                width = 3 if col == "Primary" else 1.5
                dash = 'solid' if col == "Primary" else 'dot'
                fig_line.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data[col], mode='lines', name=col, line=dict(width=width, dash=dash)))
                
            fig_line.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig_line, use_container_width=True)
            
    except Exception as e:
        st.error(f"Gagal radar: {str(e)}")

# -------------------------------------------------------------------
# ML ENGINES
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: white;'> Predictive Architectures</h2>", unsafe_allow_html=True)

col_ai1, col_ai2 = st.columns(2, gap="large")

with col_ai1:
    # Lasso container
    with st.container(border=True):
        st.markdown("<h3 style='font-family: Space Mono; color: #00d2ff;'> Lasso Regression</h3>", unsafe_allow_html=True)
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
                    
                    # Eval model
                    predictions = model_ml.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    next_day_pred = model_ml.predict(X.iloc[-1].values.reshape(1, -1))[0]
                    
                    l1, l2 = st.columns(2)
                    l1.metric("Prediksi", f"${next_day_pred:,.2f}")
                    l2.metric("RMSE", f"${rmse:,.2f}")
            except Exception as e:
                st.error(f"ML Error: {e}")

with col_ai2:
    # PyTorch container
    with st.container(border=True):
        st.markdown("<h3 style='font-family: Space Mono; color: #b026ff;'> PyTorch LSTM</h3>", unsafe_allow_html=True)
        if st.button("INITIALIZE TENSOR", use_container_width=True):
            with st.spinner("Iterasi pelatihan..."):
                try:
                    seq_length = 10
                    raw_dl = yf.download(ticker, period="2y", progress=False)
                    
                    # Scale data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(raw_dl[['Close']].values)
                    
                    # Create sequence
                    X_dl, y_dl = [], []
                    for i in range(len(scaled_data) - seq_length):
                        X_dl.append(scaled_data[i:(i + seq_length), 0])
                        y_dl.append(scaled_data[i + seq_length, 0]) 
                        
                    # Convert tensor
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
                    
                    # Predict data
                    model_dl.eval()
                    with torch.no_grad():
                        pred_scaled = model_dl(X_tensor[-1:].clone().detach())
                        
                    lstm_pred = scaler.inverse_transform(pred_scaled.numpy())[0][0]
                    lstm_actual = raw_dl['Close'].iloc[-1].item()
                    
                    d1, d2 = st.columns(2)
                    d1.metric("Aktual", f"${lstm_actual:,.2f}")
                    d2.metric("Proyeksi", f"${lstm_pred:,.2f}", f"{lstm_pred - lstm_actual:+.2f}")
                except Exception as e:
                    st.error(f"PyTorch Error: {e}")

# -------------------------------------------------------------------
# VECTORBT BACKTEST
# -------------------------------------------------------------------
st.divider()
st.markdown("<h2 style='font-family: Bebas Neue; color: white;'> 5-Year Backtest</h2>", unsafe_allow_html=True)

with st.spinner("Komputasi historis..."):
    try:
        import vectorbt as vbt
        
        # Fetch data
        bt_data = yf.download(ticker, period="5y", progress=False)
        if isinstance(bt_data.columns, pd.MultiIndex):
            bt_data.columns = bt_data.columns.droplevel(1)
        price_series = bt_data['Close']
            
        # Vector signal
        fast_ma = vbt.MA.run(price_series, short_window)
        slow_ma = vbt.MA.run(price_series, long_window)
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Run sim
        portfolio = vbt.Portfolio.from_signals(price_series, entries, exits, init_cash=account_capital, fees=0.001)
        
        # Calc metrics
        col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
        col_bt1.metric("Return", f"{portfolio.total_return() * 100:.2f}%")
        col_bt2.metric("Profit", f"${portfolio.total_profit():,.2f}")
        col_bt3.metric("Win Rate", f"{portfolio.trades.win_rate() * 100:.2f}%")
        col_bt4.metric("Drawdown", f"{portfolio.max_drawdown() * 100:.2f}%")
        
        # Plot chart
        fig_bt = portfolio.plot()
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
st.markdown("<h2 style='font-family: Bebas Neue; color: white;'> Monte Carlo</h2>", unsafe_allow_html=True)
st.markdown("<span style='font-family: Space Mono; color: #8b949e;'>Simulasi stokastik.</span>", unsafe_allow_html=True)

with st.spinner("Menjalankan simulasi..."):
    try:
        if 'df' in locals() and not df.empty:
            historical_closes = df['Close']
            
            # Run function
            sim_df = run_monte_carlo(historical_closes, days_ahead=30, simulations=500)
            v95, v99 = calculate_risk_metrics(sim_df)
            
            # Render metrics
            col_mc1, col_mc2 = st.columns(2)
            col_mc1.metric("95% Confidence", f"${v95:,.2f}")
            col_mc2.metric("99% Confidence", f"${v99:,.2f}")
            
            # Plotly chart
            fig_mc = go.Figure()
            for col in sim_df.columns[:100]:
                fig_mc.add_trace(go.Scatter(
                    y=sim_df[col], mode='lines', 
                    line=dict(width=1, color='rgba(212, 175, 55, 0.08)'), 
                    showlegend=False
                ))
                
            fig_mc.update_layout(
                height=450,
                xaxis_title="Days Ahead",
                yaxis_title="Projected Price",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#8b949e',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
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
st.markdown("<h2 style='font-family: Bebas Neue; color: white;'> ✧ AI Agent</h2>", unsafe_allow_html=True)

# API Key
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Agent")
groq_api_key = st.sidebar.text_input("Groq Key:", type="password")

# Init Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Kamu AI Quant. Gunakan tools untuk mengambil data riil dari dashboard. Jawablah layaknya analis finansial profesional."},
        {"role": "assistant", "content": "Halo. Saya AI Quant Agent. Ada yang bisa dianalisis hari ini?"}
    ]

# Render Chat
for msg in st.session_state.messages:
    # Cek tipe
    role = msg.get("role") if isinstance(msg, dict) else msg.role
    content = msg.get("content") if isinstance(msg, dict) else msg.content
    
    # Filter UI
    if role not in ["system", "tool"] and content:
        with st.chat_message(role):
            st.markdown(content)

# User Input
if prompt := st.chat_input("Ketik instruksi..."):
    
    # Show User
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check Key
    if not groq_api_key:
        st.error("Masukkan API.")
    else:
        # Call Groq
        from groq import Groq
        import json
        client = Groq(api_key=groq_api_key)
        
        # Tools Def
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
                # Request AI
                response = client.chat.completions.create(
                    messages=st.session_state.messages,
                    model="llama-3.3-70b-versatile", 
                    tools=tools,
                    tool_choice="auto"
                )
                
                response_msg = response.choices[0].message
                
                # Cek Tools
                if response_msg.tool_calls:
                    # Simpan memori
                    st.session_state.messages.append(response_msg.model_dump())
                    
                    for tool_call in response_msg.tool_calls:
                        func_name = tool_call.function.name
                        
                        # Execute Tool
                        if func_name == "get_predictions":
                            val_lasso = locals().get('next_day_pred', 'Belum dikomputasi')
                            val_lstm = locals().get('lstm_pred', 'Belum dikomputasi')
                            result = f"Prediksi Lasso: {val_lasso}, Prediksi PyTorch LSTM: {val_lstm}"
                        elif func_name == "get_backtest":
                            port = locals().get('portfolio', None)
                            if port:
                                result = f"Return: {port.total_return()*100:.2f}%, Profit: ${port.total_profit():.2f}, Max Drawdown: {port.max_drawdown()*100:.2f}%"
                            else:
                                result = "Data backtest kosong."
                        else:
                            result = "Fungsi invalid."
                            
                        # Save Tool
                        st.session_state.messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": func_name,
                            "content": result
                        })
                        
                    # Request Final
                    final_response = client.chat.completions.create(
                        messages=st.session_state.messages,
                        model="llama-3.3-70b-versatile"
                    )
                    final_reply = final_response.choices[0].message.content
                    msg_placeholder.markdown(final_reply)
                    st.session_state.messages.append({"role": "assistant", "content": final_reply})
                    
                else:
                    # Normal Reply
                    final_reply = response_msg.content
                    msg_placeholder.markdown(final_reply)
                    st.session_state.messages.append({"role": "assistant", "content": final_reply})

            except Exception as e:
                # Catch Error
                st.error(f"Error AI: {e}")
