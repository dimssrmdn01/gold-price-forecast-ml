import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Pengaturan layout terminal profesional
st.set_page_config(
    page_title="Institutional Gold Quant Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS untuk nuansa Bloomberg Terminal Dark Mode
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
    df = yf.download(symbol, start=start_date, end=end_date)
    if hasattr(data.columns, 'levels'):
            data.columns = data.columns.get_level_values(0)
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
        
        # Filter data
        df_filtered = df.tail(backtest_days).copy()
        
        # -------------------------------------------------------------------
        # ALGORITHMIC BACKTEST EXECUTION ENGINE & RISK METRICS
        # -------------------------------------------------------------------
        df_filtered['Signal'] = np.where(df_filtered['MA_Fast'] > df_filtered['MA_Slow'], 1, -1)
        df_filtered['Strategy_Return'] = df_filtered['Log_Return'] * df_filtered['Signal'].shift(1)
        
        # Deteksi Titik Eksekusi Order (Sinyal Berubah)
        df_filtered['Position_Changes'] = df_filtered['Signal'].diff()
        df_filtered['Buy_Markers'] = np.where(df_filtered['Position_Changes'] == 2, df_filtered['Close'], np.nan)
        df_filtered['Sell_Markers'] = np.where(df_filtered['Position_Changes'] == -2, df_filtered['Close'], np.nan)
        
        # Kalkulasi Finansial Lanjutan
        latest_price = float(df_filtered['Close'].iloc[-1])
        current_atr = float(df_filtered['ATR'].iloc[-1])
        
        asset_cum_return = (np.exp(df_filtered['Log_Return'].sum()) - 1) * 100
        strategy_cum_return = (np.exp(df_filtered['Strategy_Return'].sum()) - 1) * 100
        
        # UPGRADE: Kalkulasi Maximum Drawdown (Penurunan Modal Terparah)
        strategy_cum_wealth = np.exp(df_filtered['Strategy_Return'].cumsum())
        peak = strategy_cum_wealth.cummax()
        drawdown = (strategy_cum_wealth - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        total_trades = df_filtered['Position_Changes'].abs().sum() / 2
        win_trades = df_filtered[df_filtered['Strategy_Return'] > 0].shape[0]
        win_rate = (win_trades / df_filtered.shape[0]) * 100

        # UPGRADE: Dynamic Position Sizing Calculator (Simulator Manajemen Risiko)
        cash_risk = account_capital * (risk_percentage / 100)
        stop_loss_distance = current_atr * 2 # Standar industri: Stop Loss berjarak 2x nilai ATR
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
            
            # Plot Garis Harga dan Moving Average
            ax1.plot(df_filtered.index, df_filtered['Close'], label='XAUUSD Spot Price', color='#D4AF37', linewidth=2, alpha=0.9)
            ax1.plot(df_filtered.index, df_filtered['MA_Fast'], label=f'Fast MA ({short_window}D)', color='#00D2FF', linestyle='--', linewidth=1.2)
            ax1.plot(df_filtered.index, df_filtered['MA_Slow'], label=f'Slow MA ({long_window}D)', color='#FF3B30', linestyle='--', linewidth=1.2)
            
            # UPGRADE: Menempelkan Sinyal Buy (Segitiga Hijau Ke atas) dan Sell (Segitiga Merah Ke bawah)
            ax1.scatter(df_filtered.index, df_filtered['Buy_Markers'], label='EXECUTE LONG (BUY)', color='#34C759', marker='^', s=150, zorder=5)
            ax1.scatter(df_filtered.index, df_filtered['Sell_Markers'], label='EXECUTE SHORT (SELL)', color='#FF3B30', marker='v', s=150, zorder=5)
            
            ax1.set_ylabel("Price (USD)", color='white', fontsize=12)
            ax1.tick_params(colors='white')
            ax1.legend(loc='upper left', facecolor='#0e1117', edgecolor='#30363d', labelcolor='white')
            ax1.set_title("Algorithmic Order Execution Tracking Framework", color='white', fontsize=14, fontweight='bold')
            
            st.pyplot(fig)

        with tab2:
            st.subheader("Automated Position Sizing & Capital Allocation")
            st.markdown("Risk management calculation parameters designed to preserve fund principal under market stress.")
            
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"Allowed Cash Risk: ${cash_risk:,.2f} per trade (Based on {risk_percentage}% parameter)")
                st.markdown(f"**Recommended Target Stop Loss Distance:** ${stop_loss_distance:.2f} (2x ATR Buffer)")
            with c2:
                st.success(f"SIMULATED POSITION SIZE ORDER: Allocation of {simulated_position_size:.3f} units suggested.")
                st.markdown(f"**Estimated Capital Allocation Deployment:** ${simulated_position_size * latest_price:,.2f}")
                
            st.markdown("""
                > **Risk Mitigation Protocol:** Maximum drawdown of metrics defines the historical worst-case equity drop. Institutional systems require active allocation modification when simulated drawdown thresholds breach limits.
            """)

        with tab3:
            st.subheader("Automated Computed Matrix Stream")
            st.dataframe(
                df_filtered[['Close', 'MA_Fast', 'MA_Slow', 'ATR', 'Strategy_Return', 'Signal']].tail(15),
                use_container_width=True
            )
            
except Exception as e:
    st.error(f"Critical System Failure: {str(e)}")
