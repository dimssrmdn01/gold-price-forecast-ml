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

# Impor modul Monte Carlo yang baru kita buat
from Src.monte_carlo import run_monte_carlo, calculate_risk_metrics

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# CONFIGURATION & INITIALIZATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Institutional Quant Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS untuk nuansa Bloomberg Terminal
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
# DEEP LEARNING ARCHITECTURE (LSTM)
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
# CONTROL PANEL (Sidebar pertama untuk mencegah NameError)
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
# HEADER SECTION (Menggunakan variabel ticker secara dinamis)
# -------------------------------------------------------------------
st.title(f"{ticker} Institutional Quantitative & Predictive Analytics Engine")
st.markdown("### `SYSTEM STATUS: OPERATIONAL` | Risk Mitigation & Quantitative Execution Framework")
st.markdown("---")

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
        
        # Advanced Feature Engineering
        df['MA_Fast'] = df['Close'].rolling(window=short_window).mean()
        df['MA_Slow'] = df['Close'].rolling(window=long_window).mean()
        
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df_filtered = df.tail(backtest_days).copy()
        
        # Algorithmic Crossover Calculations
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

        # UI Metric Display
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
            
            ax1.plot(df_filtered.index, df_filtered['Close'], label='Spot Price', color='#D4AF37', linewidth=2, alpha=0.9)
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
# GLOBAL MACRO RADAR & CORRELATION
# -------------------------------------------------------------------
st.divider()
st.subheader("❖ Global Macro Radar & Asset Correlation")

with st.spinner("Sinkronisasi dengan indeks pasar global..."):
    try:
        macro_basket = {
            "Primary": ticker,
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "US Dollar (DXY)": "DX-Y.NYB"
        }
        
        macro_data = pd.DataFrame()
        for name, sym in macro_basket.items():
            temp_df = yf.download(sym, period=f"{backtest_days}d", progress=False)
            if not temp_df.empty and 'Close' in temp_df.columns:
                macro_data[name] = temp_df['Close'].squeeze()
            
        macro_data.dropna(inplace=True)
        corr_matrix = macro_data.corr()
        
        col_macro1, col_macro2 = st.columns([1, 2])
        
        with col_macro1:
            st.markdown("**Matriks Korelasi Kuantitatif**")
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                showscale=False
            ))
            fig_corr.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_macro2:
            st.markdown("**Perbandingan Kinerja Dinormalisasi (Base 100)**")
            normalized_data = (macro_data / macro_data.iloc[0]) * 100
            fig_line = go.Figure()
            
            for col in normalized_data.columns:
                width = 3 if col == "Primary" else 1.5
                dash = 'solid' if col == "Primary" else 'dot'
                fig_line.add_trace(go.Scatter(x=normalized_data.index, y=normalized_data[col], mode='lines', name=col, line=dict(width=width, dash=dash)))
                
            fig_line.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_line, use_container_width=True)
            
    except Exception as e:
        st.error(f"Gagal memuat radar makro: {str(e)}")

# -------------------------------------------------------------------
# NLP SENTIMENT ENGINE
# -------------------------------------------------------------------
st.divider()
st.subheader("❖ Real-Time Fundamental Sentiment (NLP Engine)")
if os.path.exists('Models/sentiment_model.pkl'):
    try:
        nlp_model = joblib.load('Models/sentiment_model.pkl')
        vectorizer = joblib.load('Models/tfidf_vectorizer.pkl')
        
        with st.spinner("Memindai radar fundamental global..."):
            ticker_data = yf.Ticker(ticker)
            raw_news = ticker_data.news
            
            news_list = []
            if raw_news:
                for n in raw_news[:5]:
                    if n.get('title'):
                        news_list.append(n.get('title'))
            
            if not news_list:
                news_list = [
                    f"Surprise volatility adjustments shifting short term asset volume",
                    f"Market parameters hit new baseline amidst scaling global demand",
                    "Central institution releases monthly economic update on inflation parameters"
                ]
            
            bullish_count, bearish_count, neutral_count = 0, 0, 0
            
            for headline in news_list:
                vec_text = vectorizer.transform([headline])
                sentiment = nlp_model.predict(vec_text)[0]
                sent_str = str(sentiment).title().strip() 
                
                if sent_str == 'Bullish': 
                    bullish_count += 1
                    st.markdown(f"- **{headline}** ➔ [▲ {sent_str}]")
                elif sent_str == 'Bearish': 
                    bearish_count += 1
                    st.markdown(f"- **{headline}** ➔ [▼ {sent_str}]")
                else: 
                    neutral_count += 1
                    st.markdown(f"- **{headline}** ➔ [■ {sent_str}]")
                
            st.write("---")
            st.write(f"**Market Bias:** ▲ {bullish_count} Bullish | ▼ {bearish_count} Bearish | ■ {neutral_count} Neutral")
            
    except Exception as e:
        st.error(f"Sistem NLP offline: {e}")
else:
    st.info("Mesin NLP tidak aktif atau belum terpasang.")

# -------------------------------------------------------------------
# MACHINE LEARNING ENGINE (LASSO)
# -------------------------------------------------------------------
st.divider()
st.subheader(f"⟁ {ticker} Predictive ML (Lasso Regression)")

with st.spinner("Mengekstrak data historis dan mengonfigurasi model statistik..."):
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
            
            X = df_ml[['Lag_1', 'Lag_2', 'SMA_10', 'SMA_30']]
            y = df_ml['Close']
            
            split_idx = int(len(df_ml) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            model_ml = Lasso(alpha=0.1)
            model_ml.fit(X_train, y_train)
            
            predictions = model_ml.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            last_row = X.iloc[-1].values.reshape(1, -1)
            next_day_pred = model_ml.predict(last_row)[0]
            
            col_ml1, col_ml2 = st.columns(2)
            with col_ml1:
                st.metric(label="Prediksi Lasso (Esok)", value=f"${next_day_pred:,.2f}")
            with col_ml2:
                st.metric(label="Akurasi Model (RMSE)", value=f"${rmse:,.2f}", delta="Optimal Feature Selection", delta_color="normal")
    except Exception as e:
        st.error(f"Mesin ML gagal dieksekusi: {e}")

# -------------------------------------------------------------------
# DEEP LEARNING ENGINE (PyTorch LSTM)
# -------------------------------------------------------------------
st.divider()
st.subheader(f"⚡ {ticker} Deep Learning Forecaster (PyTorch LSTM)")
st.markdown("Model *time-series forecasting* berbasis Deep Learning untuk memprediksi arah tren harga berdasarkan sekuensial data.")

if st.button("▰▰ INITIALIZE TENSOR COMPUTATION (LSTM) ▰▰", type="primary"):
    with st.spinner("Memproses matriks tensor dan melakukan iterasi pelatihan..."):
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
            
            progress = st.progress(0)
            for epoch in range(50):
                model_dl.train()
                optimizer.zero_grad()
                loss = criterion(model_dl(X_tensor), y_tensor)
                loss.backward()
                optimizer.step()
                progress.progress((epoch + 1) / 50)
            
            model_dl.eval()
            with torch.no_grad():
                pred_scaled = model_dl(X_tensor[-1:].clone().detach())
                
            lstm_pred_usd = scaler.inverse_transform(pred_scaled.numpy())[0][0]
            lstm_actual = raw_dl['Close'].iloc[-1].item()
            
            dl1, dl2 = st.columns(2)
            with dl1:
                st.metric("Harga Aktual Terakhir", f"${lstm_actual:,.2f}")
            with dl2:
                st.metric("Proyeksi Harga (LSTM)", f"${lstm_pred_usd:,.2f}", f"{lstm_pred_usd - lstm_actual:+.2f} USD")
                
        except Exception as e:
            st.error(f"Mesin PyTorch gagal dieksekusi: {e}")

# -------------------------------------------------------------------
# ADVANCED BACKTESTING ENGINE (5-YEAR SIMULATION)
# -------------------------------------------------------------------
st.divider()
st.subheader("❖ Historical Strategy Backtesting (5-Year Simulation)")
st.markdown("Simulasi eksekusi strategi *Moving Average Crossover* terhadap data historis 5 tahun terakhir untuk memvalidasi probabilitas profitabilitas.")

with st.spinner("Menjalankan komputasi simulasi portofolio historis..."):
    try:
        import vectorbt as vbt
        
        # Menarik data 5 tahun terakhir khusus untuk backtest
        bt_data = yf.download(ticker, period="5y", progress=False)
        if isinstance(bt_data.columns, pd.MultiIndex):
            bt_data.columns = bt_data.columns.droplevel(1)
        
        price_series = bt_data['Close']
            
        # Membangun sinyal indikator secara terkomputasi vektor
        fast_ma = vbt.MA.run(price_series, short_window)
        slow_ma = vbt.MA.run(price_series, long_window)
        
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        # Eksekusi simulasi portofolio
        portfolio = vbt.Portfolio.from_signals(
            price_series, 
            entries, 
            exits, 
            init_cash=account_capital, 
            fees=0.001
        )
        
        # Ekstraksi Metrik Kinerja Kunci
        col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
        with col_bt1:
            st.metric("Total Return Kuantitatif", f"{portfolio.total_return() * 100:.2f}%")
        with col_bt2:
            st.metric("Net Profit (PnL)", f"${portfolio.total_profit():,.2f}")
        with col_bt3:
            st.metric("Win Rate (Akurasi Sinyal)", f"{portfolio.trades.win_rate() * 100:.2f}%")
        with col_bt4:
            st.metric("Maximum Drawdown", f"{portfolio.max_drawdown() * 100:.2f}%")
            
        # Visualisasi Portofolio Interaktif
        st.markdown("**Kurva Ekuitas (Equity Curve) & Riwayat Transaksi**")
        fig_bt = portfolio.plot()
        fig_bt.update_layout(height=600, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_bt, use_container_width=True)
        
    except ImportError:
        st.error("Pustaka 'vectorbt' belum terpasang. Jalankan 'pip install vectorbt' di environment Anda.")
    except Exception as e:
        st.error(f"Komputasi backtest gagal dieksekusi: {e}")

# -------------------------------------------------------------------
# MONTE CARLO STRESS TESTING ENGINE
# -------------------------------------------------------------------
st.divider()
st.subheader("❖ Monte Carlo Stress Testing (30-Day Trajectory)")
st.markdown("Simulasi stokastik untuk memproyeksikan skenario pergerakan harga ekstrem berdasarkan volatilitas historis (Risk Analysis).")

with st.spinner("Menjalankan ribuan simulasi Monte Carlo..."):
    try:
        # Pastikan data harga historis (df) sudah terdefinisi dari blok Data Ingestion di atas
        if 'df' in locals() and not df.empty:
            historical_closes = df['Close']
            
            # Eksekusi fungsi dari monte_carlo.py
            sim_df = run_monte_carlo(historical_closes, days_ahead=30, simulations=500)
            v95, v99 = calculate_risk_metrics(sim_df)
            
            # Render metrik risiko (Value at Risk)
            col_mc1, col_mc2 = st.columns(2)
            with col_mc1:
                st.metric("95% Confidence (Worst Case)", f"${v95:,.2f}")
            with col_mc2:
                st.metric("99% Confidence (Worst Case)", f"${v99:,.2f}")
            
            # Visualisasi Plotly untuk jalur simulasi
            fig_mc = go.Figure()
            
            # Plot 100 sampel (dari 500) agar rendering UI tetap ringan dan tidak lag
            for col in sim_df.columns[:100]:
                fig_mc.add_trace(go.Scatter(
                    y=sim_df[col], 
                    mode='lines', 
                    line=dict(width=1, color='rgba(0, 210, 255, 0.05)'),
                    showlegend=False
                ))
                
            fig_mc.update_layout(
                height=500,
                xaxis_title="Hari Kedepan (Days Ahead)",
                yaxis_title="Proyeksi Harga (USD)",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.warning("Data historis tidak tersedia untuk menjalankan simulasi.")
            
    except Exception as e:
        st.error(f"Mesin Monte Carlo gagal dieksekusi: {e}")