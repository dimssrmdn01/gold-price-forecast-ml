import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import warnings

# Mengabaikan warning dari yfinance agar terminal tetap bersih
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. ARSITEKTUR DEEP LEARNING (LSTM)
# ---------------------------------------------------------
class XAUUSDForecasterLSTM(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1):
        super(XAUUSDForecasterLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# ---------------------------------------------------------
# 2. PIPELINE DATA REAL-TIME
# ---------------------------------------------------------
def fetch_and_preprocess_market_data(ticker_symbol: str = "GC=F", period: str = "2y", sequence_length: int = 10):
    print(f"[*] Menghubungi server Yahoo Finance untuk ditarik data historis {ticker_symbol}...")
    
    data = yf.download(ticker_symbol, period=period, progress=False)
    if data.empty:
        raise ValueError("Gagal menarik data dari server.")
        
    print(f"[*] Berhasil menarik {len(data)} baris data perdagangan harian.")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length), 0])
        y.append(scaled_data[i + sequence_length, 0])
        
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y).reshape(-1, 1)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    print(f"[*] Normalisasi dan pemotongan sequence selesai.")
    return X_tensor, y_tensor, scaler, data

# ---------------------------------------------------------
# 3. MESIN PELATIHAN (TRAINING LOOP)
# ---------------------------------------------------------
def train_forecaster(X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 0.01):
    print("=====================================================")
    print("     MEMULAI SIKLUS PELATIHAN NEURAL NETWORK         ")
    print("=====================================================")
    
    model = XAUUSDForecasterLSTM(input_dim=1, hidden_dim=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch: {epoch+1:03d}/{epochs} | Eror Pelatihan (MSE Loss): {loss.item():.6f}")
            
    print("=====================================================")
    return model

# ---------------------------------------------------------
# EKSEKUSI UTAMA
# ---------------------------------------------------------
if __name__ == "__main__":
    SEQ_LENGTH = 10
    
    try:
        X_train, y_train, price_scaler, raw_df = fetch_and_preprocess_market_data(sequence_length=SEQ_LENGTH)
        
        trained_model = train_forecaster(X_train, y_train, epochs=100)
        
        trained_model.eval() 
        latest_sequence = X_train[-1:].clone().detach() 
        
        with torch.no_grad():
            predicted_scaled = trained_model(latest_sequence)
            
        predicted_price_usd = price_scaler.inverse_transform(predicted_scaled.numpy())
        
        # Ambil nilai skalar dari Pandas DataFrame (aman dari error multidimensi)
        last_actual_price = raw_df['Close'].iloc[-1].item()
        
        print(f"\n>> STATUS PASAR REAL-TIME (Harga Penutupan Terakhir): ${last_actual_price:,.2f}")
        print(f">> PREDIKSI NEURAL NETWORK (Harga Penutupan Esok)   : ${predicted_price_usd[0][0]:,.2f}\n")
        
    except Exception as e:
        print(f"Terjadi kesalahan sistem: {e}")