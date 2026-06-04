import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n_days = 500
dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='B')

trend = np.linspace(2000, 2400, n_days)
noise = np.random.normal(0, 15, n_days)
prices = trend + noise

df = pd.DataFrame({'Tanggal': dates, 'Harga_Emas': prices})

df['Lag_1'] = df['Harga_Emas'].shift(1)
df['Lag_2'] = df['Harga_Emas'].shift(2)
df.dropna(inplace=True)

X = df[['Lag_1', 'Lag_2']]
y = df['Harga_Emas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== 🪙 GOLD PRICE FORECASTING ENGINE ===")
print("Berhasil memuat dataset histori dan ekstraksi fitur lag.\n")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- 📊 LAPORAN EVALUASI MODEL ---")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"R-Squared (R2 Score)     : {r2 * 100:.2f}%")