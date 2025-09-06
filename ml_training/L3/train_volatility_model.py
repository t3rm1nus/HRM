# train_volatility_model_garch_lstm.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from arch import arch_model

# -----------------------
# Rutas
# -----------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data" / "datos_para_modelos_l3" / "volatility"
MODEL_DIR = BASE_DIR.parent / "models" / "L3"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TICKER = "ETH-USD"
CSV_PATH = DATA_DIR / f"{TICKER}.csv"

# -----------------------
# Funciones
# -----------------------
def download_data():
    if not CSV_PATH.exists():
        print(f"{TICKER}.csv no existe, descargando...")
        df = yf.download(TICKER, period="max")
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CSV_PATH)
    else:
        print(f"{TICKER}.csv ya existe, usando archivo local.")

def load_prices(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    for col in ["Adj Close", "Adj_Close", "Close"]:
        if col in df.columns:
            prices = df[col].copy()
            prices = prices.replace(",", "", regex=True)
            prices = pd.to_numeric(prices, errors="coerce")
            prices = prices.dropna()
            return prices
    raise ValueError(f"Ninguna columna de cierre encontrada en {path}. Columnas disponibles: {df.columns.tolist()}")

def create_dataset(series, window=20):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    download_data()
    prices = load_prices(CSV_PATH)

    # Log-returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # -----------------------
    # GARCH
    # -----------------------
    print("Entrenando modelo GARCH(1,1)...")
    am = arch_model(log_returns*100, vol='Garch', p=1, q=1, dist='Normal')
    garch_model = am.fit(disp='off')
    joblib.dump(garch_model, MODEL_DIR / f"{TICKER}_volatility_garch.pkl")
    print(f"GARCH guardado en {MODEL_DIR / f'{TICKER}_volatility_garch.pkl'}")

    # -----------------------
    # LSTM
    # -----------------------
    print("Preparando datos para LSTM...")
    series = log_returns.values.reshape(-1,1)
    window_size = 20
    X, y = create_dataset(series, window=window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(window_size,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    print("Entrenando modelo LSTM...")
    model.fit(X, y, epochs=100, batch_size=32, callbacks=[es], verbose=1)
    model.save(MODEL_DIR / f"{TICKER}_volatility_lstm.h5")
    print(f"LSTM guardado en {MODEL_DIR / f'{TICKER}_volatility_lstm.h5'}")
