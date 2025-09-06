# ml_training/L3/download_portfolio_data_hrm.py
import os
import yfinance as yf
import pandas as pd

DATA_DIR = "data/datos_para_modelos_l3/portfolio"
os.makedirs(DATA_DIR, exist_ok=True)

# Lista extendida de activos
TICKERS = ["AAPL", "MSFT", "GOOG", "BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"]

START_DATE = "2018-01-01"
END_DATE = "2025-01-01"

for ticker in TICKERS:
    print(f"Descargando {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    df.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"))

print("Datos descargados y guardados en:", DATA_DIR)
