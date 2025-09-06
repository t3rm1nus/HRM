"""
Script para descargar datos históricos necesarios para Volatility Forecasting
y guardarlos en la carpeta lógica de HRM.
"""

import os
import yfinance as yf

# Carpeta donde se guardarán los datos
DATA_DIR = "data/datos_para_modelos_l3/volatility"
os.makedirs(DATA_DIR, exist_ok=True)

# Lista de tickers relevantes para volatilidad
TICKERS = ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "GOOG"]

# Periodo de datos históricos
START = "2018-01-01"
END = "2025-09-01"

for ticker in TICKERS:
    print(f"Descargando {ticker}...")
    df = yf.download(ticker, start=START, end=END, auto_adjust=True)
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(path)
    print(f"Guardado en: {path}")

print("✅ Datos descargados y listos en:", DATA_DIR)
