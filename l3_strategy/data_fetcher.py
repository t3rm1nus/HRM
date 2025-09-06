"""
Data fetcher
"""
import os
import pandas as pd
from utils import logging, mock_prices

DATA_DIR = "data/datos_para_modelos_l3"
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "GOOG", "BTC-USD", "ETH-USD"]

# Mock de precios
prices = mock_prices(TICKERS)
for t in TICKERS:
    path = os.path.join(DATA_DIR, f"{t}.csv")
    prices[[t]].to_csv(path)
    logging.info(f"Mock de {t} guardado en {path}")

# Aquí podrías añadir indicadores macro y sentiment si lo deseas
