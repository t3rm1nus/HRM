# train_portfolio_model.py
"""
Entrena el modelo de optimización de portafolio usando:
- Modelo de Markowitz (Frontera Eficiente)
- Black-Litterman (combina expectativas de mercado con opiniones de expertos)
"""

import os
import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier, BlackLittermanModel

# Configuración de rutas
DATA_DIR = "data/datos_para_modelos_l3/portfolio"
MODEL_DIR = "models/L3/portfolio"
os.makedirs(MODEL_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "GOOG", "BTC-USD", "ETH-USD"]

# Cargar precios históricos
prices = pd.DataFrame()
for ticker in TICKERS:
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Usar columna 'Close' como referencia
        if "Adj Close" in df.columns:
            prices[ticker] = pd.to_numeric(df["Adj Close"], errors='coerce')
        elif "Close" in df.columns:
            prices[ticker] = pd.to_numeric(df["Close"], errors='coerce')
        else:
            print(f"⚠️ {ticker} no tiene columna Close/Adj Close, se salta.")
    else:
        print(f"⚠️ {ticker}.csv no encontrado, se salta.")

# Eliminar filas con NaN
prices.dropna(inplace=True)

if prices.empty:
    raise ValueError("No se cargaron datos de precios válidos. Verifica los CSV.")

# Calcular retornos esperados y matriz de covarianza
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

# Capitalización de mercado de cada activo (para pi="market")
market_caps = pd.Series({
    "AAPL": 3.0e12,
    "MSFT": 2.5e12,
    "GOOG": 1.5e12,
    "BTC-USD": 5.0e11,
    "ETH-USD": 2.5e11
})

# Opiniones de experto (ejemplo)
P = np.array([[1, -1, 0, 0, 0]])  # Comparación AAPL vs MSFT
Q = np.array([0.02])              # AAPL > MSFT en 2%

# Crear modelo Black-Litterman
bl = BlackLittermanModel(S, pi="market", market_caps=market_caps, P=P, Q=Q)
ret_bl = bl.bl_returns()
cov_bl = bl.bl_cov()

# Optimización con Frontera Eficiente
ef = EfficientFrontier(ret_bl, cov_bl)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

# Guardar resultados
weights_path = os.path.join(MODEL_DIR, "bl_weights.csv")
cov_path = os.path.join(MODEL_DIR, "bl_cov.csv")

pd.DataFrame(cleaned_weights, index=["weight"]).T.to_csv(weights_path)
cov_bl.to_csv(cov_path)

print("Modelo entrenado y guardado en:", MODEL_DIR)
print("Pesos óptimos:\n", cleaned_weights)
