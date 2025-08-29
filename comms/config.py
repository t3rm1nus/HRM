# comms/config.py
import os

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"  # Convertir a boolean
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(',')
MODE = os.getenv("BINANCE_MODE", "PAPER")  # Nuevo: modo de operación

# Límites de riesgo (pueden venir de .env o valores por defecto)
RISK_LIMIT_BTC = float(os.getenv("RISK_LIMIT_BTC", 0.05))
RISK_LIMIT_ETH = float(os.getenv("RISK_LIMIT_ETH", 1.0))
EXPOSURE_MAX_BTC = float(os.getenv("EXPOSURE_MAX_BTC", 0.20))
EXPOSURE_MAX_ETH = float(os.getenv("EXPOSURE_MAX_ETH", 0.15))
CORRELATION_LIMIT = float(os.getenv("CORRELATION_LIMIT", 0.80))