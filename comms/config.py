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

# Ejemplo de definición en comms/config.py
TECHNICAL_THRESHOLD = float(os.getenv("TECHNICAL_THRESHOLD", 0.1))
FINRL_THRESHOLD = float(os.getenv("FINRL_THRESHOLD", 0.2))
MEAN_REVERSION_THRESHOLD = float(os.getenv("MEAN_REVERSION_THRESHOLD", 0.3))
MIN_SIGNAL_STRENGTH = float(os.getenv("MIN_SIGNAL_STRENGTH", 0.4))
ENABLED_GENERATORS = os.getenv("ENABLED_GENERATORS", "technical,finrl,mean_reversion").split(',')