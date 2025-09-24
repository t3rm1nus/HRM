"""
Configuración global para el sistema HRM.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Variables sueltas para compatibilidad con módulos existentes
from l2_tactic.utils import safe_float

RISK_LIMIT_BTC = safe_float(os.getenv("RISK_LIMIT_BTC", 0.05))
RISK_LIMIT_ETH = safe_float(os.getenv("RISK_LIMIT_ETH", 1.0))
EXPOSURE_MAX_BTC = safe_float(os.getenv("EXPOSURE_MAX_BTC", 0.20))
EXPOSURE_MAX_ETH = safe_float(os.getenv("EXPOSURE_MAX_ETH", 0.15))
CORRELATION_LIMIT = safe_float(os.getenv("CORRELATION_LIMIT", 0.80))
TECHNICAL_THRESHOLD = safe_float(os.getenv("TECHNICAL_THRESHOLD", 0.1))
FINRL_THRESHOLD = safe_float(os.getenv("FINRL_THRESHOLD", 0.2))
MEAN_REVERSION_THRESHOLD = safe_float(os.getenv("MEAN_REVERSION_THRESHOLD", 0.3))
MIN_SIGNAL_STRENGTH = safe_float(os.getenv("MIN_SIGNAL_STRENGTH", 0.4))
ENABLED_GENERATORS = os.getenv("ENABLED_GENERATORS", "technical,finrl,mean_reversion").split(',')
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
USE_TESTNET = os.getenv("USE_TESTNET", "false").lower() == "true"  # Default to real data
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(',')
MODE = os.getenv("BINANCE_MODE", "TESTNET")

# Objeto config para módulos nuevos
config = {
    "SYMBOLS": SYMBOLS,
    "RISK_CONFIG": {
        "max_drawdown_limit": 0.01,
        "risk_limit_btc": RISK_LIMIT_BTC,
        "risk_limit_eth": RISK_LIMIT_ETH,
        "exposure_max_btc": EXPOSURE_MAX_BTC,
        "exposure_max_eth": EXPOSURE_MAX_ETH,
        "correlation_limit": CORRELATION_LIMIT
    },
    "FINRL_CONFIG": {
    "model_path": "models/L2/ai_model_data_multiasset.zip"
    },
    "SIGNALS": {
        "universe": SYMBOLS,
        "technical_threshold": TECHNICAL_THRESHOLD,
        "finrl_threshold": FINRL_THRESHOLD,
        "mean_reversion_threshold": MEAN_REVERSION_THRESHOLD,
        "min_signal_strength": MIN_SIGNAL_STRENGTH,
        "enabled_generators": ENABLED_GENERATORS
    },
    "BINANCE_API_KEY": BINANCE_API_KEY,
    "BINANCE_API_SECRET": BINANCE_API_SECRET,
    "USE_TESTNET": USE_TESTNET,
    "MODE": MODE,
    "L2_CONFIG": {}
}
