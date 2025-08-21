# l1_operational/config.py
"""
Configuración centralizada de L1_operational.
Contiene todos los límites de riesgo y parámetros de ejecución.
"""

# Modo de operación
OPERATION_MODE = "paper"  # "paper", "live", "development", "testing"

# Límites de riesgo por operación
RISK_LIMITS = {
    "MAX_ORDER_SIZE_BTC": 0.05,      # máximo BTC por orden
    "MAX_ORDER_SIZE_USDT": 1000,     # máximo valor en USDT por orden
    "MIN_ORDER_SIZE_USDT": 10,       # mínimo valor en USDT por orden
    "MAX_ORDER_SIZE_ETH": 1.0,       # máximo ETH por orden
    "MAX_ORDER_SIZE_ADA": 1000,      # máximo ADA por orden
}

# Límites de riesgo por portafolio
PORTFOLIO_LIMITS = {
    "MAX_PORTFOLIO_EXPOSURE_BTC": 0.2,  # máximo 20% del portafolio en BTC
    "MAX_PORTFOLIO_EXPOSURE_ETH": 0.15, # máximo 15% del portafolio en ETH
    "MAX_DAILY_DRAWDOWN": 0.05,         # máximo 5% de drawdown diario
    "MIN_ACCOUNT_BALANCE_USDT": 20,     # umbral mínimo de capital (fixed key name)
    "MAX_LEVERAGE": 1.0,                # sin apalancamiento
}

# Configuración de ejecución
EXECUTION_CONFIG = {
    "DEFAULT_ORDER_TYPE": "market",
    "MAX_SLIPPAGE_BPS": 50,            # máximo 0.5% de slippage
    "ORDER_TIMEOUT_SECONDS": 30,       # timeout para órdenes
    "RETRY_ATTEMPTS": 3,               # intentos de reintento
    "PAPER_MODE": True,                # modo paper por defecto
}

# Configuración de alertas
ALERT_CONFIG = {
    "ENABLE_RISK_ALERTS": True,
    "ENABLE_EXECUTION_ALERTS": True,
    "ENABLE_PERFORMANCE_ALERTS": True,
}

# Configuración de logging
LOGGING_CONFIG = {
    "LEVEL": "INFO",
    "FORMAT": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "ENABLE_FILE_LOGGING": True,
    "LOG_FILE": "l1_operational.log",
}

# Umbrales IA
TREND_THRESHOLD = 0.6