# l1_operational/config.py
"""
Configuración centralizada de L1_operational.

Fase 1:
- Operación exclusivamente PAPER/SIMULATED.
- LIVE y TESTNET permanecen deshabilitados.
- Los límites de riesgo se centralizan en este módulo.
"""

from typing import Dict, Optional, List, Any

# Modo de operación permitido durante la Fase 1.
OPERATION_MODE = "PAPER"

# Límites de riesgo por operación.
RISK_LIMITS = {
    "MAX_ORDER_SIZE_BTC": 0.05,
    "MAX_ORDER_SIZE_USDT": 1500,
    "MIN_ORDER_SIZE_USDT": 2,
    "MAX_ORDER_SIZE_ETH": 0.5,
    "MAX_ORDER_SIZE_ADA": 100,
}

# Límites de riesgo por cartera.
PORTFOLIO_LIMITS = {
    "MAX_PORTFOLIO_EXPOSURE_BTC": 0.40,
    "MAX_PORTFOLIO_EXPOSURE_ETH": 0.40,
    "MAX_POSITION_SIZE_USDT": 1200,
    "MIN_USDT_RESERVE": 0.20,
    "REBALANCE_THRESHOLD": 0.15,
    "REBALANCE_TRIGGER_THRESHOLD": 5,
    "REBALANCE_INTERVAL": 60,
    "REBALANCE_MIN_AMOUNT": 500,
    "ROTATION_AMOUNT": 0.25,
    "MAX_DAILY_DRAWDOWN": 0.08,
    "MIN_ACCOUNT_BALANCE_USDT": 500,
    "MAX_LEVERAGE": 1.0,
    "ENABLE_AUTO_REBALANCE": True,
    "REBALANCE_CHECKSUM_VERIFICATION": True,
    "REBALANCE_DRY_RUN_ENABLED": False,
    "REBALANCE_FEES_BUFFER": 1.01,
    "REBALANCE_COOLDOWN_EXTENDED": 300,
    "REBALANCE_MIN_ORDER_CHECK": True,
    "REBALANCE_FORCE_INTERVAL": 100,
}

# Configuración de ejecución.
EXECUTION_CONFIG = {
    "DEFAULT_ORDER_TYPE": "market",
    "MAX_SLIPPAGE_BPS": 50,
    "ORDER_TIMEOUT_SECONDS": 30,
    "RETRY_ATTEMPTS": 3,
    "PAPER_MODE": True,
}

# Configuración de alertas.
ALERT_CONFIG = {
    "ENABLE_RISK_ALERTS": True,
    "ENABLE_EXECUTION_ALERTS": True,
    "ENABLE_PERFORMANCE_ALERTS": True,
}

# Configuración de logging.
LOGGING_CONFIG = {
    "LEVEL": "INFO",
    "FORMAT": (
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | "
        "{name}:{function}:{line} | {message}"
    ),
    "ENABLE_FILE_LOGGING": True,
    "LOG_FILE": "l1_operational.log",
}

# Umbrales de IA.
TREND_THRESHOLD = 0.1

# Rutas de modelos de IA.
AI_MODELS = {
    "MODELO1_PATH": "models/L1/modelo1_lr.pkl",
    "MODELO2_PATH": "models/L1/modelo2_rf.pkl",
    "MODELO3_PATH": "models/L1/modelo3_lgbm.pkl",
    "AI_CONFIDENCE_THRESHOLD": 0.4,
    "ENSEMBLE_THRESHOLD": TREND_THRESHOLD,
}


class ConfigObject:
    """
    Objeto de configuración compatible con acceso por instancia
    y por clase.

    Durante la Fase 1, el modo no se puede modificar mediante
    variables de entorno.
    """

    OPERATION_MODE = OPERATION_MODE
    RISK_LIMITS = RISK_LIMITS
    PORTFOLIO_LIMITS = PORTFOLIO_LIMITS
    EXECUTION_CONFIG = EXECUTION_CONFIG
    ALERT_CONFIG = ALERT_CONFIG
    LOGGING_CONFIG = LOGGING_CONFIG
    TREND_THRESHOLD = TREND_THRESHOLD
    AI_MODELS = AI_MODELS

    # Valores fijos de seguridad para Fase 1.
    BINANCE_API_KEY = ""
    BINANCE_API_SECRET = ""
    BINANCE_MODE = "paper"
    USE_TESTNET = False

    def __init__(self):
        self.OPERATION_MODE = OPERATION_MODE
        self.RISK_LIMITS = RISK_LIMITS
        self.PORTFOLIO_LIMITS = PORTFOLIO_LIMITS
        self.EXECUTION_CONFIG = EXECUTION_CONFIG
        self.ALERT_CONFIG = ALERT_CONFIG
        self.LOGGING_CONFIG = LOGGING_CONFIG
        self.TREND_THRESHOLD = TREND_THRESHOLD
        self.AI_MODELS = AI_MODELS

        # No se leen credenciales ni modos externos en Fase 1.
        self.BINANCE_API_KEY = ""
        self.BINANCE_API_SECRET = ""
        self.BINANCE_MODE = "paper"
        self.USE_TESTNET = False

    def get_risk_limit(
        self,
        asset: str,
        limit_type: Optional[str] = None,
    ) -> float:
        """Devuelve el límite de tamaño de orden para un activo."""
        if not isinstance(asset, str) or not asset.strip():
            return 0.0

        clave = f"MAX_ORDER_SIZE_{asset.strip().upper()}"
        return float(self.RISK_LIMITS.get(clave, 0))

    def get_portfolio_limit(self, asset: str) -> float:
        """Devuelve el límite de exposición de cartera de un activo."""
        if not isinstance(asset, str) or not asset.strip():
            return 0.0

        clave = f"MAX_PORTFOLIO_EXPOSURE_{asset.strip().upper()}"
        return float(self.PORTFOLIO_LIMITS.get(clave, 0))

    def is_paper_mode(self) -> bool:
        """Indica si la configuración permite operar en PAPER."""
        return (
            self.OPERATION_MODE.upper() == "PAPER"
            and self.EXECUTION_CONFIG.get("PAPER_MODE") is True
            and self.BINANCE_MODE == "paper"
            and self.USE_TESTNET is False
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Permite consultar la configuración como un diccionario."""
        return getattr(self, key, default)

    @classmethod
    def get_class_risk_limit(
        cls,
        asset: str,
        limit_type: Optional[str] = None,
    ) -> float:
        """Devuelve un límite de riesgo desde la clase."""
        if not isinstance(asset, str) or not asset.strip():
            return 0.0

        clave = f"MAX_ORDER_SIZE_{asset.strip().upper()}"
        return float(cls.RISK_LIMITS.get(clave, 0))

    @classmethod
    def get_class_portfolio_limit(cls, asset: str) -> float:
        """Devuelve un límite de exposición desde la clase."""
        if not isinstance(asset, str) or not asset.strip():
            return 0.0

        clave = f"MAX_PORTFOLIO_EXPOSURE_{asset.strip().upper()}"
        return float(cls.PORTFOLIO_LIMITS.get(clave, 0))


# Instancia compartida.
config = ConfigObject()

# Alias de compatibilidad.
Config = ConfigObject

__all__ = [
    "config",
    "Config",
    "ConfigObject",
    "OPERATION_MODE",
    "RISK_LIMITS",
    "PORTFOLIO_LIMITS",
    "EXECUTION_CONFIG",
    "ALERT_CONFIG",
    "LOGGING_CONFIG",
    "TREND_THRESHOLD",
    "AI_MODELS",
]