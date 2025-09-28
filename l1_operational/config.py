# l1_operational/config.py
"""
Configuración centralizada de L1_operational.
Contiene todos los límites de riesgo y parámetros de ejecución.
"""
import os

# Modo de operación
OPERATION_MODE = "PAPER"  # "paper", "live", "development", "testing"

# Límites de riesgo por operación - OPTIMIZADOS PARA 3K USDT
RISK_LIMITS = {
    "MAX_ORDER_SIZE_BTC": 0.05,      # ~$5,420 por orden (más agresivo)
    "MAX_ORDER_SIZE_USDT": 1500,     # Máximo 50% del capital por orden
    "MIN_ORDER_SIZE_USDT": 5,        # Mínimo muy reducido para máximo volumen de trading
    "MAX_ORDER_SIZE_ETH": 0.5,       # ~$2,185 por orden (más agresivo)
    "MAX_ORDER_SIZE_ADA": 100,       # Sin cambios
}
# Límites de riesgo por portafolio - OPTIMIZADOS PARA ROTACIÓN
PORTFOLIO_LIMITS = {
    "MAX_PORTFOLIO_EXPOSURE_BTC": 0.40,  # máximo 40% del portafolio en BTC
    "MAX_PORTFOLIO_EXPOSURE_ETH": 0.40,  # máximo 40% del portafolio en ETH
    "MAX_POSITION_SIZE_USDT": 1200,      # máximo $1200 por posición individual (40% de $3000)
    "MIN_USDT_RESERVE": 0.20,            # mínimo 20% siempre en USDT libre
    "REBALANCE_THRESHOLD": 0.15,         # rebalancear si USDT < 15% del total
    "ROTATION_AMOUNT": 0.25,             # vender 25% cuando se active rotación
    "MAX_DAILY_DRAWDOWN": 0.08,          # aumentado a 8% de drawdown diario
    "MIN_ACCOUNT_BALANCE_USDT": 500,     # mínimo $500 USDT libre para operaciones
    "MAX_LEVERAGE": 1.0,                 # sin apalancamiento
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
TREND_THRESHOLD = 0.1

# Rutas de modelos IA
AI_MODELS = {
    "MODELO1_PATH": "models/L1/modelo1_lr.pkl",
    "MODELO2_PATH": "models/L1/modelo2_rf.pkl", 
    "MODELO3_PATH": "models/L1/modelo3_lgbm.pkl",
    "AI_CONFIDENCE_THRESHOLD": 0.4,
    "ENSEMBLE_THRESHOLD": TREND_THRESHOLD,  # Usar tu TREND_THRESHOLD existente
}

# ============================================================================
# SOLUCIÓN AL ERROR DE IMPORTACIÓN - MEJORADA
# ============================================================================

class ConfigObject:
    """
    Objeto de configuración que encapsula todas las constantes.
    Soporta acceso tanto como clase como instancia.
    """
    
    # Atributos de clase (para acceso directo tipo ConfigObject.RISK_LIMITS)
    OPERATION_MODE = 'TESTNET'
    RISK_LIMITS = RISK_LIMITS
    PORTFOLIO_LIMITS = PORTFOLIO_LIMITS
    EXECUTION_CONFIG = EXECUTION_CONFIG
    ALERT_CONFIG = ALERT_CONFIG
    LOGGING_CONFIG = LOGGING_CONFIG
    TREND_THRESHOLD = TREND_THRESHOLD
    AI_MODELS = AI_MODELS
    
    def __init__(self):
        # Atributos de instancia (para acceso tipo config.RISK_LIMITS)
        self.OPERATION_MODE = OPERATION_MODE
        self.RISK_LIMITS = RISK_LIMITS
        self.PORTFOLIO_LIMITS = PORTFOLIO_LIMITS
        self.EXECUTION_CONFIG = EXECUTION_CONFIG
        self.ALERT_CONFIG = ALERT_CONFIG
        self.LOGGING_CONFIG = LOGGING_CONFIG
        self.TREND_THRESHOLD = TREND_THRESHOLD
        self.AI_MODELS = AI_MODELS
        
        # Configuración de Binance para compatibilidad
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
        self.BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
        self.BINANCE_MODE = os.getenv('BINANCE_MODE', OPERATION_MODE.upper())
        self.USE_TESTNET = os.getenv('USE_TESTNET', 'true').lower() == 'true'
        
        # También como atributos de clase para compatibilidad total
        ConfigObject.BINANCE_API_KEY = self.BINANCE_API_KEY
        ConfigObject.BINANCE_API_SECRET = self.BINANCE_API_SECRET
        ConfigObject.BINANCE_MODE = self.BINANCE_MODE
        ConfigObject.USE_TESTNET = self.USE_TESTNET
    
    def get_risk_limit(self, asset, limit_type=None):
        """Obtener límite de riesgo específico"""
        key = f"MAX_ORDER_SIZE_{asset.upper()}"
        return self.RISK_LIMITS.get(key, 0)
    
    def get_portfolio_limit(self, asset):
        """Obtener límite de portafolio específico"""
        key = f"MAX_PORTFOLIO_EXPOSURE_{asset.upper()}"
        return self.PORTFOLIO_LIMITS.get(key, 0)
    
    def is_paper_mode(self):
        """Verificar si estamos en modo paper"""
        return self.OPERATION_MODE.upper() == "PAPER" or self.EXECUTION_CONFIG["PAPER_MODE"]
    
    @classmethod
    def get_class_risk_limit(cls, asset, limit_type=None):
        """Método de clase para obtener límites"""
        key = f"MAX_ORDER_SIZE_{asset.upper()}"
        return cls.RISK_LIMITS.get(key, 0)
    
    @classmethod 
    def get_class_portfolio_limit(cls, asset):
        """Método de clase para obtener límites de portafolio"""
        key = f"MAX_PORTFOLIO_EXPOSURE_{asset.upper()}"
        return cls.PORTFOLIO_LIMITS.get(key, 0)

# Crear la instancia que será importada
config = ConfigObject()

# IMPORTANTE: También exportar la clase para acceso directo
Config = ConfigObject

# Mantener compatibilidad: exportar todo lo que ya existía
__all__ = [
    'config',        # Instancia (config.RISK_LIMITS)
    'Config',        # Clase (Config.RISK_LIMITS)
    'ConfigObject',  # Alias para la clase
    'OPERATION_MODE',
    'RISK_LIMITS', 
    'PORTFOLIO_LIMITS',
    'EXECUTION_CONFIG',
    'ALERT_CONFIG',
    'LOGGING_CONFIG',
    'TREND_THRESHOLD',
    'AI_MODELS'
]
