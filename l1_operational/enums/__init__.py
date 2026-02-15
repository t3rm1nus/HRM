# Empty __init__.py file for l1_operational.enums package

from enum import Enum

class L1SignalType(Enum):
    """Tipos de señales L1"""
    MOMENTUM_SHORT = "momentum_short"
    MOMENTUM_MEDIUM = "momentum_medium"
    TECHNICAL_RSI = "technical_rsi"
    TECHNICAL_MACD = "technical_macd"
    TECHNICAL_BOLLINGER = "technical_bollinger"
    VOLUME_FLOW = "volume_flow"
    VOLUME_LIQUIDITY = "volume_liquidity"

class SignalType(Enum):
    """Tipos de señales de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

    # Compatibilidad con nuestro sistema actual
    buy = "buy"
    sell = "sell"

class SignalSource(Enum):
    """Fuentes de las señales"""
    L2_TACTIC = "L2_TACTIC"
    L3_STRATEGY = "L3_STRATEGY"
    MANUAL = "MANUAL"
    RISK_MANAGER = "RISK_MANAGER"

class OrderStatus(Enum):
    """Estados de órdenes"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class ExecutionStatus(Enum):
    """Estados de ejecución para reportes"""
    EXECUTED = "EXECUTED"
    REJECTED_SAFETY = "REJECTED_SAFETY"
    REJECTED_AI = "REJECTED_AI"
    EXECUTION_ERROR = "EXECUTION_ERROR"
