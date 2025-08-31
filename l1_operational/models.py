# l1_operational/models.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import time

# ============================================================================
# NUEVAS CLASES AGREGADAS (para compatibilidad con imports)
# ============================================================================

class SignalType(Enum):
    """Tipos de señales de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    
    # Compatibilidad con tu sistema actual
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

# ============================================================================
# TUS CLASES EXISTENTES (mantenidas tal como están)
# ============================================================================

@dataclass
class Signal:
    """Señal de trading recibida de L2/L3"""
    signal_id: str
    strategy_id: str
    timestamp: float
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    order_type: str = "market"  # market, limit
    price: Optional[float] = None  # para limit orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.5
    technical_indicators: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = {}
    
    # Métodos de compatibilidad con los enums
    def get_signal_type(self) -> SignalType:
        """Convertir side a SignalType"""
        if self.side.lower() == 'buy':
            return SignalType.BUY
        elif self.side.lower() == 'sell':
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def get_asset_from_symbol(self) -> str:
        """Extraer el asset base del símbolo (ej: BTCUSDT -> BTC)"""
        if self.symbol.endswith('USDT'):
            return self.symbol[:-4]
        elif self.symbol.endswith('BUSD'):
            return self.symbol[:-4]
        elif self.symbol.endswith('USD'):
            return self.symbol[:-3]
        else:
            # Fallback: tomar los primeros 3-4 caracteres
            return self.symbol[:3] if len(self.symbol) >= 6 else self.symbol[:4]

@dataclass
class OrderIntent:
    """Intención de orden después de validaciones"""
    signal_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    created_timestamp: float = None

    def __post_init__(self):
        if self.created_timestamp is None:
            self.created_timestamp = time.time()

@dataclass
class ExecutionResult:
    """Resultado de ejecución del exchange"""
    order_id: str
    filled_qty: float
    avg_price: float
    fees: float
    latency_ms: float
    status: str  # FILLED, PARTIAL, REJECTED

@dataclass
class ExecutionReport:
    """Reporte completo de ejecución para el bus"""
    signal_id: str
    status: str  # EXECUTED, REJECTED_SAFETY, REJECTED_AI, EXECUTION_ERROR
    timestamp: float
    reason: Optional[str] = None
    executed_qty: Optional[float] = None
    executed_price: Optional[float] = None
    fees: Optional[float] = None
    latency_ms: Optional[float] = None
    ai_confidence: Optional[float] = None
    ai_risk_score: Optional[float] = None
    ai_model_votes: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class RiskAlert:
    """Alerta de riesgo generada por el sistema"""
    alert_id: str
    level: str  # WARNING, CRITICAL
    message: str
    signal_id: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class ValidationResult:
    """Resultado de validación de riesgo"""
    is_valid: bool
    reason: str = ""
    risk_score: float = 0.0
    warnings: Optional[List[str]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

# ============================================================================
# FUNCIONES HELPER PARA COMPATIBILIDAD
# ============================================================================

def create_signal(
    signal_id: str,
    symbol: str,
    side: str,  # 'buy' o 'sell'
    qty: float,
    strategy_id: str = "L2_TACTIC",
    order_type: str = "market",
    price: Optional[float] = None,
    confidence: float = 0.5,
    timestamp: Optional[float] = None
) -> Signal:
    """
    Función helper para crear señales fácilmente
    """
    return Signal(
        signal_id=signal_id,
        strategy_id=strategy_id,
        timestamp=timestamp or time.time(),
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        price=price,
        confidence=confidence
    )

# ============================================================================
# EXPORTACIONES
# ============================================================================

__all__ = [
    # Enums nuevos
    'SignalType',
    'SignalSource', 
    'OrderStatus',
    'ExecutionStatus',
    
    # Clases existentes
    'Signal',
    'OrderIntent',
    'ExecutionResult',
    'ExecutionReport',
    'RiskAlert',
    'ValidationResult',
    
    # Helper functions
    'create_signal',
]