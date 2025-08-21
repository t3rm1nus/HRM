# l1_operational/models.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time

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
