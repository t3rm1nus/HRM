# l1_operational/models.py
"""
Modelos de datos para L1_operational.
Define las estructuras de entrada y salida que L1 debe manejar.
"""

from typing import Literal, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Tipos literales para validación
Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "ioc", "post_only"]
OrderStatus = Literal["accepted", "rejected", "partial_fill", "filled", "canceled", "expired"]

@dataclass
class Signal:
    """
    Señal de trading recibida desde L2/L3.
    L1 solo ejecuta, no modifica estas señales.
    """
    signal_id: str
    strategy_id: str
    timestamp: float
    symbol: str
    side: Side
    qty: float
    order_type: OrderType
    price: Optional[float] = None
    time_in_force: Optional[str] = None
    risk: Dict[str, float] = None  # max_slippage_bps, stop_loss, take_profit, max_notional
    metadata: Dict[str, Any] = None  # confidence, rationale, etc.

    def __post_init__(self):
        if self.risk is None:
            self.risk = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class OrderIntent:
    """
    Intención de orden validada por L1.
    Representa la orden que se enviará al exchange.
    """
    client_order_id: str
    symbol: str
    side: Side
    type: OrderType
    qty: float
    price: Optional[float] = None
    time_in_force: Optional[str] = None
    route: Literal["PAPER", "LIVE", "REPLAY"] = "PAPER"

@dataclass
class ExecutionReport:
    """
    Reporte de ejecución enviado a L2/L3.
    Contiene el estado final de la orden y métricas de ejecución.
    """
    client_order_id: str
    status: OrderStatus
    filled_qty: float = 0.0
    avg_price: float = 0.0
    fees: float = 0.0
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    error_code: Optional[str] = None
    error_msg: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

@dataclass
class RiskAlert:
    """
    Alerta de riesgo generada por L1.
    Se envía a L2/L3 para notificar problemas de riesgo.
    """
    alert_id: str
    alert_type: Literal["fat_finger", "slippage_exceeded", "limit_breached", "kill_switch_triggered"]
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

@dataclass
class MarketData:
    """
    Datos de mercado obtenidos por L1.
    Se usan para validaciones de riesgo y ejecución.
    """
    symbol: str
    last_price: float
    bid: float
    ask: float
    timestamp: float
    volume_24h: Optional[float] = None
    spread_bps: Optional[float] = None

    def __post_init__(self):
        if self.spread_bps is None and self.bid and self.ask:
            self.spread_bps = ((self.ask - self.bid) / self.bid) * 10000
