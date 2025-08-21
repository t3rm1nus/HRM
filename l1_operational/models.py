# l1_operational/models.py
"""
Estructuras de datos centrales para L1:
- Signal: señal proveniente de L2/L3
- ExecutionReport: resultado de ejecución de órdenes
- RiskAlert: alertas de riesgo generadas por validaciones hard-coded o IA
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class Signal:
    signal_id: str                    # ID único de la señal
    strategy_id: str                  # Estrategia que generó la señal
    timestamp: float                  # Timestamp de creación de la señal
    symbol: str                       # Símbolo a operar (BTC/USDT, ETH/USDT, etc.)
    side: str                         # 'buy' o 'sell'
    qty: float                        # Cantidad a operar
    order_type: str = "market"        # 'market' o 'limit'
    price: Optional[float] = None     # Precio para orden limit
    stop_loss: Optional[float] = None # Stop-loss obligatorio para validación de riesgo
    risk: Dict[str, Any] = field(default_factory=dict)  # Parámetros de riesgo, e.g., max_slippage_bps
    metadata: Dict[str, Any] = field(default_factory=dict)  # Info adicional, e.g., confianza, notas


@dataclass
class ExecutionReport:
    client_order_id: str              # ID local de trazabilidad
    exchange_order_id: Optional[str] = None  # ID asignado por el exchange
    status: str = "pending"           # 'pending', 'filled', 'rejected', 'failed'
    filled_qty: float = 0.0
    avg_price: Optional[float] = None
    fees: float = 0.0
    slippage_bps: Optional[float] = None
    latency_ms: Optional[float] = None
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    message: Optional[str] = None     # Mensaje de error o información adicional


@dataclass
class RiskAlert:
    alert_id: str                     # ID único de alerta
    signal_id: Optional[str] = None   # Relacionada a qué señal
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    severity: str = "medium"          # 'low', 'medium', 'high'
    type: str = "risk_limit"          # Tipo de alerta: 'risk_limit', 'liquidity', 'stop_loss', etc.
    message: str = ""                  # Descripción detallada de la alerta
    metadata: Dict[str, Any] = field(default_factory=dict)  # Info adicional, e.g., riesgo detectado


@dataclass
class OrderIntent:
    """
    Representa un intento de ejecución determinista derivado de una Signal.
    """
    client_order_id: str
    symbol: str
    side: str                         # 'buy' o 'sell'
    qty: float
    type: str = "market"               # 'market' o 'limit'
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    created_at: float = field(default_factory=lambda: datetime.utcnow().timestamp())
