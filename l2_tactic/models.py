# models.py - L2 Tactical data models
from __future__ import annotations
from enum import Enum



from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

class SignalDirection(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class SignalSource(Enum):
    AI = "ai_model"
    TECHNICAL = "technical"
    PATTERN = "pattern"
    COMPOSITE = "composite"
# ---------------------------
# Señal táctica (output L2)
# ---------------------------
@dataclass
class TacticalSignal:
    symbol: str
    side: str                     # "buy" | "sell"
    strength: float               # 0..1
    confidence: float             # 0..1 (calidad/convicción)
    price: float                  # precio de referencia (close/last)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    horizon: str = "1h"
    source: str = "l2_tactical"
    model_name: Optional[str] = None
    features_used: Dict[str, float] = field(default_factory=dict)
    reasoning: Optional[str] = None

    # niveles sugeridos por el generador/AI (opcionales)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def is_long(self) -> bool:
        return self.side.lower() == "buy"

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


# ---------------------------------
# Features de mercado (input L2)
# ---------------------------------
@dataclass
class MarketFeatures:
    volatility: Optional[float] = None           # vol anualizada (0.2 = 20%)
    volume_ratio: Optional[float] = None         # vol actual / vol media
    price_momentum: Optional[float] = None       # retorno acumulado de ventana corta
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None            # "bullish"/"bearish"/"neutral"
    atr: Optional[float] = None                  # Average True Range (en unidades de precio)
    support: Optional[float] = None
    resistance: Optional[float] = None
    spread_bps: Optional[float] = None           # coste implícito
    liquidity_score: Optional[float] = None      # 0..1


# ---------------------------------
# Tamaño de posición (sizing L2)
# ---------------------------------
@dataclass
class PositionSize:
    symbol: str
    side: str                        # "buy" | "sell"
    price: float                     # precio de entrada estimado
    size: float                      # cantidad de unidades (ej. BTC)
    notional: float                  # tamaño nocional = size * price
    risk_amount: float               # capital en riesgo (moneda base)
    kelly_fraction: float            # fracción Kelly aplicada (0..1)
    vol_target_leverage: float       # multiplicador de apalancamiento por vol-targeting
    max_loss: float                  # pérdida máxima tolerada en este trade
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: Optional[float] = None
    margin_required: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------
# Métricas de riesgo (diagnóstico)
# ---------------------------------
@dataclass
class RiskMetrics:
    var_95: Optional[float] = None            # Value-at-Risk al 95% (en % o fracción)
    expected_shortfall: Optional[float] = None
    max_drawdown: Optional[float] = None      # máximo DD histórico (fracción)
    sharpe_ratio: Optional[float] = None
    volatility: Optional[float] = None        # vol anualizada
    correlation_impact: Optional[float] = None
    beta: Optional[float] = None
    liquidity_score: Optional[float] = None


# ---------------------------------
# Decisiones de L3 que condicionan L2
# ---------------------------------
@dataclass
class StrategicDecision:
    regime: str = "neutral"                    # bull/bear/range/neutral...
    target_exposure: float = 0.5               # 0..1
    risk_appetite: str = "moderate"            # "conservative" | "moderate" | "aggressive"
    preferred_assets: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    time_horizon: str = "1h"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------
# Estado interno de L2 para integración
# ---------------------------------
@dataclass
class L2State:
    """Estado del módulo L2_tactic"""
    signals: List[TacticalSignal] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    last_update: Optional[datetime] = None

    def add_signal(self, signal: TacticalSignal) -> None:
        self.signals.append(signal)
        self.last_update = datetime.utcnow()

    def cleanup_expired(self, expiry_minutes: int = 15) -> None:
        if not self.signals:
            return
        now = datetime.utcnow()
        expiry = timedelta(minutes=expiry_minutes)
        self.signals = [
            s for s in self.signals
            if (now - s.timestamp) < expiry
        ]

    def clear(self) -> None:
        self.signals.clear()
        self.metrics.clear()
        self.last_update = datetime.utcnow()

    def update_metrics(self, new_metrics: Dict) -> None:
        self.metrics.update(new_metrics)
        self.last_update = datetime.utcnow()

    def get_active_signals(self) -> List[TacticalSignal]:
        return self.signals

    @property
    def active_signals(self) -> List[TacticalSignal]:
        """Alias para compatibilidad con código existente"""
        return self.get_active_signals()

    def __len__(self) -> int:
        return len(self.signals)
