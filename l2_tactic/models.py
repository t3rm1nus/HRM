# models.py - L2 Tactical data models
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

class SignalDirection(Enum):
    LONG = "buy"
    SHORT = "sell"
    NEUTRAL = "hold"

class SignalSource(Enum):
    AI = "ai"
    TECHNICAL = "technical"
    PATTERN = "pattern"
    COMPOSITE = "composite"

@dataclass
class TacticalSignal:
    symbol: str
    side: str
    strength: float
    confidence: float
    price: float
    timestamp: datetime = field(default_factory=lambda: pd.Timestamp.now(tz="UTC"))
    horizon: str = "1h"
    source: str = "l2_tactical"
    model_name: Optional[str] = None
    features_used: Dict[str, float] = field(default_factory=dict)
    reasoning: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def is_long(self) -> bool:
        return self.side.lower() == "buy"

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

@dataclass
class MarketFeatures:
    volatility: Optional[float] = None
    volume_ratio: Optional[float] = None
    price_momentum: Optional[float] = None
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    atr: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    spread_bps: Optional[float] = None
    liquidity_score: Optional[float] = None

@dataclass
class PositionSize:
    symbol: str
    side: str
    price: float
    size: float
    notional: float
    risk_amount: float
    kelly_fraction: float
    vol_target_leverage: float
    max_loss: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: Optional[float] = None
    margin_required: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RiskMetrics:
    var_95: Optional[float] = None
    expected_shortfall: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    volatility: Optional[float] = None
    correlation_impact: Optional[float] = None
    beta: Optional[float] = None
    liquidity_score: Optional[float] = None

@dataclass
class StrategicDecision:
    regime: str = "neutral"
    target_exposure: float = 0.5
    risk_appetite: str = "moderate"
    preferred_assets: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    time_horizon: str = "1h"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class L2State:
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

    def get_active_signals(self) -> List[TacticalSignal]:
        return self.signals

    @property
    def active_signals(self) -> List[TacticalSignal]:
        return self.get_active_signals()

    def __len__(self) -> int:
        return len(self.signals)