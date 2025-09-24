# models.py - L2 Tactical data models
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd


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
    def __init__(self,
                 symbol: str,
                 strength: float,
                 confidence: float,
                 side: str,
                 quantity: float = None,  # Añadir quantity como parámetro explícito
                 signal_type: str = None,
                 source: str = 'unknown',
                 features: dict = None,
                 timestamp = None,
                 metadata: dict = None,
                 **kwargs):
        self.symbol = symbol
        self.strength = strength
        self.confidence = confidence
        self.side = side
        self.quantity = quantity  # Guardar la cantidad explícita
        self.signal_type = signal_type or side
        self.source = source
        # Ensure features is always a dict
        if features is None:
            self.features = {}
        elif isinstance(features, dict):
            self.features = features.copy()
        else:
            # If features is not a dict, create empty dict and log warning
            print(f"WARNING: features parameter is {type(features)} instead of dict, setting to empty dict")
            self.features = {}
        self.timestamp = pd.Timestamp.now() if timestamp is None else pd.to_datetime(timestamp)
        self.metadata = metadata or {}
        for key, value in kwargs.items():
            # Don't overwrite features if it's being set via kwargs
            if key != 'features':
                setattr(self, key, value)

    def is_long(self) -> bool:
        return self.side.lower() == "buy"

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d
        
    def to_order_signal(self) -> Dict[str, Any]:
        """Convert TacticalSignal to a format expected by OrderManager"""
        import uuid
        
        # Extract technical indicators from features
        tech_indicators = {}
        if self.features:
            tech_indicators = {
                'rsi': self.features.get('rsi', 50),
                'macd': self.features.get('macd', 0),
                'macd_signal': self.features.get('macd_signal', 0),
                'sma_20': self.features.get('sma_20', 0),
                'sma_50': self.features.get('sma_50', 0),
                'vol_zscore': self.features.get('vol_zscore', 0),
                'bollinger_upper': self.features.get('bollinger_upper', 0),
                'bollinger_lower': self.features.get('bollinger_lower', 0)
            }
        
        # Get price from features if not set directly
        price = getattr(self, 'price', None)
        if price is None and 'close' in self.features:
            price = self.features['close']
        
        from .utils import safe_float
        return {
            'signal_id': str(uuid.uuid4()),
            'strategy_id': 'L2_TACTIC',
            'symbol': self.symbol,
            'side': self.side.lower(),
            'type': getattr(self, 'type', 'market'),
            'order_type': 'market',
            'qty': safe_float(self.quantity) if self.quantity is not None else None,
            'price': price,
            'stop_loss': getattr(self, 'stop_loss', None),
            'take_profit': getattr(self, 'take_profit', None),
            'strength': self.strength,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'technical_indicators': tech_indicators,
            'features': self.features,
            'metadata': {
                **self.metadata,
                'source': self.source,
                'signal_type': getattr(self, 'signal_type', self.side)
            }
        }
        
    def __str__(self) -> str:
        return f"TacticalSignal({self.symbol}, {self.side}, strength={self.strength:.3f}, confidence={self.confidence:.3f})"

    @property
    def action(self) -> str:
        """Alias for side attribute for backward compatibility"""
        return self.side

    @action.setter
    def action(self, value: str):
        """Set action (maps to side)"""
        self.side = value

    @property
    def features(self):
        """Get features, ensuring it's always a dict"""
        return self._features

    @features.setter
    def features(self, value):
        """Set features, ensuring it's always a dict"""
        if value is None:
            self._features = {}
        elif isinstance(value, dict):
            self._features = value.copy()
        else:
            print(f"WARNING: Attempted to set features to {type(value)} instead of dict, setting to empty dict")
            self._features = {}


@dataclass
class PositionSize:
    """Tactical position sizing result"""
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
    leverage: float = 1.0
    margin_required: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class MarketFeatures:
    """Market features for position sizing and risk control"""
    volatility: float
    atr: float
    support: float
    resistance: float
    adv_notional: float
    volume: float
    price: float
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    liquidity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Tactical risk metrics for position evaluation"""
    symbol: str
    timestamp: datetime
    position_risk: float  # risk amount for this position
    portfolio_heat: float  # current portfolio utilization 0-1
    correlation_risk: float  # correlation-based risk adjustment
    liquidity_risk: float  # liquidity-based risk adjustment
    volatility_risk: float  # volatility-based risk adjustment
    max_drawdown_risk: float  # drawdown-based risk adjustment
    total_risk_score: float  # composite risk score 0-1
    risk_limit_breached: bool = False
    risk_warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class L2State:
    """L2 tactical layer state"""
    signals: List[TacticalSignal] = field(default_factory=list)
    market_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    regime: str = "neutral"
    allocation: Optional[Dict[str, Any]] = None
    active_signals: List[TacticalSignal] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    last_update: Optional[datetime] = None
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    count: int = 0
    symbol: str = ""
    regime_context: Dict[str, Any] = field(default_factory=dict)
    generator_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        if self.last_update:
            data["last_update"] = self.last_update.isoformat()
        # Convert signals to dicts
        data["signals"] = [s.asdict() if hasattr(s, 'asdict') else s for s in self.signals]
        data["active_signals"] = [s.asdict() if hasattr(s, 'asdict') else s for s in self.active_signals]
        return data
