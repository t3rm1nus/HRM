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
        self.quantity = quantity  # Guardar la cantidad explícitamente
        self.signal_type = signal_type or side
        self.source = source
        self.features = features or {}
        self.timestamp = pd.Timestamp.now() if timestamp is None else pd.to_datetime(timestamp)
        self.metadata = metadata or {}
        for key, value in kwargs.items():
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
        
        return {
            'signal_id': str(uuid.uuid4()),
            'strategy_id': 'L2_TACTIC',
            'symbol': self.symbol,
            'side': self.side.lower(),
            'type': getattr(self, 'type', 'market'),
            'order_type': 'market',
            'qty': float(self.quantity) if self.quantity is not None else None,
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
    # Agregar estos campos:
    adv_notional: Optional[float] = None  # Average Daily Volume in notional
    liquidity: Optional[float] = None     # Alias for liquidity metrics
    volume: Optional[float] = None        # Current volume
    price: Optional[float] = None         # Current price


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

    def __post_init__(self):
        # Initialize signals as empty list if None
        if self.signals is None:
            self.signals = []
        
        # Convert signals to list if possible
        if not isinstance(self.signals, list):
            try:
                self.signals = list(self.signals) if hasattr(self.signals, '__iter__') else []
            except (TypeError, ValueError):
                self.signals = []
        
        # Filter out non-TacticalSignal objects
        self.signals = [s for s in self.signals if isinstance(s, TacticalSignal)]
        
        # Initialize metrics as empty dict if None
        if not isinstance(self.metrics, dict):
            try:
                self.metrics = dict(self.metrics) if self.metrics is not None else {}
            except (TypeError, ValueError):
                self.metrics = {}
        
        # Set last_update if None
        if self.last_update is None:
            self.last_update = datetime.utcnow()
    
    def add_signal(self, signal: TacticalSignal) -> None:
        """Add a signal safely"""
        if isinstance(signal, TacticalSignal):
            if self.signals is None:
                self.signals = []
            self.signals.append(signal)
    
    def get_signals(self) -> List[TacticalSignal]:
        """Get signals safely"""
        if not isinstance(self.signals, list):
            self.signals = []
        return self.signals
    
    def clear_signals(self) -> None:
        """Clear signals"""
        self.signals = []
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
