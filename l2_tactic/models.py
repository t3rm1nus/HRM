"""
Modelos de datos para L2_tactic
===============================

Define las estructuras de datos específicas del nivel táctico.
"""
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class SignalDirection(Enum):
    """Dirección de la señal de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class SignalSource(Enum):
    """Fuente de la señal"""
    AI_MODEL = "AI_MODEL"
    TECHNICAL = "TECHNICAL"
    PATTERN = "PATTERN"
    COMPOSITE = "COMPOSITE",
    AGGREGATED = "aggregated"


@dataclass
class TacticalSignal:
    """
    Señal de trading generada por L2
    
    Attributes:
        symbol: Símbolo del activo (ej: "BTCUSDT")
        direction: Dirección de la operación
        strength: Fuerza de la señal [0.0-1.0]
        confidence: Confianza del modelo [0.0-1.0] 
        price: Precio de referencia
        timestamp: Momento de generación
        source: Fuente de la señal
        metadata: Información adicional específica de la fuente
        expires_at: Cuándo expira la señal (opcional)
    """
    symbol: str
    direction: SignalDirection
    strength: float
    confidence: float
    price: float
    timestamp: datetime
    source: SignalSource
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validaciones post-inicialización"""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Strength debe estar entre 0.0 y 1.0, got {self.strength}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence debe estar entre 0.0 y 1.0, got {self.confidence}")
        if self.price <= 0:
            raise ValueError(f"Price debe ser positivo, got {self.price}")
    
    @property
    def is_expired(self) -> bool:
        """Verifica si la señal ha expirado"""
        if self.expires_at is None:
            return False
        return pd.Timestamp.now(tz="UTC") > self.expires_at
    
    @property 
    def effective_strength(self) -> float:
        """Fuerza efectiva considerando confianza"""
        return self.strength * self.confidence


@dataclass
class PositionSize:
    """
    Resultado del cálculo de position sizing
    
    Attributes:
        symbol: Símbolo del activo
        quantity: Cantidad a operar (positiva para LONG, negativa para SHORT)
        stop_loss: Precio de stop loss
        take_profit: Precio de take profit (opcional)
        max_loss_usd: Pérdida máxima esperada en USD
        reasoning: Explicación del cálculo
        kelly_fraction: Fracción de Kelly utilizada
        risk_adjusted: Si fue ajustado por controles de riesgo
    """
    symbol: str
    quantity: float
    stop_loss: float
    take_profit: Optional[float]
    max_loss_usd: float
    reasoning: str
    kelly_fraction: float = 0.0
    risk_adjusted: bool = False
    
    def __post_init__(self):
        """Validaciones post-inicialización"""
        if self.quantity == 0:
            raise ValueError("Quantity no puede ser cero")
        if self.max_loss_usd < 0:
            raise ValueError(f"Max loss debe ser positivo, got {self.max_loss_usd}")


@dataclass
class RiskMetrics:
    """
    Métricas de riesgo para una operación
    
    Attributes:
        symbol: Símbolo del activo
        var_1d: Value at Risk 1 día (95%)
        expected_vol: Volatilidad esperada (anualizada)
        correlation_impact: Impacto por correlación con posiciones existentes
        liquidity_score: Score de liquidez [0.0-1.0]
        max_position_size: Tamaño máximo recomendado
        risk_score: Score de riesgo total [0.0-1.0]
    """
    symbol: str
    var_1d: float
    expected_vol: float
    correlation_impact: float
    liquidity_score: float
    max_position_size: float
    risk_score: float
    
    def __post_init__(self):
        """Validaciones post-inicialización"""
        if not 0.0 <= self.liquidity_score <= 1.0:
            raise ValueError(f"Liquidity score debe estar entre 0.0 y 1.0")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(f"Risk score debe estar entre 0.0 y 1.0")


@dataclass  
class L2State:
    """
    Estado interno del módulo L2
    
    Maintains cache of recent signals, risk metrics, and performance stats
    """
    active_signals: List[TacticalSignal] = field(default_factory=list)
    recent_positions: List[PositionSize] = field(default_factory=list)
    risk_cache: Dict[str, RiskMetrics] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_update: Optional[datetime] = None
    
    def add_signal(self, signal: TacticalSignal) -> None:
        """Añade una nueva señal al estado"""
        # Remover señales expiradas del mismo símbolo
        self.active_signals = [
            s for s in self.active_signals 
            if not (s.symbol == signal.symbol and s.is_expired)
        ]
        
        # Añadir nueva señal
        self.active_signals.append(signal)
        self.last_update = pd.Timestamp.now(tz="UTC")
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[TacticalSignal]:
        """Obtiene señales activas, opcionalmente filtradas por símbolo"""
        signals = [s for s in self.active_signals if not s.is_expired]
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
            
        return signals
    
    def cleanup_expired(self) -> None:
        """Limpia señales y datos expirados"""
        self.active_signals = [s for s in self.active_signals if not s.is_expired]
        
        # Mantener solo las últimas 100 posiciones para limitar memoria
        if len(self.recent_positions) > 100:
            self.recent_positions = self.recent_positions[-100:]