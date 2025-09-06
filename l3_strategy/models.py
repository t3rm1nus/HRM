"""
L3 Strategic Models
Modelos de datos para el nivel estratégico de toma de decisiones
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from .config import MarketRegime, RiskAppetite


# ===== MARKET DATA MODELS =====

@dataclass
class MacroIndicator:
    """Indicador macroeconómico"""
    name: str
    value: float
    timestamp: datetime
    source: str
    unit: str = ""
    change_1d: Optional[float] = None
    change_7d: Optional[float] = None
    change_30d: Optional[float] = None
    percentile_rank: Optional[float] = None  # percentil histórico
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OnChainMetric:
    """Métrica on-chain"""
    metric_name: str
    value: float
    timestamp: datetime
    asset: str  # BTC, ETH, etc.
    source: str  # glassnode, santiment
    normalized_value: Optional[float] = None  # valor normalizado 0-1
    z_score: Optional[float] = None  # z-score vs historical
    signal_strength: Optional[str] = None  # "bullish", "bearish", "neutral"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SentimentData:
    """Datos de sentimiento de mercado"""
    timestamp: datetime
    source: str  # twitter, reddit, news
    sentiment_score: float  # -1 to +1
    confidence: float  # 0 to 1
    volume: int  # número de menciones/posts
    topics: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketData:
    """Datos consolidados de mercado para L3"""
    timestamp: datetime
    
    # Price data
    prices: Dict[str, float] = field(default_factory=dict)  # {asset: price}
    returns_1d: Dict[str, float] = field(default_factory=dict)
    returns_7d: Dict[str, float] = field(default_factory=dict)
    returns_30d: Dict[str, float] = field(default_factory=dict)
    
    # Volatility data
    volatilities_30d: Dict[str, float] = field(default_factory=dict)
    volatilities_90d: Dict[str, float] = field(default_factory=dict)
    
    # Volume data
    volumes_24h: Dict[str, float] = field(default_factory=dict)
    volumes_7d_avg: Dict[str, float] = field(default_factory=dict)
    
    # Correlation matrix
    correlation_matrix: Optional[np.ndarray] = None
    correlation_assets: List[str] = field(default_factory=list)
    
    # Macro indicators
    macro_indicators: Dict[str, MacroIndicator] = field(default_factory=dict)
    
    # On-chain metrics
    onchain_metrics: Dict[str, List[OnChainMetric]] = field(default_factory=dict)
    
    # Sentiment data
    sentiment_data: List[SentimentData] = field(default_factory=list)
    sentiment_consolidated: Optional[float] = None  # -1 to +1
    
    def get_asset_price(self, asset: str) -> Optional[float]:
        """Obtiene precio de un activo"""
        return self.prices.get(asset)
    
    def get_asset_return(self, asset: str, period: str = "1d") -> Optional[float]:
        """Obtiene retorno de un activo para un período"""
        returns_dict = getattr(self, f"returns_{period}", {})
        return returns_dict.get(asset)
    
    def get_correlation(self, asset1: str, asset2: str) -> Optional[float]:
        """Obtiene correlación entre dos activos"""
        if self.correlation_matrix is None or not self.correlation_assets:
            return None
        
        try:
            idx1 = self.correlation_assets.index(asset1)
            idx2 = self.correlation_assets.index(asset2)
            return self.correlation_matrix[idx1, idx2]
        except (ValueError, IndexError):
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario serializable"""
        data = asdict(self)
        # Convert numpy array to list for JSON serialization
        if self.correlation_matrix is not None:
            data['correlation_matrix'] = self.correlation_matrix.tolist()
        return data


# ===== STRATEGIC ANALYSIS MODELS =====

@dataclass
class RegimeAnalysis:
    """Análisis de régimen de mercado"""
    timestamp: datetime
    detected_regime: MarketRegime
    confidence: float  # 0 to 1
    regime_probabilities: Dict[MarketRegime, float] = field(default_factory=dict)
    
    # Características del régimen
    trend_strength: float = 0.0  # -1 to +1
    volatility_level: str = "normal"  # "low", "normal", "high", "extreme"
    momentum: float = 0.0  # -1 to +1
    
    # Duración estimada
    regime_duration_days: Optional[int] = None
    regime_start_date: Optional[datetime] = None
    
    # Supporting evidence
    supporting_indicators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert enum keys to strings
        data['detected_regime'] = self.detected_regime.value
        data['regime_probabilities'] = {
            regime.value: prob for regime, prob in self.regime_probabilities.items()
        }
        return data


@dataclass
class RiskMetrics:
    """Métricas de riesgo estratégico"""
    timestamp: datetime
    
    # Value at Risk
    var_1d: float  # 1-day VaR at configured confidence level
    var_7d: float  # 7-day VaR
    var_30d: float  # 30-day VaR
    
    # Expected Shortfall (CVaR)
    cvar_1d: float
    cvar_7d: float
    cvar_30d: float
    
    # Portfolio volatility
    portfolio_volatility_annual: float
    volatility_forecast_30d: float
    
    # Drawdown metrics
    current_drawdown: float  # current unrealized drawdown
    max_drawdown_1y: float   # max drawdown in last year
    
    # Correlation risk
    avg_correlation: float           # average pairwise correlation
    max_correlation: float           # maximum pairwise correlation
    correlation_risk_level: str      # "low", "moderate", "high", "extreme"
    
    # Liquidity metrics
    portfolio_liquidity_score: float  # 0 to 1
    days_to_liquidate: float          # estimated days to fully liquidate
    
    # Stress test results
    stress_test_results: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssetAllocation:
    """Asignación de activos optimizada"""
    timestamp: datetime
    allocation: Dict[str, float]  # {asset: weight} donde weights suman 1.0
    
    # Optimization details
    optimization_method: str
    expected_return: float        # annual expected return
    expected_volatility: float    # annual expected volatility
    sharpe_ratio: float          # expected Sharpe ratio
    
    # Risk budgets
    risk_budgets: Dict[str, float] = field(default_factory=dict)  # {asset: risk_contribution}
    
    # Constraints satisfied
    constraints_satisfied: bool = True
    constraint_violations: List[str] = field(default_factory=list)
    
    # Rebalancing info
    rebalance_required: bool = False
    current_allocation: Dict[str, float] = field(default_factory=dict)
    rebalance_trades: Dict[str, float] = field(default_factory=dict)  # {asset: weight_change}
    
    def get_allocation(self, asset: str) -> float:
        """Obtiene la asignación para un activo específico"""
        return self.allocation.get(asset, 0.0)
    
    def get_risk_budget(self, asset: str) -> float:
        """Obtiene el budget de riesgo para un activo"""
        return self.risk_budgets.get(asset, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===== STRATEGIC DECISION MODELS =====

@dataclass
class StrategicGuidelines:
    """Directrices estratégicas para L2"""
    
    # Risk limits
    max_single_asset_exposure: float
    min_correlation_diversification: float
    volatility_target: float
    
    # Liquidity requirements
    min_daily_volume: float
    max_slippage: float
    
    # Position sizing
    base_position_size: float
    volatility_adjustment: float
    momentum_adjustment: float
    
    # Defensive measures
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketContext:
    """Contexto de mercado para L2"""
    
    # Correlation matrix
    correlation_matrix: Dict[str, Dict[str, float]]
    
    # Volatility forecasts
    volatility_forecast: Dict[str, float]
    
    # Sentiment
    sentiment_score: float  # -1 to +1
    
    # Macro indicators relevantes
    macro_indicators: Dict[str, Any]
    
    def get_correlation(self, asset1: str, asset2: str) -> Optional[float]:
        """Obtiene correlación entre dos activos"""
        return self.correlation_matrix.get(asset1, {}).get(asset2)
    
    def get_volatility_forecast(self, asset: str) -> Optional[float]:
        """Obtiene forecast de volatilidad para un activo"""
        return self.volatility_forecast.get(asset)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategicSignal:
    """Señal estratégica completa para L2"""
    strategy_id: str
    timestamp: datetime
    
    # Market regime
    market_regime: MarketRegime
    
    # Asset allocation
    asset_allocation: Dict[str, float]
    
    # Risk parameters
    risk_appetite: RiskAppetite
    target_exposure: float  # 0 to 1
    
    # Rebalancing
    rebalance_frequency: str  # "daily", "weekly", "monthly"
    
    # Strategic guidelines
    strategic_guidelines: StrategicGuidelines
    
    # Market context
    market_context: MarketContext
    
    # Validity
    valid_until: datetime
    confidence_level: float  # 0 to 1
    
    # Performance tracking
    signal_id: str = ""
    parent_signal_id: Optional[str] = None  # for signal updates
    
    def is_valid(self, current_time: Optional[datetime] = None) -> bool:
        """Verifica si la señal sigue siendo válida"""
        if current_time is None:
            current_time = datetime.utcnow()
        return current_time <= self.valid_until
    
    def get_asset_allocation(self, asset: str) -> float:
        """Obtiene la asignación para un activo específico"""
        return self.asset_allocation.get(asset, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario serializable"""
        data = asdict(self)
        # Convert enums to strings
        data['market_regime'] = self.market_regime.value
        data['risk_appetite'] = self.risk_appetite.value
        # Convert timestamps
        data['timestamp'] = self.timestamp.isoformat()
        data['valid_until'] = self.valid_until.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategicSignal':
        """Crea StrategicSignal desde diccionario"""
        # Convert string enums back
        data['market_regime'] = MarketRegime(data['market_regime'])
        data['risk_appetite'] = RiskAppetite(data['risk_appetite'])
        
        # Convert timestamp strings back
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['valid_until'] = datetime.fromisoformat(data['valid_until'])
        
        # Reconstruct nested objects
        if 'strategic_guidelines' in data and isinstance(data['strategic_guidelines'], dict):
            data['strategic_guidelines'] = StrategicGuidelines(**data['strategic_guidelines'])
        
        if 'market_context' in data and isinstance(data['market_context'], dict):
            data['market_context'] = MarketContext(**data['market_context'])
        
        return cls(**data)


# ===== PERFORMANCE TRACKING MODELS =====

@dataclass
@dataclass
class StrategicPerformance:
    """Métricas de performance estratégica"""
    
    # Fechas del período
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    
    # Returns
    total_return: float
    annualized_return: float
    excess_return: float  # vs benchmark
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Decision accuracy
    regime_accuracy: float  # accuracy of regime detection
    allocation_efficiency: float  # how good were allocation decisions
    
    # Regime-specific performance (default vacío)
    regime_performance: Dict[MarketRegime, Dict[str, float]] = field(default_factory=dict)
    
    # Attribution analysis
    asset_contributions: Dict[str, float] = field(default_factory=dict)
    alpha_generation: float = 0.0
    beta_exposure: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convertir keys de enum a string
        if self.regime_performance:
            data['regime_performance'] = {
                regime.value: metrics for regime, metrics in self.regime_performance.items()
            }
        return data


# ===== UTILITY FUNCTIONS =====

def create_default_strategic_signal(
    market_regime: MarketRegime = MarketRegime.RANGING_MARKET,
    risk_appetite: RiskAppetite = RiskAppetite.MODERATE
) -> StrategicSignal:
    """Crea una señal estratégica por defecto"""
    
    now = datetime.utcnow()
    
    # Default allocation (equal weight BTC/ETH with some cash)
    default_allocation = {
        "BTC": 0.40,
        "ETH": 0.30,
        "USDT": 0.30
    }
    
    # Default guidelines
    guidelines = StrategicGuidelines(
        max_single_asset_exposure=0.70,
        min_correlation_diversification=0.30,
        volatility_target=0.25,
        min_daily_volume=1000000.0,
        max_slippage=0.002,
        base_position_size=0.1,
        volatility_adjustment=1.0,
        momentum_adjustment=1.0
    )
    
    # Default market context
    context = MarketContext(
        correlation_matrix={
            "BTC": {"ETH": 0.75, "USDT": 0.0},
            "ETH": {"BTC": 0.75, "USDT": 0.0},
            "USDT": {"BTC": 0.0, "ETH": 0.0}
        },
        volatility_forecast={
            "BTC": 0.50,
            "ETH": 0.60
        },
        sentiment_score=0.0,
        macro_indicators={}
    )
    
    return StrategicSignal(
        strategy_id=f"default_{now.strftime('%Y%m%d_%H%M%S')}",
        timestamp=now,
        market_regime=market_regime,
        asset_allocation=default_allocation,
        risk_appetite=risk_appetite,
        target_exposure=0.70,
        rebalance_frequency="weekly",
        strategic_guidelines=guidelines,
        market_context=context,
        valid_until=now + timedelta(hours=24),
        confidence_level=0.50,
        signal_id=f"sig_{now.strftime('%Y%m%d_%H%M%S')}"
    )


def validate_allocation(allocation: Dict[str, float], tolerance: float = 1e-6) -> bool:
    """Valida que la asignación sume aproximadamente 1.0"""
    total = sum(allocation.values())
    return abs(total - 1.0) <= tolerance


def normalize_allocation(allocation: Dict[str, float]) -> Dict[str, float]:
    """Normaliza asignación para que sume exactamente 1.0"""
    total = sum(allocation.values())
    if total == 0:
        return allocation
    return {asset: weight / total for asset, weight in allocation.items()}