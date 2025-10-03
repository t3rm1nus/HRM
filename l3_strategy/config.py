# l3_strategy/config.py - Configuración para el módulo L2_tactic (adaptado para multiasset: BTC y ETH)

import os
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class MarketRegime(Enum):
    """Regímenes de mercado identificables"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    RANGING_MARKET = "ranging_market"  
    VOLATILE_MARKET = "volatile_market"
    TRANSITION = "transition"


class RiskAppetite(Enum):
    """Niveles de apetito de riesgo"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class ModelConfig:
    """Configuración de modelos ML para L3"""
    # Regime Detection Models
    regime_model_path: str = "models/L3/regime_detector.pkl"
    regime_model_type: str = "random_forest"
    regime_retrain_frequency: int = 168  # hours (1 week)
    regime_confidence_threshold: float = 0.45
    
    # Sentiment Analysis
    sentiment_model_path: str = "models/L3/sentiment_analyzer.pkl" 
    sentiment_model_type: str = "bert"
    sentiment_sources: List[str] = field(default_factory=lambda: ["twitter", "reddit", "news"])
    sentiment_weight: float = 0.25
    
    # Portfolio Optimization
    optimization_method: str = "black_litterman"  # "markowitz", "risk_parity"
    optimization_window: int = 252  # trading days (1 year)
    rebalance_frequency: str = "weekly"  # "daily", "monthly"
    
    # Volatility Models
    volatility_model: str = "garch"  # "ewma", "historical"
    volatility_window: int = 30  # days
    
    # LSTM for volatility forecasting
    lstm_model_path: str = "models/L3/volatility_lstm.h5"
    lstm_sequence_length: int = 60
    lstm_forecast_horizon: int = 30


@dataclass
class RiskConfig:
    """Configuración de gestión de riesgo estratégico"""
    # VaR Configuration
    var_confidence_level: float = 0.95
    var_holding_period: int = 1  # days
    var_method: str = "historical"  # "parametric", "monte_carlo"
    
    # Expected Shortfall (CVaR)
    cvar_confidence_level: float = 0.95
    
    # Portfolio Limits
    max_single_asset_exposure: float = 0.75
    min_correlation_diversification: float = 0.30
    max_portfolio_volatility: float = 0.35
    min_liquidity_requirement: float = 1000000  # USD daily volume
    max_slippage_tolerance: float = 0.002  # 20 bps
    
    # Risk Triggers
    max_drawdown_trigger: float = 0.15  # 15%
    volatility_spike_trigger: float = 2.0  # 2x normal volatility
    correlation_breakdown_trigger: float = 0.90  # when all correlations > 0.9
    
    # Stress Testing
    stress_scenarios: List[str] = field(default_factory=lambda: [
        "market_crash_2008", "covid_crash_2020", "crypto_winter_2022",
        "flash_crash", "liquidity_crisis", "regulatory_shock"
    ])


@dataclass 
class DataConfig:
    """Configuración de fuentes de datos"""
    # Market Data
    market_data_providers: List[str] = field(default_factory=lambda: ["binance", "yahoo_finance"])
    market_data_frequency: str = "1h"  # para análisis estratégico
    market_data_history: int = 365 * 2  # days (2 years)
    
    # Macro Data
    macro_data_providers: List[str] = field(default_factory=lambda: ["fred", "oecd", "tradingeconomics"])
    macro_indicators: List[str] = field(default_factory=lambda: [
        "GDP_US", "CPI_US", "UNEMPLOYMENT_US", "FED_FUNDS_RATE",
        "DXY", "VIX", "GOLD", "OIL_WTI", "YIELD_10Y"
    ])
    macro_update_frequency: str = "daily"
    
    # On-chain Data  
    onchain_providers: List[str] = field(default_factory=lambda: ["glassnode", "santiment"])
    onchain_metrics: List[str] = field(default_factory=lambda: [
        "active_addresses", "transaction_volume", "hash_rate",
        "exchange_inflows", "exchange_outflows", "whale_activity",
        "mvrv_ratio", "puell_multiple", "fear_greed_index"
    ])
    
    # Alternative Data
    sentiment_data_sources: List[str] = field(default_factory=lambda: [
        "twitter_api", "reddit_api", "news_api", "google_trends"
    ])


@dataclass
class OptimizationConfig:
    """Configuración de optimización de cartera"""
    # Asset Universe
    supported_assets: List[str] = field(default_factory=lambda: [
        "BTC", "ETH", "BNB", "ADA", "SOL", "DOT", "LINK", "AVAX", "MATIC", "UNI"
    ])
    base_currency: str = "USDT"
    
    # Optimization Parameters
    optimization_objective: str = "max_sharpe"  # "min_variance", "max_return"
    risk_free_rate: float = 0.02  # annual
    expected_return_method: str = "historical"  # "capm", "black_litterman"
    covariance_method: str = "sample"  # "ledoit_wolf", "shrunk"
    
    # Constraints
    min_weight: float = 0.0
    max_weight: float = 0.75
    target_volatility: float = 0.25  # annual
    leverage_limit: float = 1.0  # no leverage
    
    # Rebalancing
    rebalance_threshold: float = 0.05  # 5% deviation triggers rebalancing
    transaction_cost: float = 0.001  # 10 bps
    
    # Black-Litterman specific
    bl_tau: float = 0.05  # uncertainty parameter
    bl_confidence_levels: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 0.80, "ETH": 0.75, "BNB": 0.60
    })


@dataclass
class ExecutionConfig:
    """Configuración de ejecución estratégica"""
    # Timing
    analysis_frequency: str = "hourly"  # frequency of strategic analysis
    decision_frequency: str = "daily"   # frequency of strategic decisions
    emergency_review_triggers: List[str] = field(default_factory=lambda: [
        "market_crash", "volatility_spike", "correlation_breakdown"
    ])
    
    # Operating Modes
    operating_mode: str = "automatic"  # "semi_automatic", "simulation"
    confidence_threshold: float = 0.45  # minimum confidence for decisions
    
    # Output Configuration
    output_format: str = "json"
    output_precision: int = 4
    valid_until_buffer: int = 24  # hours validity buffer
    
    # Communication with L2
    l2_update_frequency: str = "10min"
    l2_override_capability: bool = True  # L3 can override L2 decisions


# Environment Configuration
ENVIRONMENT = os.getenv("L3_ENVIRONMENT", "development")

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "data/logs/l3_strategic.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "detailed"
        },
        "console": {
            "level": "INFO", 
            "class": "logging.StreamHandler",
            "formatter": "detailed"
        }
    },
    "loggers": {
        "l3_strategic": {
            "level": "INFO",
            "handlers": ["file", "console"],
            "propagate": False
        }
    }
}

# Default Configuration Instance
DEFAULT_CONFIG = {
    "model": ModelConfig(),
    "risk": RiskConfig(),
    "data": DataConfig(),
    "optimization": OptimizationConfig(),
    "execution": ExecutionConfig()
}


def get_config(section: str = None) -> Any:
    """
    Obtiene configuración por sección
    
    Args:
        section: Sección de configuración ('model', 'risk', 'data', 'optimization', 'execution')
        
    Returns:
        Configuración solicitada o configuración completa si section=None
    """
    if section is None:
        return DEFAULT_CONFIG
    
    return DEFAULT_CONFIG.get(section)


def update_config(section: str, updates: Dict[str, Any]) -> None:
    """
    Actualiza configuración de una sección específica
    
    Args:
        section: Sección a actualizar
        updates: Diccionario con valores a actualizar
    """
    config_obj = DEFAULT_CONFIG.get(section)
    if config_obj:
        for key, value in updates.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)


# API Keys y configuración externa
API_KEYS = {
    "fred_api_key": os.getenv("FRED_API_KEY"),
    "twitter_bearer_token": os.getenv("TWITTER_BEARER_TOKEN"),
    "news_api_key": os.getenv("NEWS_API_KEY"),
    "glassnode_api_key": os.getenv("GLASSNODE_API_KEY"),
    "santiment_api_key": os.getenv("SANTIMENT_API_KEY")
}

# Paths importantes
PATHS = {
    "models": "models/L3/",
    "data": "data/L3/",
    "logs": "data/logs/",
    "cache": "data/cache/L3/",
    "configs": "configs/L3/"
}

# Crear directorios si no existen
import os
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)
