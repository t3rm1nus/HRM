# Estrategia - Configuración y parámetros de estrategia

from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel


class ConfigOptimization(BaseModel):
    """Configuración de optimización de estrategia."""
    method: Literal["black_litterman", "markowitz", "risk_parity"] = "black_litterman"
    window: int = 252  # días
    frequency: Literal["daily", "weekly", "monthly"] = "weekly"


class ModelConfig(BaseModel):
    """Configuración de modelo ML."""
    regime_model_path: str = "models/L3/regime_detector.pkl"
    regime_model_type: Literal["random_forest", "lstm", "garch"] = "random_forest"
    regime_confidence_threshold: float = 0.45
    sentiment_model_type: Literal["bert", "vader"] = "bert"
    sentiment_weight: float = 0.25


class RiskConfig(BaseModel):
    """Configuración de riesgo estratégico."""
    var_confidence_level: float = 0.95
    cvar_confidence_level: float = 0.95
    max_single_asset_exposure: float = 0.75
    max_portfolio_volatility: float = 0.35
    max_drawdown_trigger: float = 0.15
    volatility_spike_trigger: float = 2.0


class ExecutionConfig(BaseModel):
    """Configuración de ejecución estratégica."""
    output_format: Literal["json", "csv", "binance"] = "json"
    output_precision: int = 4
    cycle_interval_seconds: int = 3  # Ciclo principal
    l2_update_frequency: Literal["10min", "1min", "5min"] = "10min"


class EstrategiaConfig(BaseModel):
    """Configuración completa de la estrategia HRM."""
    nombre: str = "HRM Strategi"
    modo: Literal["paper", "live", "simulated"] = "paper"
    optimization: ConfigOptimization = ConfigOptimization()
    model: ModelConfig = ModelConfig()
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()
    created_at: datetime = datetime.utcnow
    updated_at: datetime = datetime.utcnow


class MetricasEstrategia(BaseModel):
    """Métricas de rendimiento de la estrategia."""
    total_ciclos: int = 0
    ciclos_ganadores: int = 0
    ciclos_perdedores: int = 0
    win_rate: float = 0.0
    pnl_total: float = 0.0
    pnl_pct_total: float = 0.0
    drawdown_maximal: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    ultimo_ciclo: Optional[datetime] = None