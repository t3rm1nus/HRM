# Configuración - Configuraciones del sistema y parámetros

from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ConfiguracionGeneral(BaseModel):
    """Configuración general del sistema."""
    modo_operacion: Literal["paper", "live", "simulated", "testnet"] = "paper"
    capital_inicial: float = 500.0
    apalancamiento_maximo: float = 1.0  # Sin apalancamiento por defecto
    modo: str = "simulated"
    timezone: str = "UTC"


class ConfigTrading(BaseModel):
    """Configuración de trading específica."""
    symbols: List[str] = ["BTCUSDT", "ETHUSDT"]
    timeframes: List[str] = ["1m", "5m", "1h"]
    cycle_duration_seconds: int = 10
    max_position_size_usdt: float = 1200.0
    min_usdt_reserve: float = 500.0
    risk_per_trade_percent: float = 0.02
    commission_rate: float = 0.0005
    slippage_bps: float = 0.2


class ConfigRisk(BaseModel):
    """Configuración de riesgo."""
    max_portfolio_exposure_btc: float = 0.40
    max_portfolio_exposure_eth: float = 0.40
    max_drawdown_limit: float = 0.12
    stop_loss_default_percent: float = 0.03
    take_profit_default_percent: float = 0.05


class ConfigAPI(BaseModel):
    """Configuración de la API itself."""
    title: str = "HRM Trading API"
    version: str = "1.0.0"
    description: str = "HRM Trading System API"
    openapi_url: str = "/openapi.json"
    docs_url: str = "/docs"


class ConfigCarga(BaseModel):
    """Configuración de carga inicial."""
    initial_state: Dict[str, Any] = {}
    reset_singletons: bool = True
    auto_learning: Literal["fix", "adaptive", "production", "disabled"] = "fix"
    modo: Literal["paper", "live", "simulated"] = "paper"


class ConfigResponse(BaseModel):
    """Respuesta completa de configuración."""
    general: ConfiguracionGeneral = ConfiguracionGeneral()
    trading: ConfigTrading = ConfigTrading()
    risk: ConfigRisk = ConfigRisk()
    api: ConfigAPI = ConfigAPI()
    loaded_at: datetime = datetime.utcnow