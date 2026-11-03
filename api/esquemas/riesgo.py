# Riesgo - Métricas y límites de riesgo

from datetime import datetime
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel


class MetricasRisk(BaseModel):
    """Métricas de riesgo en tiempo real."""
    portfolio_heat: float = 0.0  # 0.0 a 1.0
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    cvar_95: float = 0.0  # Conditional Value at Risk 95%
    max_drawdown: float = 0.0  # Drawdown máximo observado
    drawdown_actual: float = 0.0  # Drawdown actual
    exposure_btc: float = 0.0  # Exposición en BTC (0-1)
    exposure_eth: float = 0.0  # Exposición en ETH (0-1)
    exposure_total: float = 0.0  # Exposición total (0-1)
    liquidation_price: Optional[float] = None
    margin_usage: float = 0.0  # Margen utilizado vs disponible
    last_update: datetime = datetime.utcnow


class LimitRiesgo(BaseModel):
    """Límites de riesgo configurables."""
    max_exposición_portafolio: float = 0.40  # 40% máximo
    max_exposición_por_activo: float = 0.75  # 75% por activo individual
    max_peso_posición_usdt: float = 1200.0  # Max $1200 por posición
    min_reserva_usdt: float = 500.0  # Mínimo $500 en USDT
    max_porcentaje_drawdown: float = 0.12  # 12% drawdown máximo
    max_operaciones_diarias: int = 100  # Límite diario de operaciones


class AlertaRiesgo(BaseModel):
    """Alerta de riesgo generada."""
    id: str
    tipo: Literal["exposure", "drawdown", "margin", "liquidity"]  # Tipo de alerta
    severidad: Literal["low", "medium", "high", "critical"]  # Nivel de severidad
    mensaje: str
    simbolo: Optional[str] = None
    valor_actual: float
    valor_límite: float
    timestamp: datetime = datetime.utcnow
    resuelto: bool = False


class EvaluacionPosicion(BaseModel):
    """Evaluación de riesgo para una posición específica."""
    simbolo: str
    lado: Literal["long", "short"]
    cantidad: float
    precio_entrada: float
    precio_actual: float
    pnl_potencial: float
    pnl_pct_potencial: float
    riesgo_per_trade: float  # Riesgo por trade (porcentaje del capital)
    recomendado: bool  # Si la posición es recomendada según límites
    observaciones: Optional[str] = None