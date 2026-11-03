# Cartera - Posiciones y valuación de la cartera

from datetime import datetime
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel


class Posicion(BaseModel):
    """Posición individual en la cartera."""
    simbolo: str
    side: Literal["long", "short"]
    cantidad: float
    precio_entrada: float
    precio_actual: float
    pnl: float
    pnl_pct: float
    comision: float
    timestamp: datetime = datetime.utcnow


class ResumenCartera(BaseModel):
    """Resumen de la cartera."""
    valor_total: float
    valor_peak: float
    pnl_total: float
    pnl_pct: float
    patrimonio_liquido: float
    posiciones_abiertas: int
    efectivo_disponible: float
    fecha_valuacion: datetime = datetime.utcnow


class ExposicionPorActivo(BaseModel):
    """Exposición normalizada por activo (0.0 a 1.0)."""
    btc: float = 0.0
    eth: float = 0.0
    usdt: float = 0.0
    total: float = 0.0