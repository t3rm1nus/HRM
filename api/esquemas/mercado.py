# Mercado - Datos de mercado y régimen

from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel


class SeñalMercado(BaseModel):
    """Señal de mercado generada por L3."""
    simbolo: str
    direccion: Literal["buy", "sell", "hold"]
    fuerza: float = 0.0  # 0.0 a 1.0
    confianza: float = 0.0  # 0.0 a 1.0
    regimen: Literal["bull_market", "bear_market", "ranging_market", "volatile_market", "transition", "neutral"] = "neutral"
    riesgo_apetito: Literal["conservative", "moderate", "aggressive", "maximum"] = "moderate"
    asignacion: Dict[str, float] = {"btc": 0.0, "eth": 0.0, "usdt": 1.0}
    timestamp: datetime = datetime.utcnow


class DatosMercado(BaseModel):
    """Datos actuales de mercado."""
    simbolo: str
    precio_actual: float
    cambio_24h: float
    volumen_24h: float
    capitalizacion: Optional[float] = None
    last_update: datetime = datetime.utcnow


class ConjuntoSeñales(BaseModel):
    """Conjunto de señales de múltiples activos."""
    señales: List[SeñalMercado]
    generado_en: datetime = datetime.utcnow
    count: int