# Operaciones - Señales y trades ejecutados

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel


class TacticalSignalCreate(BaseModel):
    """Schema para crear una señal táctica L2."""
    simbolo: str
    fuerza: float  # 0.0 a 1.0
    confianza: float  # 0.0 a 1.0
    lado: Literal["buy", "sell", "hold"] = "buy"
    cantidad: Optional[float] = None
    señal_tipo: Optional[str] = None
    fuente: Literal["ai", "technical", "pattern", "composite"] = "unknown"
    features: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    intencion: Literal["ENTRY", "EXIT"] = "ENTRY"
    timestamp: datetime = datetime.utcnow


class OrderRecord(BaseModel):
    """Registro de una orden ejecutada (desde PaperTradeLogger)."""
    order_id: str
    simbolo: str
    lado: Literal["buy", "sell", "hold"]
    cantidad: float
    precio_llenado: float
    comision: float
    status: Literal["filled", "partial", "canceled", "rejected"]
    timestamp: datetime
    estrategia: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TradeRecord(BaseModel):
    """Registro de trade completado."""
    trade_id: str
    order_id: str
    simbolo: str
    lado: Literal["buy", "sell", "hold"]
    precio_entrada: float
    precio_salida: Optional[float] = None
    cantidad: float
    pnl: float
    pnl_pct: float
    comision: float
    duracion: timedelta
    timestamp_entrada: datetime
    timestamp_salida: Optional[datetime] = None
    estrategia: Optional[str] = None
    exito: bool = True


class LoteOperaciones(BaseModel):
    """Lote de operaciones para loteado de respuestas."""
    total: int
    operaciones: List[TradeRecord]
    has_more: bool = False
    pagina: int = 1
    pagina_total: int = 1