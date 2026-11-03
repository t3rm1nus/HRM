# Eventos - Eventos del sistema

from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel


class EventoSistema(BaseModel):
    """Evento genérico del sistema HRM."""
    id: str
    tipo: str
    titulo: str
    mensaje: str
    datos: Dict[str, Any] = {}
    origen: Literal["l1", "l2", "l3", "api", "sistema", "external"] = "sistema"
    severity: Literal["info", "warning", "error", "critical"] = "info"
    timestamp: datetime = datetime.utcnow
    metadata: Dict[str, Any] = {}


class TipoEvento:
    """Tipos de evento del sistema."""
    SIGNAL = "signal"
    ORDER = "order"
    TRADE = "trade"
    RISK = "risk"
    SYSTEM = "system"
    CYCLE = "cycle"
    ERROR = "error"
    WARNING = "warning"
    STATUS = "status"


class SeñalGeneradaEvento(EventoSistema):
    """Evento cuando se genera una señal de trading."""
    pass


class OrderEvento(EventoSistema):
    """Evento cuando ocurre algo con una orden."""
    pass


class TradeEvento(EventoSistema):
    """Evento cuando se ejecuta un trade."""
    pass


class CycleEvento(EventoSistema):
    """Evento de ciclo de trading."""
    tipo: Literal["cycle"] = "cycle"
    cycle_id: str = ""
    pnl: float = 0.0
    regime: str = "unknown"


class ErrorEvento(EventoSistema):
    """Evento de error."""
    tipo: Literal["error"] = "error"
    critical: bool = False


class EventoStats(BaseModel):
    """Estadísticas de eventos."""
    total_eventos: int = 0
    por_tipo: Dict[str, int] = {}
    por_severidad: Dict[str, int] = {}
    ultimo_evento: Optional[datetime] = None
    primer_evento: Optional[datetime] = None