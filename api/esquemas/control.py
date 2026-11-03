# Control - Comandos y control del sistema

from datetime import datetime
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel


class ComandoSistema(BaseModel):
    """Comando para controlar el sistema."""
    tipo: Literal["start", "stop", "restart", "pause", "resume", "shutdown"]
    parametros: Dict[str, Any] = {}
    id_solicitud: str
    timestamp: datetime = datetime.utcnow
    autor: str = "system"


class EstadoOperacion(BaseModel):
    """Estado de una operación en curso."""
    order_id: str
    simbolo: str
    lado: Literal["buy", "sell"]
    cantidad: float
    precio: float
    status: Literal["pending", "filled", "partial", "canceled", "rejected"]
    fecha_creacion: datetime = datetime.utcnow
    ultima_actualizacion: datetime = datetime.utcnow


class EventoControl(BaseModel):
    """Evento de control del sistema."""
    id: str
    tipo: Literal["order_status", "state_change", "risk_alert", "system"]
    mensaje: str
    datos: Dict[str, Any] = {}
    timestamp: datetime = datetime.utcnow
    origen: Literal["l1", "l2", "l3", "api", "sistema"] = "sistema"


class ResetSolicitud(BaseModel):
    """Solicitud para reiniciar componentes del sistema."""
    componentes: Literal["all", "config", "signals", "execution", "risk", "learning"] = "all"
    motivo: str
    preservar_estado: bool = False
    timestamp: datetime = datetime.utcnow


class CambioModo(BaseModel):
    """Cambio de modo del sistema (paper/live)."""
    nuevo_modo: Literal["paper", "live", "simulated", "testnet"]
    motivo: str
    confirmado: bool = False
    timestamp: datetime = datetime.utcnow