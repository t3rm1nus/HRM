"""Esquemas del modo de sesión y del perfil de datos PAPER."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModoSesion(str, Enum):
    """PERSISTENTE conserva su historial; LIMPIA usa un perfil aislado."""

    PERSISTENTE = "persistente"
    LIMPIA = "limpia"


class SolicitudModoSesion(BaseModel):
    modo: ModoSesion


class SolicitudReinicioDatosPaper(BaseModel):
    """Confirmación explícita para reiniciar el perfil PAPER persistente."""

    confirmacion: str = Field(pattern="^CONFIRMAR$")


class EstadoModoSesion(BaseModel):
    modo_activo: Optional[ModoSesion]
    modo_configurado: ModoSesion
    reinicio_pendiente: bool
    modo_operativo: Literal["paper", "live", "unknown"] = "unknown"
    reinicio_datos_pendiente: bool = False
