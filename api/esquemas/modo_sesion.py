"""Esquemas del modo de sesión (qué hace HRM con su estado al arrancar)."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ModoSesion(str, Enum):
    """PERSISTENTE conserva cartera e historial; LIMPIA arranca desde cero."""

    PERSISTENTE = "persistente"
    LIMPIA = "limpia"


class SolicitudModoSesion(BaseModel):
    modo: ModoSesion


class EstadoModoSesion(BaseModel):
    modo_activo: Optional[ModoSesion]
    modo_configurado: ModoSesion
    reinicio_pendiente: bool