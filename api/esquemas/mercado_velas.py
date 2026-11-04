"""Esquemas públicos del circuito de mercado (solo lectura)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

EstadoFrescura = Literal["sin_datos", "actual", "obsoleto"]


class Vela(BaseModel):
    """Una vela OHLCV. `instante` es ISO 8601 en UTC."""

    instante: str
    apertura: float
    maximo: float
    minimo: float
    cierre: float
    volumen: float


class EstadoMercado(BaseModel):
    """Estado de la fuente de datos de mercado."""

    disponible: bool = Field(description="False si el gestor aún no se ha registrado")
    frescura: EstadoFrescura
    fuente: Optional[str] = None
    edad_segundos: Optional[float] = None
    validez_segundos: Optional[float] = None
    simbolos: list[str] = Field(default_factory=list)
    modo: Literal["paper"] = "paper"


class ListaSimbolos(BaseModel):
    simbolos: list[str]


class LoteVelas(BaseModel):
    simbolo: str
    cantidad: int
    frescura: EstadoFrescura
    velas: list[Vela]


class ActualizacionMercado(BaseModel):
    """Mensaje ligero para el WebSocket: solo la última vela por símbolo."""

    frescura: EstadoFrescura
    ultimas_velas: dict[str, Vela]
