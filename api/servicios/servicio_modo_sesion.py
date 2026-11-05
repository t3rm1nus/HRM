"""Modo de sesión: se guarda en disco y se aplica en el siguiente arranque.

Este servicio solo lee y escribe el modo. Quien decide qué hacer con el estado
al arrancar es `system.preparar_sesion`.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from api.esquemas.modo_sesion import EstadoModoSesion, ModoSesion

registro = logging.getLogger(__name__)

VARIABLE_ENTORNO_RUTA = "HRM_RUTA_MODO_SESION"
RUTA_POR_DEFECTO = Path("persistent_state/modo_sesion.json")
MODO_POR_DEFECTO = ModoSesion.PERSISTENTE
CLAVE_MODO = "modo"


def obtener_ruta_archivo() -> Path:
    return Path(os.getenv(VARIABLE_ENTORNO_RUTA, str(RUTA_POR_DEFECTO)))


def convertir_a_modo(valor: object) -> ModoSesion:
    try:
        return ModoSesion(valor)
    except ValueError:
        registro.warning("Modo de sesión no válido (%r); se usa %s", valor, MODO_POR_DEFECTO.value)
        return MODO_POR_DEFECTO


def leer_contenido(ruta: Path) -> dict:
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        registro.warning("No se pudo leer %s; se usa el modo por defecto", ruta)
        return {}


def leer_modo_configurado() -> ModoSesion:
    ruta = obtener_ruta_archivo()
    if not ruta.exists():
        return MODO_POR_DEFECTO
    return convertir_a_modo(leer_contenido(ruta).get(CLAVE_MODO))


def guardar_modo_configurado(modo: ModoSesion) -> None:
    ruta = obtener_ruta_archivo()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps({CLAVE_MODO: modo.value}), encoding="utf-8")
    temporal.replace(ruta)


class RegistroModoActivo:
    """Recuerda con qué modo arrancó el proceso en marcha."""

    def __init__(self) -> None:
        self._modo: Optional[ModoSesion] = None

    def fijar(self, modo: ModoSesion) -> None:
        self._modo = modo

    def obtener(self) -> Optional[ModoSesion]:
        return self._modo


registro_modo_activo = RegistroModoActivo()


def obtener_estado_modo() -> EstadoModoSesion:
    activo = registro_modo_activo.obtener()
    configurado = leer_modo_configurado()
    return EstadoModoSesion(
        modo_activo=activo,
        modo_configurado=configurado,
        reinicio_pendiente=activo is not None and activo != configurado,
    )