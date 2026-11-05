"""Aplica el modo de sesión al arrancar, antes de crear cartera o historial."""
from __future__ import annotations

import logging
from pathlib import Path

from api.esquemas.modo_sesion import ModoSesion
from api.servicios import servicio_modo_sesion

registro = logging.getLogger(__name__)

ARCHIVO_ESTADO_CARTERA = Path("persistent_state/portfolio_state_live.json")
DIRECTORIO_OPERACIONES = Path("data/paper_trades")
PATRONES_OPERACIONES = ("*.json", "*.csv")


def borrar_archivo(ruta: Path) -> bool:
    if not ruta.is_file():
        return False
    ruta.unlink()
    return True


def borrar_estado_cartera(raiz: Path) -> int:
    return int(borrar_archivo(raiz / ARCHIVO_ESTADO_CARTERA))


def listar_archivos_operaciones(raiz: Path) -> list[Path]:
    directorio = raiz / DIRECTORIO_OPERACIONES
    return [archivo for patron in PATRONES_OPERACIONES for archivo in directorio.glob(patron)]


def borrar_historial_operaciones(raiz: Path) -> int:
    return sum(borrar_archivo(archivo) for archivo in listar_archivos_operaciones(raiz))


def vaciar_estado_de_sesion(raiz: Path) -> int:
    return borrar_estado_cartera(raiz) + borrar_historial_operaciones(raiz)


def decidir_modo_efectivo(modo_configurado: ModoSesion, es_modo_live: bool) -> ModoSesion:
    if modo_configurado is ModoSesion.LIMPIA and es_modo_live:
        registro.warning("Modo 'limpia' ignorado en live: nunca se borra el estado con dinero real")
        return ModoSesion.PERSISTENTE
    return modo_configurado


def preparar_sesion(es_modo_live: bool, raiz: Path = Path(".")) -> ModoSesion:
    modo = decidir_modo_efectivo(servicio_modo_sesion.leer_modo_configurado(), es_modo_live)
    if modo is ModoSesion.LIMPIA:
        borrados = vaciar_estado_de_sesion(raiz)
        registro.info("Sesión limpia: %s archivos de estado borrados", borrados)
    else:
        registro.info("Sesión persistente: se conserva cartera e historial")
    servicio_modo_sesion.registro_modo_activo.fijar(modo)
    return modo