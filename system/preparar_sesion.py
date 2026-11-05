"""Selecciona el perfil PAPER y prepara su almacenamiento al arrancar."""
from __future__ import annotations

import logging
from pathlib import Path

from api.esquemas.modo_sesion import ModoSesion
from api.servicios import servicio_modo_sesion
from storage.perfil_datos import (
    PERFIL_LIMPIO,
    PERFIL_PERSISTENTE,
    configurar_perfil_datos,
    limpiar_perfil_paper,
    migrar_datos_legacy_a_persistente,
    hay_reset_persistente_pendiente,
    consumir_reset_persistente,
)

registro = logging.getLogger(__name__)


def decidir_modo_efectivo(modo_configurado: ModoSesion, es_modo_live: bool) -> ModoSesion:
    """Impide aplicar el modo limpia cuando el proceso opera en LIVE."""
    if modo_configurado is ModoSesion.LIMPIA and es_modo_live:
        registro.warning("Modo 'limpia' ignorado en LIVE: no se borran datos de trading")
        return ModoSesion.PERSISTENTE
    return modo_configurado


def preparar_sesion(es_modo_live: bool, raiz: Path = Path(".")) -> ModoSesion:
    """Migra el legado, selecciona el perfil y limpia solo el perfil limpia."""
    modo = decidir_modo_efectivo(servicio_modo_sesion.leer_modo_configurado(), es_modo_live)

    if not es_modo_live:
        migracion = migrar_datos_legacy_a_persistente(raiz)
        if any(migracion.values()):
            registro.info("Migración PAPER completada: %s", migracion)

        perfil = PERFIL_LIMPIO if modo is ModoSesion.LIMPIA else PERFIL_PERSISTENTE
        configurar_perfil_datos(perfil)

        if perfil == PERFIL_LIMPIO:
            borrados = limpiar_perfil_paper(PERFIL_LIMPIO, raiz)
            registro.info("Perfil PAPER limpia preparado: %s archivos eliminados", borrados)
        else:
            if hay_reset_persistente_pendiente(raiz):
                borrados = limpiar_perfil_paper(PERFIL_PERSISTENTE, raiz)
                consumir_reset_persistente(raiz)
                registro.warning("Reinicio PAPER persistente diferido completado: %s archivos eliminados", borrados)
            else:
                registro.info("Perfil PAPER persistente seleccionado; se conserva cartera e historial")
    else:
        # LIVE conserva su ruta de cartera y no utiliza el almacenamiento PAPER.
        configurar_perfil_datos(PERFIL_PERSISTENTE)
        registro.info("Modo LIVE: no se limpia ni se migra el estado de trading")

    servicio_modo_sesion.registro_modo_activo.fijar(modo)
    return modo
