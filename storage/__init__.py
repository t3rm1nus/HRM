# storage/__init__.py
import logging

logger = logging.getLogger("storage")

def guardar_estado_csv(state: dict):
    logger.debug("[STORAGE] Guardando estado en CSV (placeholder)")

def guardar_estado_sqlite(state: dict):
    logger.debug("[STORAGE] Guardando estado en SQLite (placeholder)")