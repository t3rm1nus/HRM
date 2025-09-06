# storage/__init__.py
from core.logging import logger

logger.info("storage")

def guardar_estado_csv(state: dict):
    logger.debug("[STORAGE] Guardando estado en CSV (placeholder)")

def guardar_estado_sqlite(state: dict):
    logger.debug("[STORAGE] Guardando estado en SQLite (placeholder)")