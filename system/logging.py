"""
Configuración de Logging del Sistema HRM

Configura el sistema de logging para toda la aplicación.
"""

import logging
from logging import handlers
import os
from datetime import datetime

# Configuración de logging
LOG_LEVEL = logging.INFO
LOG_FILE = "logs/system.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging():
    """Configura el sistema de logging para toda la aplicación"""
    # Crear directorio de logs si no existe
    os.makedirs("logs", exist_ok=True)

    # Configurar logging
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )

    # Configurar logging a archivo
    file_handler = handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Agregar handler al logger raíz
    logging.getLogger().addHandler(file_handler)

    logging.info("Sistema de logging configurado correctamente")