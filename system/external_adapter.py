"""
Adaptador Externo del Sistema HRM

Proporciona interfaz para conexiones externas.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ExternalAdapter:
    """Adaptador para conexiones externas del sistema HRM"""

    def __init__(self):
        """Inicializa el adaptador externo"""
        self.logger = logger
        self.connected = False
        self.logger.info("ExternalAdapter inicializado")

    def connect(self):
        """Establece conexión con sistemas externos"""
        self.logger.info("Estableciendo conexión externa")
        self.connected = True
        self.logger.info("Conexión externa establecida")

    def disconnect(self):
        """Cierra conexión con sistemas externos"""
        self.logger.info("Cerrando conexión externa")
        self.connected = False
        self.logger.info("Conexión externa cerrada")

    def is_connected(self) -> bool:
        """Verifica si hay conexión activa

        Returns:
            bool: True si hay conexión activa, False si no
        """
        return self.connected