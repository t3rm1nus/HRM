"""
Inicialización del Sistema HRM

Configura el sistema y expone las funcionalidades principales.
"""

from .logging import setup_logging
from .config import system_config

# Inicializar logging
setup_logging()

# Exponer configuración
__all__ = ['system_config']