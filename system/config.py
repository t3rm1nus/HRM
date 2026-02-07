"""
Configuración del Sistema HRM

Configuración global del sistema HRM.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SystemConfig:
    """Configuración global del sistema HRM"""
    system_name: str = "HRM"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    temp_dir: str = "temp"
    paper_trades_dir: str = "paper_trades"
    persistent_state_dir: str = "persistent_state"
    logs_dir: str = "logs"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

    def __post_init__(self):
        """Inicializa directorios si no existen"""
        self._create_directories()

    def _create_directories(self):
        """Crea directorios necesarios"""
        directories = [
            self.temp_dir,
            self.paper_trades_dir,
            self.persistent_state_dir,
            self.logs_dir
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directorio creado: {directory}")

# Configuración global
system_config = SystemConfig()