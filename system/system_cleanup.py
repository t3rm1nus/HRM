"""
Sistema de Limpieza del Sistema HRM

Proporciona utilidades para limpiar archivos temporales, paper trades y estado persistente.
"""

import os
import shutil
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SystemCleanup:
    """Utilidad para limpiar archivos temporales y de estado del sistema HRM"""

    def __init__(self):
        """Inicializa el limpiador del sistema con rutas por defecto"""
        self.logger = logger
        self.temp_dir = "temp"
        self.paper_trades_dir = "paper_trades"
        self.persistent_state_dir = "persistent_state"
    def perform_full_cleanup(self) -> Dict[str, Any]:
        """Realiza limpieza completa del sistema

        Returns:
            Dict: Resultado de la operación de limpieza
        """
        self.logger.info("Iniciando limpieza completa del sistema")

        result = {
            "success": True,
            "cleaned_files": [],
            "errors": [],
            "duration_ms": 0
        }

        start_time = datetime.now()

        try:
            # Limpiar archivos temporales
            temp_files = self.cleanup_temp_files()
            result["cleaned_files"].extend(temp_files)

            # Limpiar paper trades
            paper_trades = self.cleanup_paper_trades()
            result["cleaned_files"].extend(paper_trades)

            # Limpiar estado persistente
            persistent_files = self.cleanup_persistent_state()
            result["cleaned_files"].extend(persistent_files)

            duration = (datetime.now() - start_time).total_seconds() * 1000
            result["duration_ms"] = duration

            self.logger.info(
                f"Limpieza completada: {len(result['cleaned_files'])} archivos limpiados "
                f"en {duration:.2f}ms"
            )

        except Exception as e:
            self.logger.error(f"Error en limpieza completa: {e}")
            result["success"] = False
            result["errors"].append(str(e))

        return result

    def cleanup_temp_files(self) -> List[str]:
        """Limpia archivos temporales del sistema

        Returns:
            List: Archivos eliminados
        """
        self.logger.info("Limpiando archivos temporales")
        cleaned_files = []

        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        cleaned_files.append(file_path)
                        self.logger.debug(f"Eliminado archivo temporal: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"No se pudo eliminar {file_path}: {e}")

            # Limpiar directorios vacíos
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    try:
                        os.rmdir(dir_path)
                        self.logger.debug(f"Directorio vacío eliminado: {dir_path}")
                    except Exception as e:
                        self.logger.debug(f"No se pudo eliminar directorio {dir_path}: {e}")

        return cleaned_files

    def cleanup_paper_trades(self) -> List[str]:
        """Limpia paper trades anteriores del sistema

        Returns:
            List: Archivos de paper trades eliminados
        """
        self.logger.info("Limpiando paper trades anteriores")
        cleaned_files = []

        if os.path.exists(self.paper_trades_dir):
            for root, dirs, files in os.walk(self.paper_trades_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        cleaned_files.append(file_path)
                        self.logger.debug(f"Eliminado paper trade: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"No se pudo eliminar {file_path}: {e}")

        return cleaned_files

    def cleanup_persistent_state(self) -> List[str]:
        """Limpia estado persistente inconsistente

        Returns:
            List: Archivos de estado persistente eliminados
        """
        self.logger.info("Limpiando estado persistente")
        cleaned_files = []

        if os.path.exists(self.persistent_state_dir):
            for root, dirs, files in os.walk(self.persistent_state_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        cleaned_files.append(file_path)
                        self.logger.debug(f"Eliminado estado persistente: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"No se pudo eliminar {file_path}: {e}")

        return cleaned_files