# system/system_cleanup.py
"""
System Cleanup - Limpieza del sistema antes de iniciar una nueva sesión.
"""

import os
import shutil
import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional
from core.logging import logger


class SystemCleanup:
    """Maneja la limpieza del sistema antes de iniciar."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.data_dir = Path("data")
        self.logs_dir = Path("logs")
        self.cache_dir = Path(".cache")

    def run_cleanup(self) -> bool:
        """Ejecuta la limpieza del sistema."""
        try:
            logger.info("🧹 Running system cleanup...")
            
            # 1. Limpiar logs antiguos (excepto el actual)
            self._clean_old_logs()
            
            # 2. Limpiar caché temporal
            self._clean_cache()
            
            # 3. Limpiar archivos temporales de datos
            self._clean_temp_data()
            
            # 4. Verificar estructura de directorios
            self._ensure_directories()
            
            logger.info("✅ System cleanup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during system cleanup: {e}")
            return False

    def _clean_old_logs(self) -> None:
        """Limpia logs antiguos (mantiene solo los últimos 7 días)."""
        if not self.logs_dir.exists():
            return
            
        import datetime
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=7)
        
        for log_file in self.logs_dir.glob("*.log"):
            try:
                # Verificar fecha de modificación
                mod_time = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
                if mod_time < cutoff_date:
                    log_file.unlink()
                    logger.debug(f"Deleted old log: {log_file.name}")
            except Exception as e:
                logger.debug(f"Could not delete {log_file}: {e}")

    def _clean_cache(self) -> None:
        """Limpia el directorio de caché."""
        if not self.cache_dir.exists():
            return
            
        try:
            # Eliminar archivos .tmp y .cache
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file():
                    if cache_file.suffix in ['.tmp', '.cache', '.pickle']:
                        cache_file.unlink()
                        logger.debug(f"Deleted cache file: {cache_file.name}")
        except Exception as e:
            logger.debug(f"Could not clean cache: {e}")

    def _clean_temp_data(self) -> None:
        """Limpia archivos temporales de datos."""
        if not self.data_dir.exists():
            return
            
        try:
            # Buscar archivos temporales en data/
            for temp_file in self.data_dir.rglob("*.tmp"):
                temp_file.unlink()
                logger.debug(f"Deleted temp file: {temp_file.name}")
                
            for temp_file in self.data_dir.rglob("*_temp.*"):
                temp_file.unlink()
                logger.debug(f"Deleted temp file: {temp_file.name}")
        except Exception as e:
            logger.debug(f"Could not clean temp data: {e}")

    def _ensure_directories(self) -> None:
        """Asegura que los directorios necesarios existan."""
        directories = [
            self.data_dir,
            self.data_dir / "paper_trades",
            self.data_dir / "models",
            self.data_dir / "backtest",
            self.logs_dir,
            self.cache_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")

    def get_cleanup_report(self) -> dict:
        """Genera un reporte de la limpieza."""
        return {
            "logs_dir_exists": self.logs_dir.exists(),
            "data_dir_exists": self.data_dir.exists(),
            "cache_dir_exists": self.cache_dir.exists(),
            "log_files_count": len(list(self.logs_dir.glob("*.log"))) if self.logs_dir.exists() else 0,
            "data_files_count": len(list(self.data_dir.rglob("*"))) if self.data_dir.exists() else 0
        }


# =========================
# FUNCIONES DE FÁCIL USO
# =========================

def perform_full_cleanup(mode: str = "paper") -> Dict[str, any]:
    """
    Realiza una limpieza completa del sistema.
    
    Args:
        mode: Modo de operación ("paper" o "live")
        
    Returns:
        Dict con resultados de la limpieza
    """
    try:
        # 1. Limpiar filesystem
        filesystem_result = filesystem_cleanup()
        
        # 2. Resetear singletons
        memory_result = memory_reset()
        
        # 3. Forzar modo paper
        paper_result = force_paper_mode()
        
        # 4. Resetear contexto async
        async_result = async_context_reset()
        
        return {
            "success": True,
            "filesystem": filesystem_result,
            "memory": memory_result,
            "paper_mode": paper_result,
            "async_context": async_result
        }
        
    except Exception as e:
        logger.error(f"❌ Full cleanup failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def filesystem_cleanup() -> Dict[str, any]:
    """Limpieza del filesystem (archivos, directorios, logs)."""
    try:
        cleanup = SystemCleanup()
        success = cleanup.run_cleanup()
        
        # Limpiar paper trades
        try:
            from storage.paper_trade_logger import get_paper_logger
            get_paper_logger(clear_on_init=True)
            logger.info("✅ Paper trades cleared")
        except Exception as e:
            logger.warning(f"⚠️ Paper trades cleanup failed: {e}")
        
        return {
            "success": success,
            "deleted_files": 0,  # Could be counted if needed
            "directories_ensured": 6
        }
        
    except Exception as e:
        logger.error(f"❌ Filesystem cleanup failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def memory_reset() -> Dict[str, any]:
    """Resetear singletons y estado en memoria."""
    try:
        # Resetear PortfolioManager singleton
        from core.portfolio_manager import PortfolioManager
        PortfolioManager.reset_instance()
        logger.info("✅ PortfolioManager singleton reset")
        
        # Resetear StateCoordinator singleton
        from system.state_coordinator import StateCoordinator
        StateCoordinator.reset_instance()
        logger.info("✅ StateCoordinator singleton reset")
        
        # Limpiar cache de importación
        if 'core.portfolio_manager' in sys.modules:
            del sys.modules['core.portfolio_manager']
        if 'system.state_coordinator' in sys.modules:
            del sys.modules['system.state_coordinator']
            
        return {
            "success": True,
            "singletons_reset": ["PortfolioManager", "StateCoordinator"]
        }
        
    except Exception as e:
        logger.error(f"❌ Memory reset failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def force_paper_mode() -> Dict[str, any]:
    """Forzar modo paper en todas las configuraciones."""
    try:
        # Setear variable de entorno
        os.environ['HRM_MODE'] = 'paper'
        
        # Forzar modo paper en config
        try:
            from core.config import get_config
            live_config = get_config("live")
            if hasattr(live_config, 'PAPER_MODE'):
                live_config.PAPER_MODE = True
            logger.info("✅ PAPER_MODE forced in config")
        except Exception as e:
            logger.warning(f"⚠️ Could not force PAPER_MODE in config: {e}")
        
        return {
            "success": True,
            "mode": "paper",
            "env_var": "HRM_MODE=paper"
        }
        
    except Exception as e:
        logger.error(f"❌ Paper mode forcing failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def async_context_reset() -> Dict[str, any]:
    """Resetear contexto async para evitar initialize_async vs init issues."""
    try:
        # Limpiar asyncio event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("⚠️ Async loop already running, skipping reset")
                return {"success": True, "message": "Loop already running"}
        except:
            pass
        
        # Crear nuevo event loop
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        
        # Resetear AsyncContextDetector
        try:
            from core.async_balance_helper import AsyncContextDetector
            AsyncContextDetector._is_in_async_context = False
            logger.info("✅ AsyncContextDetector reset")
        except Exception as e:
            logger.warning(f"⚠️ Could not reset AsyncContextDetector: {e}")
        
        return {
            "success": True,
            "new_loop_created": True
        }
        
    except Exception as e:
        logger.error(f"❌ Async context reset failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_cleanup_status() -> Dict[str, any]:
    """Obtener estado actual del sistema."""
    try:
        cleanup = SystemCleanup()
        report = cleanup.get_cleanup_report()
        
        # Verificar modo
        mode = os.environ.get('HRM_MODE', 'unknown')
        
        # Verificar singletons
        from core.portfolio_manager import PortfolioManager
        from system.state_coordinator import StateCoordinator
        
        return {
            "mode": mode,
            "filesystem": report,
            "singletons": {
                "portfolio_manager_exists": PortfolioManager._instance is not None,
                "state_coordinator_exists": StateCoordinator._instance is not None
            },
            "async_context": {
                "has_event_loop": asyncio.get_event_loop() is not None
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
