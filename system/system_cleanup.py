# system/system_cleanup.py
"""
System Cleanup - Limpieza del sistema antes de iniciar una nueva sesión.
"""

import os
import shutil
import asyncio
from pathlib import Path
from typing import Any, Dict
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
            self._clean_old_logs()
            self._clean_cache()
            self._clean_temp_data()
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
                mod_time = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
                if mod_time < cutoff_date:
                    log_file.unlink()
                    logger.debug(f"Deleted old log: {log_file.name}")
            except Exception as e:
                raise OSError(f"No se pudo eliminar el log antiguo {log_file}: {e}") from e

    def _clean_cache(self) -> None:
        """Limpia el directorio de caché."""
        if not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file() and cache_file.suffix in [".tmp", ".cache", ".pickle"]:
                try:
                    cache_file.unlink()
                    logger.debug(f"Deleted cache file: {cache_file.name}")
                except Exception as e:
                    raise OSError(f"No se pudo eliminar el archivo de caché {cache_file}: {e}") from e

    def _clean_temp_data(self) -> None:
        """Limpia archivos temporales de datos."""
        if not self.data_dir.exists():
            return

        for pattern in ("*.tmp", "*_temp.*"):
            for temp_file in self.data_dir.rglob(pattern):
                try:
                    temp_file.unlink()
                    logger.debug(f"Deleted temp file: {temp_file.name}")
                except Exception as e:
                    raise OSError(f"No se pudo eliminar el archivo temporal {temp_file}: {e}") from e

    def _ensure_directories(self) -> None:
        """Asegura que los directorios necesarios existan."""
        directories = [
            self.data_dir,
            self.data_dir / "paper" / "persistente" / "paper_trades",
            self.data_dir / "paper" / "limpia" / "paper_trades",
            self.data_dir / "models",
            self.data_dir / "backtest",
            self.logs_dir,
            self.cache_dir,
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
            "data_files_count": len(list(self.data_dir.rglob("*"))) if self.data_dir.exists() else 0,
        }


def perform_full_cleanup(mode: str = "paper") -> Dict[str, Any]:
    """
    Realiza una limpieza técnica exclusivamente en modo PAPER.

    El argumento se valida explícitamente para evitar que esta rutina se
    interprete como soporte de limpieza/preparación de otros modos.
    """
    normalized_mode = str(mode).strip().lower() if mode is not None else ""
    if normalized_mode != "paper":
        return {
            "success": False,
            "mode": normalized_mode or None,
            "error": "perform_full_cleanup solo admite mode='paper'.",
            "failed_steps": ["mode_validation"],
        }

    # Ejecutar todos los pasos aunque uno falle para devolver un diagnóstico
    # completo; el éxito global solo se declara si todos terminan correctamente.
    results = {
        "filesystem": filesystem_cleanup(),
        "memory": memory_reset(),
        "paper_mode": force_paper_mode(),
        "async_context": async_context_reset(),
    }
    failed_steps = [
        name for name, result in results.items()
        if not isinstance(result, dict) or result.get("success") is not True
    ]

    return {
        "success": not failed_steps,
        "mode": "paper",
        **results,
        "failed_steps": failed_steps,
    }


def filesystem_cleanup() -> Dict[str, Any]:
    """Limpieza del filesystem (archivos, directorios, logs)."""
    try:
        cleanup = SystemCleanup()
        success = cleanup.run_cleanup()

        # Los historiales PAPER no pertenecen a la limpieza técnica del sistema.
        # Su ciclo de vida lo controla system.preparar_sesion por perfil.
        return {
            "success": success,
            "deleted_files": 0,
            "directories_ensured": 7,
        }
    except Exception as e:
        logger.error(f"❌ Filesystem cleanup failed: {e}")
        return {"success": False, "error": str(e)}


def memory_reset() -> Dict[str, Any]:
    """Resetea singletons y cachés en memoria sin invalidar módulos importados."""
    reset_items = []

    try:
        from core.portfolio_manager import PortfolioManager
        PortfolioManager.reset_instance()
        reset_items.append("PortfolioManager")

        from system.state_coordinator import StateCoordinator
        StateCoordinator.reset_instance()
        reset_items.append("StateCoordinator")

        from storage.paper_trade_logger import reiniciar_loggers_paper
        reiniciar_loggers_paper()
        reset_items.append("PaperTradeLogger")

        logger.info("✅ Estado singleton y cachés PAPER reiniciados")
        return {
            "success": True,
            "singletons_reset": ["PortfolioManager", "StateCoordinator"],
            "caches_reset": ["PaperTradeLogger"],
            "reset_items": reset_items,
        }
    except Exception as e:
        logger.error(f"❌ Memory reset failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "reset_items": reset_items,
        }


def force_paper_mode() -> Dict[str, Any]:
    """Fija la variable de entorno del proceso en PAPER."""
    try:
        os.environ["HRM_MODE"] = "paper"

        # Mantener la compatibilidad con el cargador de configuración existente.
        try:
            from core.config import get_config
            live_config = get_config("live")
            if hasattr(live_config, "PAPER_MODE"):
                live_config.PAPER_MODE = True
            logger.info("✅ PAPER_MODE forced in config")
        except Exception as e:
            logger.warning(f"⚠️ Could not force PAPER_MODE in config: {e}")

        return {
            "success": True,
            "mode": "paper",
            "env_var": "HRM_MODE=paper",
        }
    except Exception as e:
        logger.error(f"❌ Paper mode forcing failed: {e}")
        return {"success": False, "error": str(e)}


def async_context_reset() -> Dict[str, Any]:
    """Comprueba y prepara el contexto async sin sustituir un loop activo."""
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return {
                    "success": True,
                    "new_loop_created": False,
                    "message": "Loop already running - managed by asyncio",
                }
        except RuntimeError:
            pass

        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        try:
            from core.async_balance_helper import AsyncContextDetector
            AsyncContextDetector._is_in_async_context = False
            logger.info("✅ AsyncContextDetector reset")
        except ImportError:
            logger.debug("AsyncContextDetector no está disponible; no hay estado que reiniciar")

        return {"success": True, "new_loop_created": True}
    except Exception as e:
        logger.error(f"❌ Async context reset failed: {e}")
        return {"success": False, "error": str(e)}


def get_cleanup_status() -> Dict[str, Any]:
    """Obtiene el estado actual del sistema."""
    try:
        cleanup = SystemCleanup()
        report = cleanup.get_cleanup_report()
        mode = os.environ.get("HRM_MODE", "unknown")

        from core.portfolio_manager import PortfolioManager
        from system.state_coordinator import StateCoordinator

        try:
            loop = asyncio.get_running_loop()
            has_event_loop = loop.is_running()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                has_event_loop = loop.is_running()
            except RuntimeError:
                has_event_loop = False

        return {
            "mode": mode,
            "filesystem": report,
            "singletons": {
                "portfolio_manager_exists": PortfolioManager._instance is not None,
                "state_coordinator_exists": StateCoordinator._instance is not None,
            },
            "async_context": {"has_event_loop": has_event_loop},
        }
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return {"success": False, "error": str(e)}


def _validate_sentiment_cache(cache_file_path):
    """Valida que el cache de sentimiento no contiene datos de prueba."""
    try:
        import json
        with open(cache_file_path, "r") as f:
            data = json.load(f)

        texts = data.get("texts", [])
        if not texts:
            return False

        sample_count = sum(
            1 for t in texts
            if "Sample text" in str(t) or "sample text" in str(t).lower()
        )
        sample_ratio = sample_count / len(texts)

        if sample_ratio > 0.1:
            logger.warning(
                f"⚠️ Cache de sentimiento contiene {sample_ratio:.0%} datos de prueba. "
                f"Eliminando cache corrupto: {cache_file_path}"
            )
            os.remove(cache_file_path)
            return False

        return True
    except Exception:
        return False


def cleanup_corrupt_sentiment_cache():
    """Limpia archivos de sentiment cache que contienen datos de prueba."""
    import glob

    patterns = [
        "archived/sentiment_cache_*.json",
        "sentiment_cache_*.json",
        "sentiment_inference_*.csv",
    ]

    deleted_count = 0
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            if "sentiment_cache" in file_path and file_path.endswith(".json"):
                if not _validate_sentiment_cache(file_path):
                    deleted_count += 1
            else:
                try:
                    os.remove(file_path)
                    logger.info(f"🗑️ Eliminado archivo de sentiment: {file_path}")
                    deleted_count += 1
                except OSError:
                    logger.exception("No se pudo eliminar el archivo de sentiment: %s", file_path)

    if deleted_count > 0:
        logger.info(f"✅ Eliminados {deleted_count} archivos de sentiment corruptos/obsoletos")

    return deleted_count
