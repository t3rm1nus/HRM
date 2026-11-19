# app/limpieza.py - Limpieza inicial del sistema

from core.logging import logger

try:
    from system.system_cleanup import perform_full_cleanup, force_paper_mode
    LIMPIEZA_DISPONIBLE = True
except ImportError:
    logger.warning("⚠️ SystemCleanup not available, skipping cleanup")
    LIMPIEZA_DISPONIBLE = False


def forzar_paper_como_respaldo():
    try:
        force_paper_mode()
    except Exception:
        pass


def registrar_resultado_limpieza(resultado: dict):
    if not resultado.get("success", False):
        logger.warning("⚠️ Cleanup completed with warnings")
        return
    archivos = resultado.get('filesystem', {}).get('deleted_files', 0)
    logger.info("✅ Cleanup completo exitoso")
    logger.info(f"   📁 Archivos eliminados: {archivos}")
    logger.info("   🧠 Singletons reseteados")
    logger.info("   🎯 Modo forzado: paper")


def limpiar_sistema():
    logger.info("🧹 Running system cleanup...")
    if not LIMPIEZA_DISPONIBLE:
        logger.info("⚠️ SystemCleanup not available, skipping")
        return
    try:
        registrar_resultado_limpieza(perform_full_cleanup(mode="paper"))
    except Exception as error:
        logger.warning(f"⚠️ Cleanup failed: {error}")
        forzar_paper_como_respaldo()


def limpiar_operaciones_paper():
    try:
        from storage.paper_trade_logger import get_paper_logger
        get_paper_logger(clear_on_init=True)
        logger.info("✅ Paper trades cleared")
    except Exception as error:
        logger.warning(f"⚠️ Paper trades cleanup failed: {error}")
