# app/apagado.py - Cierre ordenado del sistema

from typing import Optional

from system.shutdown_manager import shutdown
from core.incremental_signal_verifier import stop_signal_verification
from core.logging import logger

from app.modelos import Sistema


def guardar_portfolio(sistema: Optional[Sistema]):
    if sistema is None:
        return
    try:
        sistema.portfolio_manager.save_to_json()
        logger.info("💾 Portfolio state saved")
    except Exception as error:
        logger.error(f"❌ Error saving portfolio: {error}")


async def apagar_componentes():
    logger.info("🧹 Ejecutando shutdown limpio global desde bootstrap...")
    try:
        await shutdown()
        logger.info("✅ Shutdown global completado exitosamente")
    except Exception as error:
        logger.error(f"❌ Error en shutdown global: {error}")


async def detener_verificacion_senales():
    try:
        await stop_signal_verification()
        logger.info("✓ Signal verification stopped")
    except Exception as error:
        logger.warning(f"⚠️ Cleanup error: {error}")


async def apagar_sistema():
    await apagar_componentes()
    await detener_verificacion_senales()
    logger.info("👋 HRM System shutdown complete")
