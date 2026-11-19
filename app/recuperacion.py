# app/recuperacion.py - Recuperación ante errores de un ciclo

import asyncio

from core.logging import logger
from system.error_recovery_manager import RecoveryActionType

from app.constantes import SEGUNDOS_ESPERA_ERROR
from app.modelos import Sistema


async def recuperar_de_error(sistema: Sistema, error: Exception, ciclo_id: int) -> bool:
    """Devuelve False si el error no es recuperable y hay que parar el bucle."""
    try:
        accion = await sistema.error_recovery.handle_cycle_error(
            error=error, state=sistema.estado, cycle_id=ciclo_id
        )
        if accion.action == RecoveryActionType.SHUTDOWN:
            logger.critical("🛑 Unrecoverable error - shutting down")
            return False
        if accion.action == RecoveryActionType.RESET_COMPONENT:
            logger.warning("🔄 Resetting component...")
            for paso in accion.recovery_steps_taken:
                logger.info(f"   ✓ {paso}")
        await asyncio.sleep(accion.wait_seconds)
    except Exception as error_recuperacion:
        logger.error(f"❌ Recovery failed: {error_recuperacion}")
        await asyncio.sleep(SEGUNDOS_ESPERA_ERROR)
    return True
