# app/arranque.py - Secuencia completa de inicio del sistema

from typing import Optional

from core.logging import logger
from core.state_manager import enforce_fundamental_rule

from app.autoaprendizaje import integrar_auto_aprendizaje
from app.datos_mercado import asegurar_datos_mercado, cargar_datos_iniciales
from app.despliegue import desplegar_posiciones_iniciales
from app.limpieza import limpiar_operaciones_paper, limpiar_sistema
from app.modelos import Sistema
from app.sistema import crear_sistema


def aplicar_regla_fundamental():
    logger.info("🛡️ Applying FUNDAMENTAL RULE for simulated mode...")
    try:
        enforce_fundamental_rule()
        logger.info("✅ FUNDAMENTAL RULE applied successfully")
    except Exception as error:
        logger.error(f"❌ FUNDAMENTAL RULE failed: {error}")


async def iniciar_sistema() -> Optional[Sistema]:
    """Devuelve el sistema listo, o None si no hay datos de mercado."""
    limpiar_sistema()
    limpiar_operaciones_paper()
    logger.info("🚀 Starting HRM system")

    sistema = await crear_sistema()
    datos_iniciales = await cargar_datos_iniciales(sistema)
    await desplegar_posiciones_iniciales(sistema, datos_iniciales)
    await integrar_auto_aprendizaje(sistema)
    aplicar_regla_fundamental()

    datos = await asegurar_datos_mercado(sistema.market_data_manager)
    if datos is None:
        return None

    sistema.estado["market_data"] = datos
    logger.info(f"✅ Market data verificado: {len(datos)} símbolos")
    return sistema
