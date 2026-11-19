# app/despliegue.py - Despliegue inicial de posiciones

from core.data_validator import validate_market_data
from core.logging import logger

from app.modelos import Sistema


async def ejecutar_ordenes(sistema: Sistema, ordenes: list, etiqueta: str):
    procesadas = await sistema.order_manager.execute_orders(ordenes)
    await sistema.portfolio_manager.update_from_orders_async(procesadas)
    logger.info(f"✅ {etiqueta}: {len(procesadas)} orders executed")


async def desplegar_posiciones_iniciales(sistema: Sistema, market_data: dict):
    if not validate_market_data(market_data):
        logger.warning("⚠️ Skipping deployment - invalid market data")
        return

    logger.info("🚀 Calculating bootstrap deployment...")
    try:
        rotador = sistema.position_rotator
        ordenes = await rotador.calculate_bootstrap_deployment(market_data)
        etiqueta = "Bootstrap deployment"
        if not ordenes:
            logger.info("⏸️ No bootstrap deployment needed - checking initial deployment")
            ordenes = await rotador.calculate_initial_deployment(market_data)
            etiqueta = "Initial deployment"

        if ordenes:
            await ejecutar_ordenes(sistema, ordenes, etiqueta)
        else:
            logger.info("⚠️ No initial deployment needed")
    except Exception as error:
        logger.error(f"❌ Deployment failed: {error}")
