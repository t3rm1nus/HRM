# app/datos_mercado.py - Carga inicial y warmup de datos de mercado

import asyncio
from typing import Optional

from core.logging import logger
from system.market_data_manager import MarketDataManager

from app.constantes import SIMBOLOS
from app.modelos import Sistema


async def cargar_datos_iniciales(sistema: Sistema) -> dict:
    logger.info("🔄 Fetching initial market data...")
    try:
        market_data = await sistema.market_data_manager.get_data_with_fallback()
    except Exception as error:
        logger.error(f"❌ Market data fetch failed: {error}")
        return {}

    if not market_data:
        logger.warning("⚠️ No market data available, but continuing with empty data")
        return {}

    sistema.estado["market_data"] = market_data
    logger.info(f"✅ Market data ready: {len(market_data)} symbols")
    return market_data


async def calentar_datos_mercado(market_data_manager: MarketDataManager):
    logger.info("🔥 Ejecutando warmup_all_symbols()...")
    if await market_data_manager.warmup_all_symbols(timeframe="1m", limit=100):
        return
    for simbolo in SIMBOLOS:
        logger.warning(f"⚠️ Warmup general falló, intentando {simbolo}...")
        if await market_data_manager.force_warmup(symbol=simbolo, timeframe="1m", limit=100):
            return


async def asegurar_datos_mercado(market_data_manager: MarketDataManager) -> Optional[dict]:
    """Devuelve los datos tras el warmup, o None si no hay forma de obtenerlos."""
    await calentar_datos_mercado(market_data_manager)
    datos = await market_data_manager.get_data_with_fallback()
    if datos:
        return datos

    logger.critical("🚨 FAIL FAST: market_data está vacío después de bootstrap/warmup")
    logger.critical("   Intentando reintentar warmup...")
    await asyncio.sleep(5)
    if await market_data_manager.force_warmup(symbol="BTCUSDT", timeframe="1m", limit=100):
        logger.info("✅ Retry warmup exitoso - continuando al trading loop")
        return await market_data_manager.get_data_with_fallback()

    logger.critical("❌ Retry warmup falló - abortando")
    return None
