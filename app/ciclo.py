# app/ciclo.py - Bucle principal de trading

import asyncio
from typing import Optional

import pandas as pd

from core.logging import logger
from system.market_data_manager import MarketDataManager

from app.balances import verificar_balances
from app.constantes import CICLOS_ENTRE_RESUMENES, SEGUNDOS_POR_CICLO
from app.modelos import Acumulados, Sistema
from app.recuperacion import recuperar_de_error
from app.registro_ciclo import (
    registrar_comparacion_portfolio,
    registrar_precios_actuales,
    registrar_resumen_ciclo,
)
from app.sincronizacion import sincronizar_portfolio


async def esperar_siguiente_ciclo(ultimo_ciclo: pd.Timestamp) -> pd.Timestamp:
    transcurrido = (pd.Timestamp.utcnow() - ultimo_ciclo).total_seconds()
    if transcurrido < SEGUNDOS_POR_CICLO:
        espera = SEGUNDOS_POR_CICLO - transcurrido
        logger.debug(f"⏱️ Waiting {espera:.2f}s for 3-second cycle")
        await asyncio.sleep(espera)
    return pd.Timestamp.utcnow()


async def obtener_datos_ciclo(market_data_manager: MarketDataManager, ciclo_id: int) -> Optional[dict]:
    try:
        market_data = await market_data_manager.get_data_with_fallback()
    except Exception as error:
        logger.error(f"❌ Cycle {ciclo_id}: Market data error: {error}")
        return None

    if not market_data:
        logger.warning(f"⚠️ Cycle {ciclo_id}: No market data, skipping")
        return None
    return market_data


async def ejecutar_ciclo(sistema: Sistema, ciclo_id: int, acumulados: Acumulados):
    logger.info(f"🔍 Cycle {ciclo_id}: Checking SimulatedExchangeClient state")
    if not await verificar_balances(sistema.portfolio_manager):
        return

    market_data = await obtener_datos_ciclo(sistema.market_data_manager, ciclo_id)
    if market_data is None:
        await asyncio.sleep(SEGUNDOS_POR_CICLO)
        return
    sistema.estado["market_data"] = market_data

    resultado = await sistema.trading_pipeline.process_trading_cycle(
        state=sistema.estado,
        market_data=market_data,
        cycle_id=ciclo_id,
    )
    sistema.estado["total_value"] = resultado.portfolio_value
    acumulados.sumar(resultado)

    if resultado.orders_executed > 0:
        await sincronizar_portfolio(sistema, market_data, resultado.orders_executed)
    if ciclo_id % CICLOS_ENTRE_RESUMENES == 0:
        registrar_resumen_ciclo(ciclo_id, resultado, acumulados)

    registrar_precios_actuales(market_data)
    await registrar_comparacion_portfolio(sistema, ciclo_id, market_data)


async def ejecutar_bucle_trading(sistema: Sistema):
    logger.info("🔄 Starting main trading loop...")
    logger.info("=" * 80)

    acumulados = Acumulados()
    ultimo_ciclo = pd.Timestamp.utcnow()
    ciclo_id = 0

    while True:
        ciclo_id += 1
        ultimo_ciclo = await esperar_siguiente_ciclo(ultimo_ciclo)
        try:
            await ejecutar_ciclo(sistema, ciclo_id, acumulados)
        except Exception as error:
            logger.error(f"❌ Cycle {ciclo_id} error: {error}")
            if not await recuperar_de_error(sistema, error, ciclo_id):
                break
