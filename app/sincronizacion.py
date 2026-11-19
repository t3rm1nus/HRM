# app/sincronizacion.py - Sincronización de cartera y NAV tras ejecutar órdenes

from typing import Dict, Optional

from core.logging import logger

from app.constantes import SIMBOLOS
from app.modelos import Sistema


def obtener_cliente_simulado(order_manager):
    if hasattr(order_manager, 'simulated_client'):
        return order_manager.simulated_client
    return getattr(order_manager, 'client', None)


def extraer_precio_cierre(datos) -> Optional[float]:
    if isinstance(datos, dict) and "close" in datos:
        return datos["close"]
    if hasattr(datos, 'iloc'):
        return datos.iloc[-1]["close"]
    return None


def obtener_precios_mercado(cliente, market_data: dict) -> Dict[str, float]:
    if cliente and hasattr(cliente, 'get_market_price'):
        return {simbolo: cliente.get_market_price(simbolo) for simbolo in SIMBOLOS}

    precios = {}
    for simbolo in SIMBOLOS:
        precio = extraer_precio_cierre(market_data.get(simbolo))
        if precio is not None:
            precios[simbolo] = precio
    return precios


async def sincronizar_portfolio(sistema: Sistema, market_data: dict, ordenes_ejecutadas: int):
    logger.info(f"🔄 Syncing portfolio after {ordenes_ejecutadas} executed orders...")
    cliente = obtener_cliente_simulado(sistema.order_manager)

    if cliente:
        if await sistema.portfolio_manager.sync_from_exchange_async(cliente):
            logger.info("✅ Portfolio synced successfully after order execution")
        else:
            logger.error("❌ Portfolio sync failed after order execution")

    precios = obtener_precios_mercado(cliente, market_data)
    if precios:
        await sistema.portfolio_manager.update_nav_async(precios)
