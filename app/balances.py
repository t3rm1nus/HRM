# app/balances.py - Verificación de balances del cliente simulado

import inspect
from typing import Dict, Optional

from core.logging import logger

from app.constantes import BALANCES_INICIALES, MONEDAS


async def ejecutar_sync_o_async(funcion, *argumentos):
    resultado = funcion(*argumentos)
    return await resultado if inspect.isawaitable(resultado) else resultado


def obtener_cliente_paper(portfolio_manager):
    """Devuelve el cliente si opera en paper/simulado, si no None."""
    cliente = getattr(portfolio_manager, 'client', None)
    if not cliente:
        return None
    es_paper = getattr(
        cliente, 'paper_mode', getattr(portfolio_manager, 'mode', None) == "simulated"
    )
    return cliente if es_paper else None


async def obtener_balances(cliente) -> Optional[Dict[str, float]]:
    if hasattr(cliente, 'get_account_balances'):
        return await ejecutar_sync_o_async(cliente.get_account_balances)
    if hasattr(cliente, 'get_balance'):
        return {m: await ejecutar_sync_o_async(cliente.get_balance, m) for m in MONEDAS}
    return None


def todos_los_balances_en_cero(balances: Dict[str, float]) -> bool:
    return all(balances.get(moneda, 0.0) == 0.0 for moneda in MONEDAS)


async def restaurar_balances_iniciales(cliente):
    logger.critical("ERROR: Pérdida de estado - todos los balances son cero")
    logger.info("🔧 Intentando restaurar balances iniciales...")
    try:
        if not hasattr(cliente, 'force_reset'):
            raise RuntimeError("force_reset no disponible")
        cliente.force_reset(BALANCES_INICIALES)
        logger.info("✅ Balances iniciales restaurados")
        logger.info(f"✅ Nuevos balances: {await obtener_balances(cliente)}")
    except Exception as error:
        logger.error(f"❌ Error al restaurar balances: {error}")
        raise RuntimeError("Pérdida de estado - todos los balances son cero")


async def verificar_balances(portfolio_manager) -> bool:
    """Devuelve False si el ciclo debe saltarse por no poder leer balances."""
    cliente = obtener_cliente_paper(portfolio_manager)
    if cliente is None:
        return True

    balances = await obtener_balances(cliente)
    if balances is None:
        logger.warning("⚠️ Client has no get_account_balances or get_balance method")
        return False

    logger.info(f"   SIM_STATE_ID: {id(cliente)}")
    logger.info(f"   SIM_BALANCES: {balances}")
    if todos_los_balances_en_cero(balances):
        await restaurar_balances_iniciales(cliente)
    return True
