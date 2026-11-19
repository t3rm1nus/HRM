# app/registro_ciclo.py - Logs de cada ciclo de trading

from core.logging import logger

from app.comparacion_portfolio import log_portfolio_comparison
from app.modelos import Acumulados, Sistema


def registrar_resumen_ciclo(ciclo_id: int, resultado, acumulados: Acumulados):
    logger.info("=" * 80)
    logger.info(f"📊 CYCLE {ciclo_id} SUMMARY")
    logger.info(f"   Signals: {resultado.signals_generated}")
    logger.info(f"   Orders Executed: {resultado.orders_executed}")
    logger.info(f"   Orders Rejected: {resultado.orders_rejected}")
    logger.info(f"   Cooldown Blocked: {resultado.cooldown_blocked}")
    logger.info(f"   Portfolio Value: ${resultado.portfolio_value:.2f}")
    logger.info(f"   L3 Regime: {resultado.l3_regime}")
    logger.info("=" * 80)
    logger.info(f"📈 CUMULATIVE (Cycles 1-{ciclo_id})")
    logger.info(f"   Total Signals: {acumulados.senales}")
    logger.info(f"   Total Orders: {acumulados.ordenes}")
    logger.info(f"   Total Rejected: {acumulados.rechazadas}")
    logger.info(f"   Avg Orders/Cycle: {acumulados.ordenes / ciclo_id:.2f}")
    logger.info("=" * 80)


def registrar_precios_actuales(market_data: dict):
    for simbolo, df in market_data.items():
        if df is not None and not df.empty:
            logger.info(f"📊 {simbolo}: ${float(df.iloc[-1]['close']):.2f}")


async def registrar_comparacion_portfolio(sistema: Sistema, ciclo_id: int, market_data: dict):
    await log_portfolio_comparison(
        cycle_context=sistema.trading_pipeline.get_cycle_context(),
        portfolio_manager=sistema.portfolio_manager,
        cycle_id=ciclo_id,
        market_data=market_data,
    )
