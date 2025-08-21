"""
L1_operational - Nivel de ejecución de órdenes.
Solo ejecuta órdenes seguras, sin tomar decisiones estratégicas ni tácticas.
"""

import asyncio
from loguru import logger
from l1_operational.models import Signal, ExecutionReport
from l1_operational.order_manager import order_manager
from l1_operational.bus_adapter import bus_adapter

async def procesar_l1(state: dict) -> dict:
    """
    Procesa las órdenes recibidas desde L2, las valida y ejecuta.
    L1 solo ejecuta órdenes seguras, sin tomar decisiones estratégicas ni tácticas.
    """
    nuevas_ordenes = []

    logger.info("[L1] Iniciando procesamiento de órdenes. Cantidad inicial: {}", len(state.get("ordenes", [])))

    for orden in state.get("ordenes", []):
        try:
            # Convertir orden del estado a señal
            signal = Signal(
                signal_id=orden.get("id", f"signal_{len(nuevas_ordenes)}"),
                strategy_id=orden.get("strategy_id", "unknown"),
                timestamp=orden.get("timestamp", 0),
                symbol=orden["symbol"],
                side=orden["side"],
                qty=orden["amount"],
                order_type=orden.get("type", "market"),
                price=orden.get("price"),
                risk=orden.get("risk", {}),
                metadata=orden.get("metadata", {})
            )
            logger.info(f"[L1] Señal convertida: {signal.signal_id} - {signal.side} {signal.qty} {signal.symbol}")

            # Procesar señal usando el gestor de órdenes (ahora con validaciones integradas)
            report: ExecutionReport = await order_manager.handle_signal(signal)

            nuevas_ordenes.append({
                "id": report.client_order_id,
                "status": report.status,
                "symbol": signal.symbol,
                "side": signal.side,
                "amount": signal.qty,
                "price": signal.price,
                "execution_report": {
                    "filled_qty": getattr(report, "filled_qty", 0),
                    "avg_price": getattr(report, "avg_price", 0),
                    "fees": getattr(report, "fees", 0),
                    "slippage_bps": getattr(report, "slippage_bps", 0),
                    "latency_ms": getattr(report, "latency_ms", 0),
                    "error_code": getattr(report, "error_code", None),
                    "error_msg": getattr(report, "error_msg", None)
                }
            })
            logger.info(f"[L1] Reporte generado para {report.client_order_id}: status={report.status}")

        except Exception as e:
            logger.error(f"[L1] Error procesando orden: {e}")
            nuevas_ordenes.append({
                "id": f"error_{len(nuevas_ordenes)}",
                "status": "error",
                "error_code": "EXCEPTION",
                "error_msg": str(e)
            })

    state["ordenes"] = nuevas_ordenes

    # Métricas de L1
    active_orders = await order_manager.get_active_orders_count() if asyncio.iscoroutinefunction(order_manager.get_active_orders_count) else order_manager.get_active_orders_count()
    pending_reports = await bus_adapter.get_pending_reports() if asyncio.iscoroutinefunction(bus_adapter.get_pending_reports) else bus_adapter.get_pending_reports()
    pending_alerts = await bus_adapter.get_pending_alerts() if asyncio.iscoroutinefunction(bus_adapter.get_pending_alerts) else bus_adapter.get_pending_alerts()

    state["l1_metrics"] = {
        "active_orders": active_orders,
        "pending_reports": len(pending_reports),
        "pending_alerts": len(pending_alerts)
    }
    logger.info(f"[L1] Métricas actualizadas: active_orders={active_orders}, pending_reports={len(pending_reports)}, pending_alerts={len(pending_alerts)}")

    return state

def get_l1_status() -> dict:
    """
    Retorna el estado actual de L1.
    """
    status = {
        "active_orders": order_manager.get_active_orders_count(),
        "pending_reports": len(bus_adapter.get_pending_reports()),
        "pending_alerts": len(bus_adapter.get_pending_alerts()),
        "risk_limits": "configurados",
        "execution_mode": "determinista"
    }
    logger.info(f"[L1] Estado consultado: {status}")
    return status