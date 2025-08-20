# l1_operational/__init__.py
"""
L1_operational - Nivel de ejecución de órdenes.
Solo ejecuta órdenes seguras, sin tomar decisiones estratégicas ni tácticas.
"""

from l1_operational.order_manager import order_manager
from l1_operational.bus_adapter import bus_adapter
from l1_operational.risk_guard import validate_order
from l1_operational.executor import execute_order
from l1_operational.data_feed import get_ticker, get_balance
from l1_operational.models import Signal, ExecutionReport, RiskAlert
from loguru import logger

def procesar_l1(state: dict) -> dict:
    """
    Procesa las órdenes recibidas desde L2, las valida y ejecuta en Binance.
    L1 solo ejecuta órdenes seguras, sin tomar decisiones estratégicas ni tácticas.
    """
    nuevas_ordenes = []
    
    # Procesar señales del bus (en un sistema real, esto vendría de L2/L3)
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
            
            # Procesar señal usando el gestor de órdenes
            report = order_manager.handle_signal(signal)
            
            # Agregar resultado al estado
            nuevas_ordenes.append({
                "id": report.client_order_id,
                "status": report.status,
                "symbol": signal.symbol,
                "side": signal.side,
                "amount": signal.qty,
                "price": signal.price,
                "execution_report": {
                    "filled_qty": report.filled_qty,
                    "avg_price": report.avg_price,
                    "fees": report.fees,
                    "slippage_bps": report.slippage_bps,
                    "latency_ms": report.latency_ms,
                    "error_code": report.error_code,
                    "error_msg": report.error_msg
                }
            })
            
        except Exception as e:
            logger.error(f"Error procesando orden: {e}")
            nuevas_ordenes.append({
                "id": f"error_{len(nuevas_ordenes)}",
                "status": "error",
                "error": str(e)
            })

    # L1 no actualiza portfolio ni mercado - eso es responsabilidad de niveles superiores
    state["ordenes"] = nuevas_ordenes
    
    # Agregar métricas de L1 al estado
    state["l1_metrics"] = {
        "active_orders": order_manager.get_active_orders_count(),
        "pending_reports": len(bus_adapter.get_pending_reports()),
        "pending_alerts": len(bus_adapter.get_pending_alerts())
    }
    
    return state

def get_l1_status() -> dict:
    """
    Retorna el estado actual de L1.
    """
    return {
        "active_orders": order_manager.get_active_orders_count(),
        "pending_reports": len(bus_adapter.get_pending_reports()),
        "pending_alerts": len(bus_adapter.get_pending_alerts()),
        "risk_limits": "configurados",
        "execution_mode": "determinista"
    }
