# l1_operational/__init__.py
"""
L1_operational - Nivel de ejecución de órdenes.
Solo ejecuta órdenes seguras, sin tomar decisiones estratégicas ni tácticas.
Compatible con la arquitectura HRM existente.
"""

import asyncio
import time
from loguru import logger
from .models import Signal, ExecutionReport, RiskAlert, OrderIntent
from .order_manager import order_manager
from .bus_adapter import bus_adapter

async def procesar_l1(state: dict) -> dict:
    """
    Procesa las órdenes recibidas desde L2, las valida y ejecuta.
    L1 solo ejecuta órdenes seguras, sin tomar decisiones estratégicas ni tácticas.
    Compatible con la arquitectura de state del sistema HRM.
    """
    nuevas_ordenes = []

    logger.info("[L1] Iniciando procesamiento de órdenes. Cantidad inicial: {}", len(state.get("ordenes", [])))

    for orden in state.get("ordenes", []):
        try:
            # Convertir orden del estado a señal usando nuestro modelo mejorado
            signal = Signal(
                signal_id=orden.get("id", f"signal_{len(nuevas_ordenes)}"),
                strategy_id=orden.get("strategy_id", "unknown"),
                timestamp=orden.get("timestamp", time.time()),
                symbol=orden["symbol"],
                side=orden["side"],
                qty=orden["amount"],
                order_type=orden.get("type", "market"),
                price=orden.get("price"),
                stop_loss=orden.get("risk", {}).get("stop_loss"),
                take_profit=orden.get("risk", {}).get("take_profit"),
                confidence=orden.get("metadata", {}).get("confidence", 0.5),
                technical_indicators=orden.get("metadata", {}).get("technical_indicators", {})
            )
            logger.info(f"[L1] Señal convertida: {signal.signal_id} - {signal.side} {signal.qty} {signal.symbol}")

            # Procesar señal usando el gestor de órdenes mejorado con validaciones integradas
            report: ExecutionReport = await order_manager.handle_signal(signal)

            # Mapear nuestro ExecutionReport al formato esperado por el state
            nuevas_ordenes.append({
                "id": signal.signal_id,
                "status": _map_status_to_legacy(report.status),
                "symbol": signal.symbol,
                "side": signal.side,
                "amount": signal.qty,
                "price": signal.price,
                "execution_report": {
                    "filled_qty": report.executed_qty or 0,
                    "avg_price": report.executed_price or 0,
                    "fees": report.fees or 0,
                    "slippage_bps": 0,  # TODO: calcular slippage
                    "latency_ms": report.latency_ms or 0,
                    "error_code": "RISK_REJECTION" if report.status.startswith("REJECTED") else None,
                    "error_msg": report.reason,
                    "ai_confidence": report.ai_confidence,
                    "ai_risk_score": report.ai_risk_score
                }
            })
            logger.info(f"[L1] Reporte generado para {signal.signal_id}: status={report.status}")

        except Exception as e:
            logger.error(f"[L1] Error procesando orden: {e}")
            nuevas_ordenes.append({
                "id": f"error_{len(nuevas_ordenes)}",
                "status": "error",
                "error_code": "EXCEPTION",
                "error_msg": str(e)
            })

    state["ordenes"] = nuevas_ordenes

    # Métricas de L1 usando nuestro sistema mejorado
    metrics = order_manager.get_metrics()
    
    state["l1_metrics"] = {
        "active_orders": metrics["total_signals_processed"],
        "pending_reports": 0,  # Los reports se procesan inmediatamente
        "pending_alerts": 0,   # Las alertas se procesan inmediatamente
        "executed": metrics["executed"],
        "rejected_safety": metrics["rejected_safety"],
        "execution_errors": metrics["execution_errors"],
        "success_rate": metrics["success_rate"],
        "avg_latency_ms": metrics["executor_metrics"]["avg_latency_ms"],
        "current_positions": metrics["current_positions"],
        "daily_pnl": metrics["daily_pnl"]
    }
    
    logger.info(f"[L1] Métricas actualizadas: success_rate={metrics['success_rate']:.2%}, "
               f"executed={metrics['executed']}, rejected={metrics['rejected_safety']}")

    return state

def _map_status_to_legacy(new_status: str) -> str:
    """Mapea nuestros status a los esperados por el sistema legacy"""
    status_mapping = {
        "EXECUTED": "filled",
        "REJECTED_SAFETY": "rejected",
        "REJECTED_AI": "rejected", 
        "EXECUTION_ERROR": "error",
        "PROCESSING_ERROR": "error"
    }
    return status_mapping.get(new_status, "unknown")

def get_l1_status() -> dict:
    """
    Retorna el estado actual de L1 usando nuestro sistema mejorado.
    """
    metrics = order_manager.get_metrics()
    
    status = {
        "active_orders": metrics["total_signals_processed"],
        "pending_reports": 0,  # Procesamiento inmediato
        "pending_alerts": 0,   # Procesamiento inmediato  
        "risk_limits": "configurados y validados",
        "execution_mode": "determinista con IA",
        "success_rate": f"{metrics['success_rate']:.2%}",
        "avg_latency_ms": metrics["executor_metrics"]["avg_latency_ms"],
        "current_positions": metrics["current_positions"],
        "daily_pnl": metrics["daily_pnl"],
        "account_balance": metrics["account_balance"]
    }
    logger.info(f"[L1] Estado consultado: {status}")
    return status

# Mantener compatibilidad con el sistema anterior
def get_l1_metrics():
    """Alias para compatibilidad - obtiene métricas consolidadas de L1"""
    return order_manager.get_metrics()

__all__ = ['procesar_l1', 'get_l1_status', 'get_l1_metrics', 'Signal', 'ExecutionReport', 'RiskAlert', 'OrderIntent']