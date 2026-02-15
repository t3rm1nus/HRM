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
from .order_manager import OrderManager
from .bus_adapter import BusAdapterAsync  # Importamos la clase, no la instancia

# Importar SmartCooldownManager si existe
try:
    from .smart_cooldown_manager import SmartCooldownManager
    SMART_COOLDOWN_AVAILABLE = True
except ImportError:
    SMART_COOLDOWN_AVAILABLE = False
    logger.warning("⚠️ SmartCooldownManager no disponible en __init__.py")

async def procesar_l1(state: dict) -> dict:
    """
    Procesa las órdenes recibidas desde L2, las valida y ejecuta.
    L1 solo ejecuta órdenes seguras, sin tomar decisiones estratégicas ni tácticas.
    Compatible con la arquitectura de state del sistema HRM.
    """
    logger.debug("[L1] Iniciando procesamiento de órdenes en L1")
    nuevas_ordenes = []

    # Verificar si hay órdenes en el estado
    ordenes = state.get("ordenes", [])
    logger.info(f"[L1] Cantidad de órdenes iniciales: {len(ordenes)}")

    if not ordenes:
        logger.warning("[L1] No se encontraron órdenes en state['ordenes']. Verificando señales...")
        signals = state.get("senales", {}).get("signals", [])
        logger.debug(f"[L1] Señales disponibles: {len(signals)}")
        for signal in signals:
            try:
                if signal["confidence"] < 0.6:
                    logger.debug(f"[L1] Señal para {signal['symbol']} descartada: confianza {signal['confidence']} < 0.6")
                    continue

                symbol = signal["symbol"]
                if symbol not in state["mercado"]:
                    logger.error(f"[L1] Símbolo {symbol} no encontrado en datos de mercado")
                    continue

                price = state["mercado"][symbol]["close"].iloc[-1]
                sim_signal = Signal(
                    signal_id=f"signal_{symbol}_{time.time()}",
                    strategy_id="l2_tactic",
                    timestamp=time.time(),
                    symbol=symbol,
                    side=signal["direction"],
                    qty=0.1,
                    order_type="market",
                    price=price,
                    confidence=signal["confidence"]
                )
                logger.info(f"[L1] Señal convertida a orden simulada: {sim_signal.signal_id}")

                report = await OrderManager.handle_signal(sim_signal)
                nuevas_ordenes.append({
                    "id": sim_signal.signal_id,
                    "status": _map_status_to_legacy(report.status),
                    "symbol": sim_signal.symbol,
                    "side": sim_signal.side,
                    "amount": sim_signal.qty,
                    "price": sim_signal.price,
                    "execution_report": {
                        "filled_qty": report.executed_qty or 0,
                        "avg_price": report.executed_price or 0,
                        "fees": report.fees or 0,
                        "slippage_bps": 0,
                        "latency_ms": report.latency_ms or 0,
                        "error_code": "RISK_REJECTION" if report.status.startswith("REJECTED") else None,
                        "error_msg": report.reason,
                        "ai_confidence": report.ai_confidence,
                        "ai_risk_score": report.ai_risk_score
                    }
                })
                logger.info(f"[L1] Reporte generado para {sim_signal.signal_id}: status={report.status}")
            except Exception as e:
                logger.error(f"[L1] Error procesando señal simulada: {e}", exc_info=True)
                nuevas_ordenes.append({
                    "id": f"error_{len(nuevas_ordenes)}",
                    "status": "error",
                    "error_code": "EXCEPTION",
                    "error_msg": str(e)
                })

    for orden in ordenes:
        try:
            signal = Signal(
                signal_id=orden.get("id", f"signal_{len(nuevas_ordenes)}"),
                strategy_id=orden.get("strategy_id", "unknown"),
                timestamp=orden.get("timestamp", time.time()),
                symbol=orden["symbol"],
                side=orden["side"],
                qty=orden.get("quantity", orden.get("amount", 0.0)),
                order_type=orden.get("type", "market"),
                price=orden.get("price"),
                stop_loss=orden.get("risk", {}).get("stop_loss"),
                take_profit=orden.get("risk", {}).get("take_profit"),
                confidence=orden.get("metadata", {}).get("confidence", 0.5),
                technical_indicators=orden.get("metadata", {}).get("technical_indicators", {})
            )
            logger.info(f"[L1] Señal convertida: {signal.signal_id} - {signal.side} {signal.qty} {signal.symbol}")

            report = await OrderManager.handle_signal(signal)
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
                    "slippage_bps": 0,
                    "latency_ms": report.latency_ms or 0,
                    "error_code": "RISK_REJECTION" if report.status.startswith("REJECTED") else None,
                    "error_msg": report.reason,
                    "ai_confidence": report.ai_confidence,
                    "ai_risk_score": report.ai_risk_score
                }
            })
            logger.info(f"[L1] Reporte generado para {signal.signal_id}: status={report.status}")
        except Exception as e:
            logger.error(f"[L1] Error procesando orden: {e}", exc_info=True)
            nuevas_ordenes.append({
                "id": f"error_{len(nuevas_ordenes)}",
                "status": "error",
                "error_code": "EXCEPTION",
                "error_msg": str(e)
            })

    state["ordenes"] = nuevas_ordenes

    metrics = OrderManager.get_metrics()
    state["l1_metrics"] = {
        "active_orders": metrics["total_signals_processed"],
        "pending_reports": 0,
        "pending_alerts": 0,
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
    logger.debug(f"[L1] Órdenes procesadas: {nuevas_ordenes}")

    return state

def _map_status_to_legacy(new_status: str) -> str:
    status_mapping = {
        "EXECUTED": "filled",
        "REJECTED_SAFETY": "rejected",
        "REJECTED_AI": "rejected", 
        "EXECUTION_ERROR": "error",
        "PROCESSING_ERROR": "error"
    }
    return status_mapping.get(new_status, "unknown")

def get_l1_status() -> dict:
    metrics = OrderManager.get_metrics()
    status = {
        "active_orders": metrics["total_signals_processed"],
        "pending_reports": 0,
        "pending_alerts": 0,
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

def get_l1_metrics():
    return OrderManager.get_metrics()

__all__ = ['procesar_l1', 'get_l1_status', 'get_l1_metrics',
           'Signal', 'ExecutionReport', 'RiskAlert', 'OrderIntent',
           'BusAdapterAsync', 'SmartCooldownManager']  # Incluimos el nuevo manager