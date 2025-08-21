import asyncio
from core.logging import setup_logger
from l1_operational.models import Signal, ExecutionReport
from l1_operational.bus_adapter import bus_adapter
from l1_operational.risk_guard import validate_order
from l1_operational.trend_ai import filter_signal
from l1_operational.executor import execute_order

logger = setup_logger()


class OrderManager:
    """
    Gestor de órdenes L1.
    """
    def __init__(self):
        self.active_orders = {}
        logger.info("[OrderManager] Inicializado")

    async def handle_signal(self, signal: Signal) -> ExecutionReport:
        """
        Procesa una señal: Valida riesgo → Filtra con Trend AI → Ejecuta → Genera report.
        """
        try:
            logger.info(f"[OrderManager] Iniciando flujo para señal {signal.signal_id}: {signal.side} {signal.qty} {signal.symbol} @ {signal.price}")

            # 1. Validación Hard-coded (Risk Guard)
            if not validate_order(
                symbol=signal.symbol,
                side=signal.side,
                amount=signal.qty,
                price=signal.price,
                stop_loss=signal.stop_loss  # Asumiendo que Signal tiene stop_loss
            ):
                logger.warning(f"[OrderManager] Señal {signal.signal_id} rechazada por validación de riesgo")
                return ExecutionReport(
                    client_order_id=signal.signal_id,
                    status="rejected",
                    error_code="RISK_VALIDATION_FAILED",
                    error_msg="No pasó hard-coded safety layer"
                )

            # 2. Filtro Trend AI
            if not filter_signal(signal.__dict__):  # Convertir a dict para trend_ai
                logger.warning(f"[OrderManager] Señal {signal.signal_id} bloqueada por Trend AI")
                return ExecutionReport(
                    client_order_id=signal.signal_id,
                    status="blocked",
                    error_code="TREND_AI_BLOCK",
                    error_msg="No superó umbral de tendencia"
                )

            # 3. Ejecución (simulada o real)
            exec_result = await execute_order(
                symbol=signal.symbol,
                side=signal.side,
                amount=signal.qty,
                price=signal.price,
                order_type=signal.order_type
            )
            logger.info(f"[OrderManager] Ejecución completada para {signal.signal_id}: status={exec_result['status']}")

            # Crear report basado en resultado
            report = ExecutionReport(
                client_order_id=exec_result["client_order_id"],
                status=exec_result["status"],
                filled_qty=exec_result.get("filled", 0),
                avg_price=exec_result.get("avg_price", signal.price or 100),
                fees=exec_result.get("fees", 0),
                slippage_bps=exec_result.get("slippage_bps", 0),
                latency_ms=exec_result.get("latency_ms", 5),
                error_code=exec_result.get("error_code"),
                error_msg=exec_result.get("message")
            )

            # Guardar en órdenes activas
            self.active_orders[report.client_order_id] = report

            # Publicar el reporte en el bus
            if bus_adapter:
                await bus_adapter.publish_report(report)
                logger.info(f"[OrderManager] Reporte publicado en bus para orden {report.client_order_id}")
            else:
                logger.warning("[OrderManager] bus_adapter no inicializado, no se publica reporte")

            return report

        except Exception as e:
            logger.error(f"[OrderManager] Error en flujo de señal {signal.signal_id}: {e}")
            return ExecutionReport(
                client_order_id=signal.signal_id,
                status="error",
                error_code="EXCEPTION",
                error_msg=str(e)
            )

    def get_active_orders_count(self) -> int:
        count = len(self.active_orders)
        logger.info(f"[OrderManager] Active orders count = {count}")
        return count


# Instancia global
order_manager = OrderManager()