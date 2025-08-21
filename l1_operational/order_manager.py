import asyncio
from core.logging import setup_logger
from l1_operational.models import Signal, ExecutionReport
from l1_operational.bus_adapter import bus_adapter

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
        Procesa una señal y genera un ExecutionReport simulado.
        """
        try:
            logger.info(f"[OrderManager] Procesando señal {signal.signal_id} "
                        f"→ {signal.side} {signal.qty} {signal.symbol} @ {signal.price}")

            # Aquí se podría validar la orden o enviarla a un exchange real
            client_order_id = signal.signal_id
            report = ExecutionReport(
                client_order_id=client_order_id,
                status="filled",  # Simulación
                filled_qty=signal.qty,
                avg_price=signal.price or 100,  # fallback
                fees=0,
                latency_ms=5
            )

            # Guardar en órdenes activas
            self.active_orders[client_order_id] = report
            logger.info(f"[OrderManager] Orden {client_order_id} completada. "
                        f"filled_qty={report.filled_qty}, avg_price={report.avg_price}")

            # Publicar el reporte en el bus
            if bus_adapter:
                await bus_adapter.publish_report(report)
                logger.info(f"[OrderManager] Reporte publicado en bus para orden {client_order_id}")
            else:
                logger.warning("[OrderManager] bus_adapter no inicializado, no se publica reporte")

            return report

        except Exception as e:
            logger.error(f"[OrderManager] Error procesando señal {signal.signal_id}: {e}")
            raise

    def get_active_orders_count(self) -> int:
        count = len(self.active_orders)
        logger.info(f"[OrderManager] Active orders count = {count}")
        return count


# Instancia global
order_manager = OrderManager()
