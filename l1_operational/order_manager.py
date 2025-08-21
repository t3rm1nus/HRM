# l1_operational/order_manager.py
"""
Order Manager de L1_operational.
Orquesta el flujo completo: validación hard-coded -> IA (Trend, Risk, Execution) -> ejecución -> reporte.
Consume señales desde L2/L3 y publica reportes y alertas automáticamente.
"""

import time
import uuid
import asyncio
from typing import Optional
from loguru import logger

from .models import Signal, ExecutionReport, RiskAlert, OrderIntent
from .risk_guard import validate_order
from .executor import execute_order
from .bus_adapter import BusAdapterAsync, bus_adapter

# Bus global (asíncrono) importado desde bus_adapter.py


class OrderManager:
    """
    Gestiona el ciclo completo de una orden en L1.
    Combina validaciones hard-coded + IA (Trend, Risk, Execution) + ejecución determinista.
    """

    def __init__(self):
        self.active_orders = {}
        self.order_counter = 0
        self._running = True

    async def handle_signal(self, signal: Signal) -> ExecutionReport:
        """
        Procesa una señal y retorna el ExecutionReport final.
        """
        start_time = time.time()
        order_id = self._generate_order_id()
        logger.info(f"[OrderManager] Procesando señal {signal.signal_id} -> Orden {order_id}")

        try:
            # 1️⃣ Validación hard-coded (determinista)
            if not self._validate_signal(signal):
                return await self._reject(order_id, signal, "Rechazada por validación hard-coded")

            # 2️⃣ Plan determinista: 1 intento de orden según señal
            intent = OrderIntent(
                client_order_id=order_id,
                symbol=signal.symbol,
                side=signal.side,
                qty=signal.qty,
                type=signal.order_type,
                price=signal.price,
                stop_loss=signal.stop_loss,
            )

            exec_result = await self._execute_order(intent)
            latency_ms = (time.time() - start_time) * 1000

            report = self._create_execution_report(order_id, signal, exec_result, latency_ms)
            await bus_adapter.publish_report(report)

            if report.status in ["filled", "partial_fill", "accepted"]:
                self.active_orders[order_id] = {
                    "signal": signal,
                    "intent": intent,
                    "report": report,
                }

            logger.info(f"[OrderManager] Orden {order_id} procesada con estado {report.status}")
            return report

        except Exception as e:
            logger.error(f"[OrderManager] Error procesando señal {signal.signal_id}: {e}")
            return await self._reject(order_id, signal, f"Error interno: {str(e)}")

    async def run(self):
        """
        Loop principal: consume señales de L2/L3 de forma asíncrona.
        """
        if not bus_adapter:
            raise RuntimeError("BusAdapterAsync no inicializado")

        logger.info("[OrderManager] Loop principal iniciado, escuchando señales...")

        while self._running:
            signal = await bus_adapter.consume_signal()
            if signal:
                asyncio.create_task(self.handle_signal(signal))  # Procesar en paralelo

    def stop(self):
        """Detiene el loop principal."""
        self._running = False
        logger.info("[OrderManager] Loop detenido")

    # 🔹 Métodos auxiliares internos

    def _validate_signal(self, signal: Signal) -> bool:
        """Validación hard-coded de riesgo y parámetros básicos."""
        try:
            if signal.qty <= 0 or (signal.price and signal.price <= 0):
                return False
            return validate_order(
                symbol=signal.symbol,
                side=signal.side,
                amount=signal.qty,
                price=signal.price,
                stop_loss=signal.stop_loss,
            )
        except Exception as e:
            logger.error(f"[OrderManager] Error validando señal {signal.signal_id}: {e}")
            return False

    async def _execute_order(self, order_intent: OrderIntent) -> dict:
        """Ejecuta la orden en el exchange o simulador."""
        try:
            return await execute_order(
                symbol=order_intent.symbol,
                side=order_intent.side,
                amount=order_intent.qty,
                price=order_intent.price,
                order_type=order_intent.type,
            )
        except Exception as e:
            logger.error(f"[OrderManager] Error ejecutando {order_intent.client_order_id}: {e}")
            return {"status": "error", "message": str(e)}

    def _create_execution_report(self, order_id, signal, execution_result, latency_ms):
        """Genera un ExecutionReport normalizado."""
        if execution_result.get("status") == "success":
            return ExecutionReport(
                client_order_id=order_id,
                status="filled",
                filled_qty=execution_result.get("filled", signal.qty),
                avg_price=float(execution_result.get("order", {}).get("price", 0)) if execution_result.get("order") else None,
                fees=float(execution_result.get("order", {}).get("fees", 0)) if execution_result.get("order") else 0.0,
                slippage_bps=0,
                latency_ms=latency_ms,
            )
        else:
            return ExecutionReport(
                client_order_id=order_id,
                status="rejected" if execution_result.get("status") == "rejected" else "failed",
                message=execution_result.get("message", "Error desconocido"),
                latency_ms=latency_ms,
            )

    async def _reject(self, order_id, signal, reason) -> ExecutionReport:
        """Genera un reporte de rechazo estándar."""
        report = ExecutionReport(
            client_order_id=order_id,
            status="rejected",
            message=reason,
        )
        await bus_adapter.publish_report(report)
        return report

    def _generate_order_id(self) -> str:
        self.order_counter += 1
        return f"L1_{int(time.time())}_{self.order_counter}_{uuid.uuid4().hex[:8]}"

    # 🔹 Utilidades
    def get_order_status(self, order_id: str) -> Optional[dict]:
        return self.active_orders.get(order_id)

    def get_active_orders_count(self) -> int:
        return len(self.active_orders)


# Instancia global
order_manager = OrderManager()
