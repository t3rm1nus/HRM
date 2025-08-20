# l1_operational/order_manager.py
"""
Gestor de órdenes de L1_operational.
Orquesta el proceso completo: validación -> ejecución -> reporte.
"""

import time
import uuid
from typing import Optional
from loguru import logger
from .models import Signal, ExecutionReport, RiskAlert, OrderIntent
from .risk_guard import validate_order
from .executor import execute_order
from .bus_adapter import bus_adapter
from .config import EXECUTION_CONFIG

class OrderManager:
    """
    Gestiona el ciclo completo de una orden en L1.
    Solo ejecuta órdenes pre-validadas, sin tomar decisiones de trading.
    """
    
    def __init__(self):
        self.active_orders = {}
        self.order_counter = 0
    
    def handle_signal(self, signal: Signal) -> ExecutionReport:
        """
        Procesa una señal de trading y retorna un reporte de ejecución.
        L1 solo ejecuta, no modifica la señal.
        """
        start_time = time.time()
        order_id = self._generate_order_id()
        
        logger.info(f"Procesando señal {signal.signal_id} -> Orden {order_id}")
        
        try:
            # 1. Validación de riesgo (sin modificar la orden)
            if not self._validate_signal(signal):
                return self._create_rejection_report(
                    order_id, signal, "Orden rechazada por validación de riesgo"
                )
            
            # 2. Crear intención de orden
            order_intent = self._create_order_intent(order_id, signal)
            
            # 3. Ejecutar orden
            execution_result = self._execute_order(order_intent)
            
            # 4. Crear reporte de ejecución
            latency_ms = (time.time() - start_time) * 1000
            report = self._create_execution_report(
                order_id, signal, execution_result, latency_ms
            )
            
            # 5. Publicar reporte
            bus_adapter.publish_report(report)
            
            # 6. Registrar orden activa si es necesario
            if report.status in ["accepted", "partial_fill"]:
                self.active_orders[order_id] = {
                    "signal": signal,
                    "intent": order_intent,
                    "report": report
                }
            
            logger.info(f"Orden {order_id} procesada: {report.status}")
            return report
            
        except Exception as e:
            logger.error(f"Error procesando señal {signal.signal_id}: {e}")
            error_report = self._create_rejection_report(
                order_id, signal, f"Error interno: {str(e)}"
            )
            bus_adapter.publish_report(error_report)
            return error_report
    
    def _validate_signal(self, signal: Signal) -> bool:
        """
        Valida que la señal cumpla con todos los límites de riesgo.
        No modifica la señal, solo valida.
        """
        try:
            # Validación básica de parámetros
            if signal.qty <= 0:
                logger.warning(f"Señal {signal.signal_id}: cantidad inválida {signal.qty}")
                return False
            
            if signal.price and signal.price <= 0:
                logger.warning(f"Señal {signal.signal_id}: precio inválido {signal.price}")
                return False
            
            # Validación de riesgo usando risk_guard
            return validate_order(signal.symbol, signal.side, signal.qty, signal.price)
            
        except Exception as e:
            logger.error(f"Error en validación de señal {signal.signal_id}: {e}")
            return False
    
    def _create_order_intent(self, order_id: str, signal: Signal) -> OrderIntent:
        """
        Crea una intención de orden basada en la señal.
        No modifica la señal original.
        """
        return OrderIntent(
            client_order_id=order_id,
            symbol=signal.symbol,
            side=signal.side,
            type=signal.order_type,
            qty=signal.qty,
            price=signal.price,
            time_in_force=signal.time_in_force,
            route=EXECUTION_CONFIG["PAPER_MODE"] and "PAPER" or "LIVE"
        )
    
    def _execute_order(self, order_intent: OrderIntent) -> dict:
        """
        Ejecuta la orden en el exchange.
        Retorna el resultado de la ejecución.
        """
        try:
            result = execute_order(
                order_intent.symbol,
                order_intent.side,
                order_intent.qty,
                order_intent.price,
                order_intent.type
            )
            return result
            
        except Exception as e:
            logger.error(f"Error ejecutando orden {order_intent.client_order_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _create_execution_report(
        self, 
        order_id: str, 
        signal: Signal, 
        execution_result: dict, 
        latency_ms: float
    ) -> ExecutionReport:
        """
        Crea un reporte de ejecución basado en el resultado.
        """
        if execution_result["status"] == "success":
            # Orden exitosa
            return ExecutionReport(
                client_order_id=order_id,
                status="filled",
                filled_qty=signal.qty,
                avg_price=execution_result.get("order", {}).get("price", 0),
                fees=execution_result.get("order", {}).get("fees", 0),
                slippage_bps=0,  # En un sistema real, se calcularía
                latency_ms=latency_ms
            )
        elif execution_result["status"] == "error":
            # Orden con error
            return ExecutionReport(
                client_order_id=order_id,
                status="rejected",
                error_code="EXECUTION_ERROR",
                error_msg=execution_result.get("message", "Error desconocido"),
                latency_ms=latency_ms
            )
        else:
            # Estado desconocido
            return ExecutionReport(
                client_order_id=order_id,
                status="rejected",
                error_code="UNKNOWN_STATUS",
                error_msg=f"Estado desconocido: {execution_result['status']}",
                latency_ms=latency_ms
            )
    
    def _create_rejection_report(
        self, 
        order_id: str, 
        signal: Signal, 
        reason: str
    ) -> ExecutionReport:
        """
        Crea un reporte de rechazo.
        """
        return ExecutionReport(
            client_order_id=order_id,
            status="rejected",
            error_code="VALIDATION_ERROR",
            error_msg=reason
        )
    
    def _generate_order_id(self) -> str:
        """
        Genera un ID único para la orden.
        """
        self.order_counter += 1
        return f"L1_{int(time.time())}_{self.order_counter}_{uuid.uuid4().hex[:8]}"
    
    def get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Obtiene el estado de una orden activa.
        """
        return self.active_orders.get(order_id)
    
    def get_active_orders_count(self) -> int:
        """
        Retorna el número de órdenes activas.
        """
        return len(self.active_orders)

# Instancia global del gestor de órdenes
order_manager = OrderManager()
