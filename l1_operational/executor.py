# l1_operational/executor.py
"""Ejecutor de órdenes para L1"""

import asyncio
import logging
import time
from typing import Optional
from .models import Signal, ExecutionResult, OrderIntent
from .config import EXECUTION_CONFIG, OPERATION_MODE

logger = logging.getLogger(__name__)

class Executor:
    """Ejecutor de órdenes con manejo de timeouts y retries"""
    
    def __init__(self):
        self.order_counter = 0
        self.execution_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'avg_latency_ms': 0.0
        }
        
    async def execute_order(self, signal: Signal) -> ExecutionResult:
        """
        Ejecuta una orden en el exchange de forma determinista
        1 intento por señal, con timeout y retry configurables
        """
        self.order_counter += 1
        order_id = f"L1_ORDER_{self.order_counter}_{int(time.time())}"
        
        logger.info(f"Executing order {order_id} for signal {signal.signal_id}")
        
        start_time = time.time()
        
        # Crear intent de orden
        order_intent = OrderIntent(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side,
            qty=signal.qty,
            order_type=signal.order_type,
            price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        # Ejecutar con retries
        for attempt in range(EXECUTION_CONFIG["MAX_RETRIES"]):
            try:
                result = await self._execute_with_exchange(order_id, order_intent)
                
                latency_ms = (time.time() - start_time) * 1000
                result.latency_ms = latency_ms
                
                # Actualizar métricas
                self._update_metrics(True, latency_ms)
                
                logger.info(f"Order {order_id} executed successfully in {latency_ms:.2f}ms")
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Order {order_id} timeout on attempt {attempt + 1}")
                if attempt < EXECUTION_CONFIG["MAX_RETRIES"] - 1:
                    await asyncio.sleep(EXECUTION_CONFIG["RETRY_DELAY_SECONDS"])
                    continue
                else:
                    raise
                    
            except Exception as e:
                logger.error(f"Order {order_id} execution error on attempt {attempt + 1}: {e}")
                if attempt < EXECUTION_CONFIG["MAX_RETRIES"] - 1:
                    await asyncio.sleep(EXECUTION_CONFIG["RETRY_DELAY_SECONDS"])
                    continue
                else:
                    raise
        
        # Si llegamos aquí, todos los intentos fallaron
        self._update_metrics(False, (time.time() - start_time) * 1000)
        raise Exception(f"Order {order_id} failed after {EXECUTION_CONFIG['MAX_RETRIES']} attempts")
    
    async def _execute_with_exchange(self, order_id: str, order_intent: OrderIntent) -> ExecutionResult:
        """
        Ejecuta la orden en el exchange (o simulación)
        Incluye timeout y manejo de errores específicos del exchange
        """
        
        if OPERATION_MODE == "PAPER":
            # Modo simulación
            return await self._simulate_execution(order_id, order_intent)
        
        elif OPERATION_MODE == "LIVE":
            # Modo real (requiere implementación del cliente del exchange)
            return await self._live_execution(order_id, order_intent)
        
        else:
            raise ValueError(f"Unknown operation mode: {OPERATION_MODE}")
    
    async def _simulate_execution(self, order_id: str, order_intent: OrderIntent) -> ExecutionResult:
        """Simulación de ejecución para testing"""
        
        # Simular latencia del exchange
        await asyncio.sleep(0.05)  # 50ms simulado
        
        # Simular precio de ejecución con pequeño slippage
        if order_intent.order_type == "market":
            # Market order: simular slippage de 0.1%
            base_price = order_intent.price or 50000  # Mock price
            slippage_factor = 1.001 if order_intent.side == 'buy' else 0.999
            execution_price = base_price * slippage_factor
        else:
            # Limit order: ejecutar al precio límite
            execution_price = order_intent.price
        
        # Simular fees (0.1%)
        fees = order_intent.qty * execution_price * 0.001
        
        return ExecutionResult(
            order_id=order_id,
            filled_qty=order_intent.qty,
            avg_price=execution_price,
            fees=fees,
            latency_ms=0.0,  # Se calculará en el caller
            status="FILLED"
        )
    
    async def _live_execution(self, order_id: str, order_intent: OrderIntent) -> ExecutionResult:
        """Ejecución real en exchange (placeholder)"""
        # TODO: Implementar cliente real del exchange
        raise NotImplementedError("Live execution not implemented yet")
    
    def _update_metrics(self, success: bool, latency_ms: float):
        """Actualiza métricas de ejecución"""
        self.execution_metrics['total_orders'] += 1
        
        if success:
            self.execution_metrics['successful_orders'] += 1
        else:
            self.execution_metrics['failed_orders'] += 1
        
        # Actualizar latencia promedio (rolling average simple)
        current_avg = self.execution_metrics['avg_latency_ms']
        total = self.execution_metrics['total_orders']
        self.execution_metrics['avg_latency_ms'] = (current_avg * (total - 1) + latency_ms) / total
        
        # Warning si latencia es alta
        if latency_ms > EXECUTION_CONFIG["LATENCY_WARNING_MS"]:
            logger.warning(f"High latency detected: {latency_ms:.2f}ms")
    
    def get_metrics(self) -> dict:
        """Retorna métricas de ejecución"""
        return self.execution_metrics.copy()
