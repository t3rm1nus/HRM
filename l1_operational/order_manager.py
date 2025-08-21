# l1_operational/order_manager.py
"""Order Manager principal de L1"""

import logging
import time
from typing import Optional
from .models import Signal, ExecutionReport, OrderIntent
from .risk_guard import RiskGuard
from .executor import Executor

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Orquestador principal de L1
    Coordina: Risk Guard -> Executor -> Reports
    """
    
    def __init__(self):
        self.risk_guard = RiskGuard()
        self.executor = Executor()
        self.processed_signals = {}  # signal_id -> ExecutionReport
        
    async def handle_signal(self, signal: Signal) -> ExecutionReport:
        """
        Flujo principal de L1:
        1. Validación de riesgo hard-coded
        2. Ejecución determinista 
        3. Reporte detallado
        """
        
        logger.info(f"L1 processing signal: {signal.signal_id}")
        start_time = time.time()
        
        # PASO 1: Validación de riesgo (NUNCA se omite)
        validation_result = self.risk_guard.validate_signal(signal)
        
        if not validation_result.is_valid:
            logger.warning(f"Signal {signal.signal_id} rejected by risk validation: {validation_result.reason}")
            
            report = ExecutionReport(
                signal_id=signal.signal_id,
                status="REJECTED_SAFETY",
                reason=validation_result.reason,
                timestamp=time.time()
            )
            
            self.processed_signals[signal.signal_id] = report
            return report
        
        # PASO 2: Ejecución determinista
        try:
            execution_result = await self.executor.execute_order(signal)
            
            # Actualizar posiciones en risk guard
            qty_change = execution_result.filled_qty if signal.side == 'buy' else -execution_result.filled_qty
            self.risk_guard.update_position(signal.symbol, qty_change)
            
            # PASO 3: Crear reporte de éxito
            report = ExecutionReport(
                signal_id=signal.signal_id,
                status="EXECUTED",
                executed_qty=execution_result.filled_qty,
                executed_price=execution_result.avg_price,
                fees=execution_result.fees,
                latency_ms=execution_result.latency_ms,
                timestamp=time.time()
            )
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Signal {signal.signal_id} executed successfully in {processing_time:.2f}ms total")
            
        except Exception as e:
            logger.error(f"Execution failed for signal {signal.signal_id}: {e}")
            
            report = ExecutionReport(
                signal_id=signal.signal_id,
                status="EXECUTION_ERROR",
                reason=str(e),
                timestamp=time.time()
            )
        
        # Guardar reporte
        self.processed_signals[signal.signal_id] = report
        return report
    
    def get_execution_report(self, signal_id: str) -> Optional[ExecutionReport]:
        """Obtiene reporte de ejecución por signal_id"""
        return self.processed_signals.get(signal_id)
    
    def get_metrics(self) -> dict:
        """Obtiene métricas consolidadas de L1"""
        executor_metrics = self.executor.get_metrics()
        
        total_processed = len(self.processed_signals)
        executed = len([r for r in self.processed_signals.values() if r.status == "EXECUTED"])
        rejected_safety = len([r for r in self.processed_signals.values() if r.status == "REJECTED_SAFETY"])
        errors = len([r for r in self.processed_signals.values() if r.status == "EXECUTION_ERROR"])
        
        return {
            'total_signals_processed': total_processed,
            'executed': executed,
            'rejected_safety': rejected_safety,
            'execution_errors': errors,
            'success_rate': (executed / total_processed) if total_processed > 0 else 0.0,
            'executor_metrics': executor_metrics,
            'current_positions': self.risk_guard.current_positions.copy(),
            'daily_pnl': self.risk_guard.daily_pnl,
            'account_balance': self.risk_guard.account_balance
        }

# Crear instancia global para uso en __init__.py
order_manager = OrderManager()