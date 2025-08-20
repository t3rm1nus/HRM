# l1_operational/bus_adapter.py
"""
Adaptador para el bus de mensajes de L1.
Permite recibir señales desde L2/L3 y enviar reportes de ejecución.
"""

import json
import time
from typing import List, Optional
from loguru import logger
from .models import Signal, ExecutionReport, RiskAlert

class BusAdapter:
    """
    Adaptador para el bus de mensajes del sistema.
    En un sistema real, esto se conectaría con el bus definido en el README.
    """
    
    def __init__(self):
        self.signals_queue = []
        self.reports_queue = []
        self.alerts_queue = []
    
    def consume_signal(self) -> Optional[Signal]:
        """
        Consume una señal del bus de mensajes.
        Retorna None si no hay señales disponibles.
        """
        if not self.signals_queue:
            return None
        
        signal_data = self.signals_queue.pop(0)
        try:
            # En un sistema real, esto vendría del bus real
            signal = Signal(**signal_data)
            logger.info(f"Señal recibida: {signal.signal_id}")
            return signal
        except Exception as e:
            logger.error(f"Error procesando señal: {e}")
            return None
    
    def publish_report(self, report: ExecutionReport):
        """
        Publica un reporte de ejecución al bus de mensajes.
        """
        try:
            report_dict = {
                "client_order_id": report.client_order_id,
                "status": report.status,
                "filled_qty": report.filled_qty,
                "avg_price": report.avg_price,
                "fees": report.fees,
                "slippage_bps": report.slippage_bps,
                "latency_ms": report.latency_ms,
                "error_code": report.error_code,
                "error_msg": report.error_msg,
                "timestamp": report.timestamp
            }
            
            self.reports_queue.append(report_dict)
            logger.info(f"Reporte publicado: {report.client_order_id} - {report.status}")
            
        except Exception as e:
            logger.error(f"Error publicando reporte: {e}")
    
    def publish_alert(self, alert: RiskAlert):
        """
        Publica una alerta de riesgo al bus de mensajes.
        """
        try:
            alert_dict = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "symbol": alert.symbol,
                "order_id": alert.order_id,
                "timestamp": alert.timestamp
            }
            
            self.alerts_queue.append(alert_dict)
            logger.warning(f"Alerta publicada: {alert.alert_type} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Error publicando alerta: {e}")
    
    def get_pending_reports(self) -> List[dict]:
        """
        Obtiene todos los reportes pendientes de envío.
        """
        reports = self.reports_queue.copy()
        self.reports_queue.clear()
        return reports
    
    def get_pending_alerts(self) -> List[dict]:
        """
        Obtiene todas las alertas pendientes de envío.
        """
        alerts = self.alerts_queue.copy()
        self.alerts_queue.clear()
        return alerts
    
    def add_test_signal(self, signal_data: dict):
        """
        Método de prueba para agregar señales al adaptador.
        En un sistema real, esto vendría del bus real.
        """
        self.signals_queue.append(signal_data)
        logger.info(f"Señal de prueba agregada: {signal_data.get('signal_id', 'unknown')}")

# Instancia global del adaptador
bus_adapter = BusAdapter()
