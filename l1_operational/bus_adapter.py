"""
Interfaz asíncrona para enviar/recibir mensajes de L2/L3.
Implementa publicación y suscripción usando tópicos explícitos.
Compatible con modelos: Signal, ExecutionReport, RiskAlert
"""

import asyncio
from loguru import logger
from typing import Optional, Any, Callable, Type, Union
from .models import Signal, ExecutionReport, RiskAlert
from comms.message_bus import MessageBus  # assuming comms/message_bus.py


class BusAdapterAsync:
    def __init__(self, bus: MessageBus, timeout: float = 5.0):
        self.bus = bus
        self.timeout = timeout
        self._running = True

        # Colas internas
        self.queue_signals: asyncio.Queue = asyncio.Queue()
        self.queue_reports: asyncio.Queue = asyncio.Queue()
        self.queue_alerts: asyncio.Queue = asyncio.Queue()

        # Colas de pendientes
        self._pending_reports = asyncio.Queue()
        self._pending_alerts = asyncio.Queue()

        logger.info("[BusAdapterAsync] Inicializado (pendiente de start())")

    async def start(self):
        """Suscribe handlers a los tópicos del bus"""
        self.bus.subscribe("signals", self._enqueue_signal)
        self.bus.subscribe("reports", self._enqueue_report)
        self.bus.subscribe("alerts", self._enqueue_alert)
        logger.info("[BusAdapterAsync] Suscrito a signals/reports/alerts")

    # Handlers que meten en colas internas
    async def _enqueue_signal(self, message):
        await self.queue_signals.put(message)

    async def _enqueue_report(self, message):
        await self.queue_reports.put(message)

    async def _enqueue_alert(self, message):
        await self.queue_alerts.put(message)
    
    
    # ----------------- CONSUMO -----------------
    async def consume_signal(self) -> Optional[Signal]:
        signal = await self._consume_generic(self.queue_signals, Signal, "Señal")
        if signal:
            logger.info(f"[BusAdapterAsync] Signal procesada: {signal.signal_id} → {signal.side} {signal.qty} {signal.symbol}")
        return signal

    async def consume_report(self) -> Optional[ExecutionReport]:
        report = await self._consume_generic(self.queue_reports, ExecutionReport, "Reporte")
        if report:
            await self._pending_reports.put(report)
            logger.info(f"[BusAdapterAsync] Report agregado a pendientes: {report.client_order_id}")
        return report

    async def consume_alert(self) -> Optional[RiskAlert]:
        alert = await self._consume_generic(self.queue_alerts, RiskAlert, "Alerta")
        if alert:
            await self._pending_alerts.put(alert)
            logger.info(f"[BusAdapterAsync] Alert agregado a pendientes: {alert.alert_id}")
        return alert

    async def _consume_generic(self, queue: asyncio.Queue, cls: Type, label: str) -> Optional[Union[Signal, ExecutionReport, RiskAlert]]:
        if not self._running:
            return None
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=self.timeout)
            instance = cls(**msg)
            logger.info(f"[BusAdapterAsync] {label} recibida: {getattr(instance, 'signal_id', getattr(instance, 'client_order_id', getattr(instance, 'alert_id', 'unknown')))}")
            return instance
        except asyncio.TimeoutError:
            logger.warning(f"[BusAdapterAsync] Timeout al recibir {label.lower()}")
            return None
        except Exception as e:
            logger.error(f"[BusAdapterAsync] Error procesando {label.lower()}: {e} | msg: {msg}")
            return None

    # ----------------- PUBLICACIÓN -----------------
    async def publish_report(self, report: ExecutionReport):
        await self._publish_generic("reports", report.__dict__, f"Reporte {report.client_order_id}")

    async def publish_alert(self, alert: RiskAlert):
        await self._publish_generic("alerts", alert.__dict__, f"Alerta {alert.alert_id}")

    async def publish_signal(self, signal: Signal):
        await self._publish_generic("signals", signal.__dict__, f"Señal {signal.signal_id}")

    async def _publish_generic(self, topic: str, payload: dict, label: str):
        try:
            await self.bus.publish(topic, payload)
            logger.info(f"[BusAdapterAsync] {label} publicada en {topic}: {payload}")
        except Exception as e:
            logger.error(f"[BusAdapterAsync] Error publicando {label} en {topic}: {e}")

    # ----------------- CONSUMO GENÉRICO -----------------
    async def consume(self, queue: asyncio.Queue, handler: Callable[[dict], Any]):
        while self._running:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=self.timeout)
                await handler(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[BusAdapterAsync] Error en consume genérico: {e} | msg: {msg}")

    # ----------------- CONTROL -----------------
    def stop(self):
        self._running = False
        logger.info("[BusAdapterAsync] Adapter detenido correctamente")

    # ----------------- MÉTODOS DE PENDIENTES -----------------
    async def get_pending_reports(self):
        """Devuelve lista de ExecutionReports pendientes."""
        items = []
        while not self._pending_reports.empty():
            report = await self._pending_reports.get()
            logger.info(f"[BusAdapterAsync] Pending report obtenido: {report.client_order_id}")
            items.append(report)
        return items

    async def get_pending_alerts(self):
        """Devuelve lista de RiskAlerts pendientes."""
        items = []
        while not self._pending_alerts.empty():
            alert = await self._pending_alerts.get()
            logger.info(f"[BusAdapterAsync] Pending alert obtenido: {alert.alert_id}")
            items.append(alert)
        return items


# ----------------- INSTANCIA POR DEFECTO -----------------
try:
    default_bus = MessageBus()
    bus_adapter = BusAdapterAsync(default_bus)
except Exception:
    bus_adapter = None
    logger.warning("[BusAdapterAsync] No se pudo inicializar bus_adapter, entorno de prueba")
