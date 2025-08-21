# l1_operational/bus_adapter_async.py
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

        # Suscribir a tópicos estándar
        self.queue_signals = bus.subscribe("signals")
        self.queue_reports = bus.subscribe("reports")
        self.queue_alerts = bus.subscribe("alerts")

    async def consume_signal(self) -> Optional[Signal]:
        """
        Consume una señal desde L2/L3.
        Retorna None si ocurre un error o timeout.
        """
        return await self._consume_generic(self.queue_signals, Signal, "Señal")

    async def consume_report(self) -> Optional[ExecutionReport]:
        """
        Consume un ExecutionReport desde L2/L3 (si aplica).
        Retorna None si ocurre un error o timeout.
        """
        return await self._consume_generic(self.queue_reports, ExecutionReport, "Reporte")

    async def consume_alert(self) -> Optional[RiskAlert]:
        """
        Consume un RiskAlert desde L2/L3 (si aplica).
        Retorna None si ocurre un error o timeout.
        """
        return await self._consume_generic(self.queue_alerts, RiskAlert, "Alerta")

    async def _consume_generic(self, queue: asyncio.Queue, cls: Type, label: str) -> Optional[Union[Signal, ExecutionReport, RiskAlert]]:
        """
        Método interno genérico para consumir mensajes de una queue y deserializar a clase específica.
        """
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

    async def publish_report(self, report: ExecutionReport):
        """Publica un ExecutionReport en el tópico 'reports'."""
        await self._publish_generic("reports", report.__dict__, f"Reporte {report.client_order_id}")

    async def publish_alert(self, alert: RiskAlert):
        """Publica un RiskAlert en el tópico 'alerts'."""
        await self._publish_generic("alerts", alert.__dict__, f"Alerta {alert.alert_id}")

    async def publish_signal(self, signal: Signal):
        """Publica un Signal en el tópico 'signals'."""
        await self._publish_generic("signals", signal.__dict__, f"Señal {signal.signal_id}")

    async def _publish_generic(self, topic: str, payload: dict, label: str):
        """Método interno genérico para publicar mensajes en cualquier tópico."""
        try:
            await self.bus.publish(topic, payload)
            logger.info(f"[BusAdapterAsync] {label} publicada en {topic}")
        except Exception as e:
            logger.error(f"[BusAdapterAsync] Error publicando {label} en {topic}: {e}")

    async def consume(self, queue: asyncio.Queue, handler: Callable[[dict], Any]):
        """
        Método genérico para consumir mensajes de cualquier queue con un handler personalizado.
        """
        while self._running:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=self.timeout)
                await handler(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[BusAdapterAsync] Error en consume genérico: {e} | msg: {msg}")

    def stop(self):
        """Detiene el consumo de mensajes de forma ordenada."""
        self._running = False
        logger.info("[BusAdapterAsync] Adapter detenido correctamente")

# Instancia por defecto para facilitar integración inmediata
try:
    default_bus = MessageBus()
    bus_adapter = BusAdapterAsync(default_bus)
except Exception:
    # Entorno de prueba puede no requerir bus
    bus_adapter = None
