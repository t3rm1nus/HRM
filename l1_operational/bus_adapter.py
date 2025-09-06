"""
Interfaz asíncrona para enviar/recibir mensajes de L2/L3.
Implementa publicación y suscripción usando tópicos explícitos.
Compatible con modelos: Signal, ExecutionReport, RiskAlert
Ahora integrado con datos REALES de Binance.
"""

import asyncio
import pandas as pd
from loguru import logger
from typing import Optional, Any, Callable, Type, Union, Dict
from dataclasses import asdict
from .models import Signal, ExecutionReport, RiskAlert
from comms.message_bus import MessageBus  # assuming comms/message_bus.py


class BusAdapterAsync:
    def __init__(self, config: dict, state: dict = None, timeout: float = 5.0):
        """
        state: diccionario con estado global, necesario para RealTimeDataLoader
        """
        from core.state_manager import validate_state_structure
        from l2_tactic.models import L2State

        # Validación de state
        self.state = validate_state_structure(state) if state else {}
        if not isinstance(self.state.get("l2"), L2State):
            logger.warning("[BusAdapterAsync] ⚠️ state['l2'] no es L2State, corrigiendo...")
            self.state["l2"] = L2State()
        
        logger.debug(f"[BusAdapterAsync] state['l2'] tipo: {type(self.state.get('l2'))}")
        
        self.bus = MessageBus()
        self.timeout = timeout
        self._running = True

        # Colas internas
        self.queue_signals: asyncio.Queue = asyncio.Queue()
        self.queue_reports: asyncio.Queue = asyncio.Queue()
        self.queue_alerts: asyncio.Queue = asyncio.Queue()

        # Colas de pendientes
        self._pending_reports = asyncio.Queue()
        self._pending_alerts = asyncio.Queue()

        # Data loader para datos reales
        self.data_loader = None
        self._initialize_data_loader()

        logger.info("[BusAdapterAsync] Inicializado con datos REALES (pendiente de start())")

    def _initialize_data_loader(self):
        """Inicializa el data loader para datos reales."""
        if not self.state:
            logger.warning("[BusAdapterAsync] ⚠️ No se proporcionó state, DataLoader no se inicializará")
            return

        try:
            # Intento flexible de importación
            try:
                from data.loaders.realtime_loader import RealTimeDataLoader
            except ImportError:
                from l1_operational.realtime_loader import RealTimeDataLoader

            self.data_loader = RealTimeDataLoader(self.state)
            logger.info("[BusAdapterAsync] ✅ DataLoader para datos REALES inicializado correctamente")
        except ImportError as e:
            logger.warning(f"[BusAdapterAsync] ⚠️ No se encontró RealTimeDataLoader: {e}")
            self.data_loader = None
        except Exception as e:
            logger.error(f"[BusAdapterAsync] ❌ Error inicializando DataLoader: {e}")
            self.data_loader = None

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
            logger.info(f"[BusAdapterAsync] Report agregado a pendientes: {report.execution_id}")
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
            logger.info(f"[BusAdapterAsync] {label} recibida: {getattr(instance, 'signal_id', getattr(instance, 'execution_id', getattr(instance, 'alert_id', 'unknown')))}")
            return instance
        except asyncio.TimeoutError:
            logger.debug(f"[BusAdapterAsync] Timeout al recibir {label.lower()}")
            return None
        except Exception as e:
            logger.error(f"[BusAdapterAsync] Error procesando {label.lower()}: {e}")
            return None

    # ----------------- PUBLICACIÓN -----------------
    async def publish_report(self, report: ExecutionReport):
        await self._publish_generic("reports", report.__dict__, f"Reporte {report.execution_id}")

    async def publish_alert(self, alert: RiskAlert):
        await self._publish_generic("alerts", alert.__dict__, f"Alerta {alert.alert_id}")

    async def publish_signal(self, signal: Signal):
        await self._publish_generic("signals", signal.__dict__, f"Señal {signal.signal_id}")

    async def _publish_generic(self, topic: str, payload: dict, label: str):
        try:
            await self.bus.publish(topic, payload)
            logger.info(f"[BusAdapterAsync] {label} publicada en {topic}")
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
                logger.error(f"[BusAdapterAsync] Error en consume genérico: {e}")

    # ----------------- DATOS REALES PARA L2/L3 -----------------
    async def handle_strategic_decision(self, message):
        """Maneja decisiones estratégicas con datos REALES."""
        try:
            from .l2_message import L2Message, StrategicDecision

            l2msg = L2Message.from_bus_message(message)
            decision = StrategicDecision(**l2msg.data)

            market_data = await self.get_real_market_data(decision.universe)
            signals = self._generate_signals_from_market_data(market_data, decision)

            for sig in signals:
                features = await self.get_real_features(sig.symbol)
                if not features.empty:
                    await self._process_signal_with_features(sig, features)
                else:
                    logger.warning(f"No hay features para {sig.symbol}, saltando señal")

        except Exception as e:
            logger.exception(f"Error handling strategic decision: {e}")

    async def get_real_market_data(self, symbols: list) -> Dict[str, pd.DataFrame]:
        market_data = {}
        if not self.data_loader:
            logger.warning("⚠️ DataLoader no disponible, usando datos simulados")
            return market_data

        for symbol in symbols:
            try:
                data = await self.data_loader.get_market_data(symbol, "1m", 100)
                if not data.empty:
                    market_data[symbol] = data
                    logger.info(f"📊 Datos REALES obtenidos para {symbol}: {len(data)} registros")
                else:
                    logger.warning(f"⚠️ No hay datos REALES para {symbol}")
            except Exception as e:
                logger.error(f"Error obteniendo datos REALES para {symbol}: {e}")
        return market_data

    async def get_real_features(self, symbol: str) -> pd.DataFrame:
        if not self.data_loader:
            logger.warning("⚠️ DataLoader no disponible, no se pueden generar features")
            return pd.DataFrame()

        try:
            features = await self.data_loader.get_features_for_symbol(symbol)
            logger.info(f"🔧 Features REALES para {symbol}: {features.shape if not features.empty else 'vacío'}")
            return features
        except Exception as e:
            logger.error(f"Error obteniendo features REALES para {symbol}: {e}")
            return pd.DataFrame()

    def _generate_signals_from_market_data(self, market_data: Dict[str, pd.DataFrame], decision) -> list:
        signals = []
        logger.info(f"Generando señales desde datos REALES para {len(market_data)} símbolos")
        return signals

    async def _process_signal_with_features(self, signal, features: pd.DataFrame):
        try:
            logger.info(f"Procesando señal {signal.signal_id} con features REALES")
            await self.publish_signal(signal)
        except Exception as e:
            logger.error(f"Error procesando señal {signal.signal_id}: {e}")

    # ----------------- MÉTODOS DE PENDIENTES -----------------
    async def get_pending_reports(self) -> list:
        items = []
        while not self._pending_reports.empty():
            try:
                report = await self._pending_reports.get()
                items.append(report)
                logger.debug(f"[BusAdapterAsync] Pending report obtenido: {report.execution_id}")
            except Exception as e:
                logger.error(f"Error obteniendo pending report: {e}")
        return items

    async def get_pending_alerts(self) -> list:
        items = []
        while not self._pending_alerts.empty():
            try:
                alert = await self._pending_alerts.get()
                items.append(alert)
                logger.debug(f"[BusAdapterAsync] Pending alert obtenido: {alert.alert_id}")
            except Exception as e:
                logger.error(f"Error obteniendo pending alert: {e}")
        return items

    # ----------------- CONTROL -----------------
    def stop(self):
        self._running = False
        logger.info("[BusAdapterAsync] Adapter detenido correctamente")

    async def cleanup(self):
        self.stop()
        if self.data_loader and hasattr(self.data_loader, 'cleanup'):
            await self.data_loader.cleanup()
        logger.info("[BusAdapterAsync] Cleanup completado")

    async def close(self):
            """
            Cierra conexiones abiertas.
            """
            try:
                if hasattr(self, 'data_feed') and self.data_feed is not None:
                    await self.data_feed.close()
                    logger.info("[BusAdapterAsync] Conexiones cerradas")
                else:
                    logger.warning("[BusAdapterAsync] ⚠️ No hay data_feed para cerrar")
            except Exception as e:
                logger.error(f"[BusAdapterAsync] ❌ Error cerrando conexiones: {e}", exc_info=True)