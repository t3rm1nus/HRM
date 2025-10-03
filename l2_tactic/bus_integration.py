#l2_tactic/bus_integration.py 

from __future__ import annotations

import asyncio
import logging
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from comms.config import config
from comms.message_bus import MessageBus, Message
from .models import (
    TacticalSignal,
    PositionSize,
    MarketFeatures,
    RiskMetrics,
    StrategicDecision,
    L2State,
)
from .tactical_signal_processor import L2TacticProcessor as SignalGenerator  # Changed to match main.py
from .position_sizer import PositionSizerManager
from .risk_controls import RiskControlManager, RiskAlert
from .config import L2Config
from l2_tactic.metrics import L2Metrics
from core.logging import logger


class MessageType(Enum):
    # Incoming (L3 -> L2)
    STRATEGIC_DECISION = "l3.strategic_decision"
    MARKET_REGIME_UPDATE = "l3.market_regime_update"
    PORTFOLIO_ALLOCATION = "l3.portfolio_allocation"

    # Outgoing (L2 -> L1)
    TACTICAL_SIGNAL = "l2.tactical_signal"
    POSITION_SIZE = "l2.position_size"
    RISK_ALERT = "l2.risk_alert"

    # Reporting (L2 -> L4)
    PERFORMANCE_REPORT = "l2.performance_report"

    # Bidirectional (L1 <-> L2)
    EXECUTION_REPORT = "l1.execution_report"
    POSITION_UPDATE = "l1.position_update"
    MARKET_DATA_UPDATE = "data.market_update"

    # Internal L2
    HEARTBEAT = "l2.heartbeat"


@dataclass
class L2Message:
    message_type: MessageType
    timestamp: datetime
    data: Dict[str, Any]
    correlation_id: Optional[str] = None

    def to_bus_message(self) -> Message:
        return Message(
            topic=self.message_type.value,
            payload={
                "timestamp": self.timestamp.isoformat(),
                "correlation_id": self.correlation_id,
                "data": self.data,
            },
        )

    @classmethod
    def from_bus_message(cls, message: Message) -> "L2Message":
        p = message.payload
        return cls(
            message_type=MessageType(message.topic),
            timestamp=datetime.fromisoformat(p["timestamp"]),
            correlation_id=p.get("correlation_id"),
            data=p.get("data", {}),
        )


class L2BusAdapter:
    def __init__(self, bus: MessageBus, config: Dict[str, Any] = None):
        self.bus = bus
        self.config = config or L2Config(**config.get("L2_CONFIG", {}))  # Use L2Config with fallback
        self.signal_generator = SignalGenerator(self.config)  # L2TacticProcessor
        self.position_sizer = PositionSizerManager(self.config)
        self.risk_manager = RiskControlManager(self.config)
        self.l2_state = L2State()
        self.is_running = False
        self.metrics = L2Metrics()
        logger.info("✅ L2BusAdapter inicializado")

    # ---------- lifecycle ----------
    async def start(self):
        if self.is_running:
            logger.warning("⚠️ L2BusAdapter ya está ejecutándose")
            return
        self.is_running = True
        await self._subscribe_topics()
        asyncio.create_task(self._heartbeat_task())
        asyncio.create_task(self._performance_report_task())
        logger.info("✅ L2BusAdapter started")

    async def _subscribe_topics(self):
        try:
            await self.bus.subscribe(MessageType.STRATEGIC_DECISION.value, self._handle_strategic_decision)
            await self.bus.subscribe(MessageType.MARKET_DATA_UPDATE.value, self._handle_market_data_update)
            await self.bus.subscribe(MessageType.EXECUTION_REPORT.value, self._handle_execution_report)
            await self.bus.subscribe(MessageType.MARKET_REGIME_UPDATE.value, self._handle_market_regime_update)
            await self.bus.subscribe(MessageType.PORTFOLIO_ALLOCATION.value, self._handle_portfolio_allocation)
            logger.info("✅ Suscrito a todos los tópicos del bus")
        except Exception as e:
            logger.error(f"❌ Error suscribiendo tópicos: {e}", exc_info=True)

    # ---------- handlers ----------
    async def _handle_strategic_decision(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            decision = StrategicDecision(**l2msg.data)
            market_data = await self._get_market_data()
            signals = await self.signal_generator.process(market_data=market_data, technical_indicators={}, state=self.l2_state)
            for sig in signals.get("signals", []):
                mf = await self._get_features(sig.symbol)
                await self._process_signal(sig, mf)
        except Exception as e:
            logger.error("❌ Error handling strategic decision", exc_info=True)

    async def _handle_market_data_update(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            symbol = l2msg.data.get("symbol")
            if not symbol:
                logger.warning("⚠️ Mensaje de market data sin símbolo")
                return
            self.l2_state.market_data[symbol] = pd.DataFrame(l2msg.data.get("data", {}))
            logger.debug(f"📊 Market data actualizado para {symbol}")
        except Exception as e:
            logger.error("❌ Error handling market data update", exc_info=True)

    async def _handle_execution_report(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            symbol = l2msg.data.get("symbol")
            status = l2msg.data.get("status")
            logger.info(f"📊 Execution report {symbol}: {status}")
            self.metrics.record_execution(symbol=symbol, status=status)
        except Exception as e:
            logger.error("❌ Error handling execution report", exc_info=True)

    async def _handle_market_regime_update(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            self.l2_state.regime = l2msg.data.get("regime", "neutral")
            logger.info(f"📊 Régimen de mercado actualizado: {self.l2_state.regime}")
        except Exception as e:
            logger.error("❌ Error handling market regime update", exc_info=True)

    async def _handle_portfolio_allocation(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            self.l2_state.allocation = l2msg.data
            logger.info(f"📊 Allocation recibida de L3: {self.l2_state.allocation}")
        except Exception as e:
            logger.error("❌ Error handling portfolio allocation", exc_info=True)

    # ---------- processing ----------
    async def _process_signal(self, signal: TacticalSignal, mf: MarketFeatures):
        try:
            # Construir estado de portfolio usando caja real cuando esté disponible
            port_state = self._portfolio_state()
            try:
                # Si existe state global accesible con USDT/total, úsalo (este adaptador puede ejecutarse aislado)
                # Mantener compatibilidad si no existe
                # port_state keys esperadas: total_capital, available_capital
                if 'total_capital' not in port_state or 'available_capital' not in port_state:
                    port_state = {
                        'total_capital': port_state.get('capital', 0.0),
                        'available_capital': port_state.get('USDT', port_state.get('capital', 0.0)),
                    }
            except Exception:
                pass

            ps = await self.position_sizer.calculate_position_size(signal, mf, port_state)
            if not ps:
                logger.info(f"❌ Sizing rejected for {signal.symbol}")
                self.metrics.record_signal(signal.symbol, accepted=False)
                return

            allow, alerts, adjusted = self.risk_manager.evaluate_pre_trade_risk(signal, ps, mf, self._portfolio_state())
            for alert in alerts:
                await self._publish(MessageType.RISK_ALERT, asdict(alert))

            if not allow or not adjusted:
                logger.warning(f"⚠️ Trade blocked by risk controls for {signal.symbol}")
                self.metrics.record_signal(signal.symbol, accepted=False)
                return

            await self._publish(MessageType.TACTICAL_SIGNAL, {"signal": signal.asdict(), "position_size": asdict(adjusted)})
            await self._publish(MessageType.POSITION_SIZE, asdict(adjusted))
            self.metrics.record_signal(signal.symbol, accepted=True)
        except Exception as e:
            logger.error(f"❌ Error procesando señal para {signal.symbol}: {e}", exc_info=True)

    # ---------- helpers ----------
    async def _get_market_data(self) -> Dict[str, pd.DataFrame]:
        if self.l2_state.market_data:
            return self.l2_state.market_data
        return {
            "BTCUSDT": pd.DataFrame({"close": [50000]}, index=[pd.Timestamp.utcnow()]),
            "ETHUSDT": pd.DataFrame({"close": [3000]}, index=[pd.Timestamp.utcnow()])
        }

    async def _get_features(self, symbol: str) -> MarketFeatures:
        return MarketFeatures(volatility=0.2, volume_ratio=1.0, price_momentum=0.05, rsi=55, macd_signal="bullish")

    def _portfolio_state(self) -> Dict[str, Any]:
        return {"capital": 100000, "exposures": {"BTCUSDT": 0.1, "ETHUSDT": 0.05}}

    async def _publish(self, mtype: MessageType, data: Dict[str, Any]):
        try:
            msg = L2Message(message_type=mtype, timestamp=datetime.utcnow(), data=data)
            await self.bus.publish(msg.to_bus_message())
            logger.debug(f"📤 Publicado mensaje {mtype.value}")
        except Exception as e:
            logger.error(f"❌ Error publicando mensaje {mtype.value}: {e}", exc_info=True)

    async def publish_performance_report(self):
        try:
            payload = {
                "metrics": self.metrics.to_dict(),
                "ts": datetime.utcnow().isoformat(),
            }
            await self._publish(MessageType.PERFORMANCE_REPORT, payload)
            logger.info("📊 Performance report publicado a L4")
        except Exception as e:
            logger.error(f"❌ Error publicando performance report: {e}", exc_info=True)

    # ---------- background ----------
    async def _heartbeat_task(self):
        while self.is_running:
            try:
                await self._publish(
                    MessageType.HEARTBEAT,
                    {"status": "ok", "active_signals": len(self.l2_state.active_signals)}
                )
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"❌ Error en heartbeat task: {e}", exc_info=True)

    async def _performance_report_task(self):
        """Loop periódico que envía métricas de L2 hacia L4."""
        while self.is_running:
            try:
                await self.publish_performance_report()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"❌ Error en performance report task: {e}", exc_info=True)

    async def close(self):
        """Cierra el adaptador y detiene las tareas."""
        self.is_running = False
        logger.info("✅ L2BusAdapter cerrado")
