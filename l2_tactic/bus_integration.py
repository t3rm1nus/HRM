# l2_tactic/bus_integration.py
# Adaptador de integración de L2 con el MessageBus:
# - Recibe inputs de L3 (decisiones, régimen, allocation)
# - Recibe datos de mercado
# - Genera señales tácticas + sizing + controles de riesgo
# - Publica a L1 (tactical_signal, position_size, risk_alert)

from __future__ import annotations

import asyncio
import logging
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from comms.message_bus import MessageBus, Message
from .models import TacticalSignal, PositionSize, MarketFeatures, RiskMetrics, StrategicDecision, L2State
from .signal_generator import SignalGenerator
from .position_sizer import PositionSizerManager
from .risk_controls import RiskControlManager, RiskAlert
from .config import L2Config

logger = logging.getLogger(__name__)


class MessageType(Enum):
    # Incoming (L3 -> L2)
    STRATEGIC_DECISION = "l3.strategic_decision"
    MARKET_REGIME_UPDATE = "l3.market_regime_update"
    PORTFOLIO_ALLOCATION = "l3.portfolio_allocation"

    # Outgoing (L2 -> L1)
    TACTICAL_SIGNAL = "l2.tactical_signal"
    POSITION_SIZE = "l2.position_size"
    RISK_ALERT = "l2.risk_alert"

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
    def __init__(self, bus: MessageBus, config: L2Config):
        self.bus = bus
        self.config = config

        self.signal_generator = SignalGenerator(config)
        self.position_sizer = PositionSizerManager(config)
        self.risk_manager = RiskControlManager(config)

        self.l2_state = L2State()
        self.is_running = False

    # ---------- lifecycle ----------
    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        await self._subscribe_topics()
        asyncio.create_task(self._heartbeat_task())
        logger.info("✅ L2BusAdapter started")

    async def _subscribe_topics(self):
        await self.bus.subscribe(MessageType.STRATEGIC_DECISION.value, self._handle_strategic_decision)
        await self.bus.subscribe(MessageType.MARKET_DATA_UPDATE.value, self._handle_market_data_update)
        await self.bus.subscribe(MessageType.EXECUTION_REPORT.value, self._handle_execution_report)

    # ---------- handlers ----------
    async def _handle_strategic_decision(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            decision = StrategicDecision(**l2msg.data)
            market_data = await self._get_market_data()
            signals = self.signal_generator.generate_signals(market_data, asdict(decision))
            for sig in signals:
                mf = await self._get_features(sig.symbol)
                await self._process_signal(sig, mf)
        except Exception:
            logger.exception("Error handling strategic decision")

    async def _handle_market_data_update(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            symbol = l2msg.data.get("symbol")
            if not symbol:
                return
            self.l2_state.market_data[symbol] = l2msg.data
        except Exception:
            logger.exception("Error handling market data update")

    async def _handle_execution_report(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            symbol = l2msg.data.get("symbol")
            status = l2msg.data.get("status")
            logger.info(f"Execution report {symbol}: {status}")
        except Exception:
            logger.exception("Error handling execution report")

    # ---------- processing ----------
    async def _process_signal(self, signal: TacticalSignal, mf: MarketFeatures):
        ps = await self.position_sizer.calculate_position_size(signal, mf, self._portfolio_state())
        if not ps:
            logger.info(f"❌ Sizing rejected for {signal.symbol}")
            return

        allow, alerts, adjusted = self.risk_manager.evaluate_pre_trade_risk(signal, ps, mf, self._portfolio_state())
        for alert in alerts:
            await self._publish(MessageType.RISK_ALERT, asdict(alert))

        if not allow or not adjusted:
            logger.warning(f"⚠️ Trade blocked by risk controls for {signal.symbol}")
            return

        await self._publish(MessageType.TACTICAL_SIGNAL, {"signal": signal.asdict(), "position_size": asdict(adjusted)})

    # ---------- helpers ----------
    async def _get_market_data(self) -> Dict[str, pd.DataFrame]:
        # Fallback: datos dummy
        return {"BTC/USDT": pd.DataFrame({"close": [50000]}), "ETH/USDT": pd.DataFrame({"close": [3000]})}

    async def _get_features(self, symbol: str) -> MarketFeatures:
        return MarketFeatures(volatility=0.2, volume_ratio=1.0, price_momentum=0.05, rsi=55, macd_signal="bullish")

    def _portfolio_state(self) -> Dict[str, Any]:
        return {"capital": 100000, "exposures": {"BTC/USDT": 0.1, "ETH/USDT": 0.05}}

    async def _publish(self, mtype: MessageType, data: Dict[str, Any]):
        msg = L2Message(message_type=mtype, timestamp=datetime.utcnow(), data=data)
        await self.bus.publish(msg.to_bus_message())

    # ---------- background ----------
    async def _heartbeat_task(self):
        while self.is_running:
            await self._publish(MessageType.HEARTBEAT, {"status": "ok", "active_signals": len(self.l2_state.active_signals)})
            await asyncio.sleep(30)
