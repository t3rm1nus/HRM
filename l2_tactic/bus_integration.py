# bus_adapter.py - L2 Tactical Bus Adapter (adaptado para multiasset: BTC y ETH)
# Nota: Renombrado de bus_integration.py para coincidir con la solicitud, y adaptado para manejar múltiples símbolos.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

# HRM bus
from comms.message_bus import MessageBus, Message
from comms.schemas import MessageSchema  # (si tu sistema lo usa para validación)

# L2 core
from .models import (
    TacticalSignal,
    PositionSize,
    MarketFeatures,
    RiskMetrics,
    StrategicDecision,
    L2State,
)
from .signal_generator import SignalGenerator  # Adaptado: usa SignalGenerator multiasset
from .position_sizer import PositionSizerManager
from .risk_controls import RiskControlManager, RiskAlert
from .config import L2Config

logger = logging.getLogger(__name__)


class MessageType(Enum):
    # Incoming (L3 -> L2)
    STRATEGIC_DECISION = "l3.strategic_decision"
    MARKET_REGIME_UPDATE = "l3.market_regime_update"
    PORTFOLIO_ALLOCATION = "l3.portfolio_allocation"
    RISK_BUDGET_UPDATE = "l3.risk_budget_update"

    # Outgoing (L2 -> L1)
    TACTICAL_SIGNAL = "l2.tactical_signal"
    POSITION_SIZE_RECOMMENDATION = "l2.position_size"
    RISK_ALERT = "l2.risk_alert"
    STOP_LOSS_UPDATE = "l2.stop_loss_update"

    # Bidirectional (L1 <-> L2)
    EXECUTION_REPORT = "l1.execution_report"
    POSITION_UPDATE = "l1.position_update"
    MARKET_DATA_UPDATE = "data.market_update"

    # Internal L2
    SIGNAL_GENERATED = "l2.signal_generated"
    RISK_CHECK_COMPLETED = "l2.risk_check_completed"
    SIZING_COMPLETED = "l2.sizing_completed"


@dataclass
class L2Message:
    message_type: MessageType
    timestamp: datetime
    source: str = "l2_tactical"
    correlation_id: Optional[str] = None
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def to_bus_message(self) -> Message:
        return Message(
            topic=self.message_type.value,
            payload={
                "timestamp": self.timestamp.isoformat(),
                "source": self.source,
                "correlation_id": self.correlation_id,
                "data": self.data or {},
                "metadata": self.metadata or {},
            },
        )

    @classmethod
    def from_bus_message(cls, message: Message) -> "L2Message":
        p = message.payload
        return cls(
            message_type=MessageType(message.topic),
            timestamp=datetime.fromisoformat(p["timestamp"]),
            source=p.get("source", "unknown"),
            correlation_id=p.get("correlation_id"),
            data=p.get("data", {}),
            metadata=p.get("metadata", {}),
        )


class L2BusAdapter:
    """
    Adaptador de integración del nivel L2 con el MessageBus:
      - Recibe decisiones de L3, market updates y estado de posiciones de L1
      - Genera señales tácticas, calcula sizing, evalúa riesgo y publica a L1
      - Publica alertas y mensajes de estado/telemetría
    Adaptado para multiasset: maneja múltiples símbolos en handlers y procesamiento.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        config: L2Config,
        signal_generator: Optional[SignalGenerator] = None,
        position_sizer: Optional[PositionSizerManager] = None,
        risk_manager: Optional[RiskControlManager] = None,
    ):
        self.bus = message_bus
        self.config = config

        self.signal_generator = signal_generator or SignalGenerator(config)  # Usa SignalGenerator multiasset
        self.position_sizer = position_sizer or PositionSizerManager(config)
        self.risk_manager = risk_manager or RiskControlManager(config)

        self.l2_state = L2State()
        self.pending_decisions: Dict[str, StrategicDecision] = {}
        self.active_correlations: Dict[str, str] = {}

        self.is_running = False
        self.processing_lock = asyncio.Lock()

        self.message_counts: Dict[str, int] = {}
        self.processing_times: Dict[str, List[float]] = {}

        logger.info("Initialized L2BusAdapter for multiasset")

    # ---------- lifecycle ----------

    async def start(self):
        if self.is_running:
            logger.warning("L2BusAdapter already running")
            return
        self.is_running = True
        await self._subscribe_to_topics()
        asyncio.create_task(self._heartbeat_task())
        asyncio.create_task(self._cleanup_task())
        logger.info("L2BusAdapter started successfully")

    async def stop(self):
        self.is_running = False
        logger.info("L2BusAdapter stopped")

    async def _subscribe_to_topics(self):
        await self.bus.subscribe(MessageType.STRATEGIC_DECISION.value, self._handle_strategic_decision)
        await self.bus.subscribe(MessageType.MARKET_REGIME_UPDATE.value, self._handle_regime_update)
        await self.bus.subscribe(MessageType.PORTFOLIO_ALLOCATION.value, self._handle_portfolio_allocation)
        await self.bus.subscribe(MessageType.EXECUTION_REPORT.value, self._handle_execution_report)
        await self.bus.subscribe(MessageType.POSITION_UPDATE.value, self._handle_position_update)
        await self.bus.subscribe(MessageType.MARKET_DATA_UPDATE.value, self._handle_market_data_update)
        logger.info("Subscribed to L2 topics")

    # ---------- handlers incoming (adaptados para multiasset) ----------

    async def _handle_strategic_decision(self, message: Message):
        start = datetime.utcnow()
        corr_id = self._new_correlation_id()
        try:
            l2msg = L2Message.from_bus_message(message)
            self._bump_count(l2msg.message_type)
            d = l2msg.data or {}
            decision = StrategicDecision(
                regime=d.get("regime", "neutral"),
                target_exposure=d.get("target_exposure", 0.5),
                risk_appetite=d.get("risk_appetite", "moderate"),
                preferred_assets=d.get("preferred_assets", self.config.signals.universe),  # Usa universo multiasset
                time_horizon=d.get("time_horizon", "1h"),
                metadata=d.get("metadata", {}),
            )
            logger.info(f"Strategic decision: regime={decision.regime} exposure={decision.target_exposure:.2f} assets={decision.preferred_assets}")
            self.pending_decisions[corr_id] = decision
            self.active_correlations[corr_id] = l2msg.correlation_id or corr_id
            asyncio.create_task(self._process_strategic_decision(decision, corr_id))
        except Exception as e:
            logger.error(f"Error handling strategic decision: {e}")
            await self._send_error_response(corr_id, str(e))

    async def _handle_regime_update(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            regime = l2msg.data.get("regime")
            if regime:
                self.l2_state.current_regime = regime
                logger.info(f"Market regime updated to {regime}")
                # Trigger re-generación de señales para todos los símbolos si es necesario
                market_data = await self._get_multi_market_data()  # Nuevo: Obtiene data para todos
                signals = self.signal_generator.generate_signals(market_data, {"regime": regime})
                for sig in signals:
                    mf = await self._get_market_features(sig.symbol)
                    await self._process_tactical_signal(sig, mf, l2msg.correlation_id)
        except Exception as e:
            logger.error(f"Error handling regime update: {e}")

    async def _handle_portfolio_allocation(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            allocations = l2msg.data.get("allocations", {})
            self.l2_state.portfolio_allocations = allocations
            logger.info(f"Portfolio allocations updated: {allocations}")
            # Ajustar sizing basado en nuevas allocations para múltiples assets
        except Exception as e:
            logger.error(f"Error handling portfolio allocation: {e}")

    async def _handle_execution_report(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            report = l2msg.data
            symbol = report.get("symbol")
            was_successful = report.get("status") == "filled"
            self.signal_generator.update_signal_performance(symbol, "execution", was_successful)
            logger.info(f"Execution report for {symbol}: {report.get('status')}")
        except Exception as e:
            logger.error(f"Error handling execution report: {e}")

    async def _handle_position_update(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            position = l2msg.data
            symbol = position.get("symbol")
            self.l2_state.active_positions[symbol] = position
            logger.info(f"Position update for {symbol}: {position.get('size')}")
            # Re-evaluar riesgo para el asset
            self.risk_manager.evaluate_position_risk(symbol, position)
        except Exception as e:
            logger.error(f"Error handling position update: {e}")

    async def _handle_market_data_update(self, message: Message):
        try:
            l2msg = L2Message.from_bus_message(message)
            data = l2msg.data
            symbol = data.get("symbol")
            if symbol in self.config.signals.universe:
                self.l2_state.market_data[symbol] = data  # Almacena por símbolo
                logger.info(f"Market data update for {symbol}")
                # Trigger generación de señales si hay decisión pendiente
                for corr_id, decision in list(self.pending_decisions.items()):
                    if symbol in decision.preferred_assets:
                        await self._generate_signals_for_symbol(symbol, decision, corr_id)
        except Exception as e:
            logger.error(f"Error handling market data update: {e}")

    # ---------- processing (adaptado para multiasset) ----------

    async def _process_strategic_decision(self, decision: StrategicDecision, correlation_id: str):
        async with self.processing_lock:
            try:
                market_data = await self._get_multi_market_data()  # Nuevo: Data para todos los assets
                signals = self.signal_generator.generate_signals(market_data, asdict(decision))
                if not signals:
                    logger.info("No signals generated from strategic decision")
                    return

                for sig in signals:
                    mf = await self._get_market_features(sig.symbol)
                    await self._process_tactical_signal(sig, mf, correlation_id)

                await self._send_processing_complete(correlation_id)
            except Exception as e:
                logger.error(f"Error processing strategic decision: {e}")
                await self._send_error_response(correlation_id, str(e))
            finally:
                self.pending_decisions.pop(correlation_id, None)

    async def _generate_signals_for_symbol(self, symbol: str, decision: StrategicDecision, correlation_id: str):
        try:
            market_data = self.l2_state.market_data.get(symbol)  # Usa data almacenada por símbolo
            if not market_data:
                logger.warning(f"No market data available for {symbol}")
                return

            # Convertir a DataFrame si es necesario (asumiendo que market_data es dict o similar)
            mf = pd.DataFrame(market_data)
            signals = self.signal_generator.generate_signals({symbol: mf}, asdict(decision))
            if not signals:
                logger.info(f"No signals generated for {symbol}")
                return

            for sig in signals:
                mf_features = await self._get_market_features(symbol)
                await self._process_tactical_signal(sig, mf_features, correlation_id)
        except Exception:
            logger.exception(f"Error generating signal for {symbol}")

    async def _process_tactical_signal(self, signal: TacticalSignal, mf: MarketFeatures, correlation_id: str):
        try:
            ps = await self._calculate_position_size(signal, mf)
            if not ps:
                logger.info(f"Sizing rejected for {signal.symbol}")
                return

            portfolio_state = self._get_current_portfolio_state()
            allow, alerts, adjusted = self.risk_manager.evaluate_pre_trade_risk(
                signal=signal,
                position_size=ps,
                market_features=mf,
                portfolio_state=portfolio_state,
            )

            for a in alerts:
                await self._send_risk_alert(a, correlation_id)

            if not allow or not adjusted:
                logger.warning(f"Trade blocked by risk controls for {signal.symbol}")
                return

            # registrar seguimiento de riesgo
            self.risk_manager.add_position(signal, adjusted, mf)
            self.l2_state.active_signals[signal.symbol] = signal

            # enviar a L1
            await self._send_tactical_signal(signal, adjusted, correlation_id)
        except Exception:
            logger.exception(f"Error processing tactical signal for {signal.symbol}")

    # ---------- helpers de negocio (adaptados) ----------

    async def _calculate_position_size(self, signal: TacticalSignal, mf: MarketFeatures) -> Optional[PositionSize]:
        ps = await self.position_sizer.calculate_position_size(
            signal=signal,
            market_features=mf,
            portfolio_state=self._get_current_portfolio_state(),
        )
        if ps:
            await self._send_intermediate(MessageType.SIZING_COMPLETED, {"symbol": signal.symbol, "position_size": asdict(ps)})
        return ps

    async def _get_market_features(self, symbol: str) -> Optional[MarketFeatures]:
        # Aquí deberías integrar con tu data layer. Fallback simple, adaptado por símbolo:
        if symbol == "BTC/USDT":
            return MarketFeatures(volatility=0.25, volume_ratio=1.2, price_momentum=0.05, rsi=55.0, macd_signal="bullish")
        elif symbol == "ETH/USDT":
            return MarketFeatures(volatility=0.30, volume_ratio=1.5, price_momentum=0.07, rsi=60.0, macd_signal="bullish")
        return None

    async def _get_multi_market_data(self) -> Dict[str, pd.DataFrame]:
        # Integrar con data layer para obtener data para todos los símbolos; fallback simple:
        return {
            "BTC/USDT": pd.DataFrame({"close": [50000.0]}),  # Ejemplo mínimo
            "ETH/USDT": pd.DataFrame({"close": [3000.0]}),
        }

    def _get_current_portfolio_state(self) -> Dict:
        # Integrar con PortfolioManager; fallback simple, con exposición por asset:
        return {
            "total_capital": 100000.0,
            "available_capital": 80000.0,
            "daily_pnl": 150.0,
            "portfolio_heat": 0.3,
            "exposures": {"BTC/USDT": 0.18, "ETH/USDT": 0.12},
        }

    # ---------- publishing (sin cambios mayores) ----------

    async def _send_tactical_signal(self, signal: TacticalSignal, ps: PositionSize, correlation_id: str):
        logger.info(f"Publishing tactical signal {signal.symbol}")
        msg = L2Message(
            message_type=MessageType.TACTICAL_SIGNAL,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            data={"signal": signal.asdict(), "position_size": asdict(ps)},
        )
        await self.bus.publish(msg.to_bus_message())

    async def _send_risk_alert(self, alert: RiskAlert, correlation_id: str):
        logger.info(f"Publishing risk alert {alert.symbol} ({alert.alert_type.value})")
        msg = L2Message(
            message_type=MessageType.RISK_ALERT,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            data=asdict(alert),
        )
        await self.bus.publish(msg.to_bus_message())

    async def _send_processing_complete(self, correlation_id: str):
        msg = L2Message(
            message_type=MessageType.SIGNAL_GENERATED,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            data={"status": "completed"},
            metadata={"processing_time": datetime.utcnow().isoformat()},
        )
        await self.bus.publish(msg.to_bus_message())

    async def _send_error_response(self, correlation_id: str, error_message: str):
        msg = L2Message(
            message_type=MessageType.SIGNAL_GENERATED,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            data={"status": "error", "error_message": error_message},
            metadata={"error_time": datetime.utcnow().isoformat()},
        )
        await self.bus.publish(msg.to_bus_message())

    async def _send_intermediate(self, mtype: MessageType, data: Dict[str, Any]):
        msg = L2Message(message_type=mtype, timestamp=datetime.utcnow(), data=data)
        await self.bus.publish(msg.to_bus_message())

    # ---------- background ----------

    async def _heartbeat_task(self):
        while self.is_running:
            try:
                hb = L2Message(
                    message_type=MessageType.SIGNAL_GENERATED,
                    timestamp=datetime.utcnow(),
                    data={
                        "status": "healthy",
                        "active_signals": len(self.l2_state.active_signals),
                        "pending_decisions": len(self.pending_decisions),
                        "message_counts": self.message_counts.copy(),
                    },
                    metadata={"heartbeat": True},
                )
                await self.bus.publish(hb.to_bus_message())
            except Exception:
                logger.exception("Error in heartbeat")
            await asyncio.sleep(30)

    async def _cleanup_task(self):
        while self.is_running:
            try:
                # lugar para limpiar caches/estados si fuera necesario
                pass
            except Exception:
                logger.exception("Error in cleanup")
            await asyncio.sleep(60)

    # ---------- metrics ----------

    def _new_correlation_id(self) -> str:
        return f"l2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    def _bump_count(self, mt: MessageType):
        k = mt.value
        self.message_counts[k] = self.message_counts.get(k, 0) + 1

    def _bump_time(self, op: str, t: float):
        self.processing_times.setdefault(op, []).append(t)
        if len(self.processing_times[op]) > 100:
            self.processing_times[op] = self.processing_times[op][-100:]