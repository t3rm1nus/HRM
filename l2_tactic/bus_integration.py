# bus_integration.py - L2 Tactical Bus Integration

"""
bus_integration.py - L2 Tactical Bus Integration

Integración de L2 con el MessageBus del sistema HRM:
- Recepción de decisiones estratégicas de L3
- Envío de señales tácticas a L1
- Manejo de confirmaciones y reportes
- Gestión de alertas de riesgo
- Comunicación asíncrona entre niveles
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

# Imports del sistema HRM
from comms.message_bus import MessageBus, Message
from comms.schemas import MessageSchema

# Imports locales L2
from .models import (
    TacticalSignal, PositionSize, MarketFeatures, RiskMetrics,
    StrategicDecision, L2State
)
from .signal_generator import TacticalSignalGenerator
from .position_sizer import PositionSizerManager
from .risk_controls import RiskControlManager, RiskAlert
from .config import L2Config


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Tipos de mensajes L2"""
    # Incoming (de L3)
    STRATEGIC_DECISION = "l3.strategic_decision"
    MARKET_REGIME_UPDATE = "l3.market_regime_update"
    PORTFOLIO_ALLOCATION = "l3.portfolio_allocation"
    RISK_BUDGET_UPDATE = "l3.risk_budget_update"
    
    # Outgoing (a L1)
    TACTICAL_SIGNAL = "l2.tactical_signal"
    POSITION_SIZE_RECOMMENDATION = "l2.position_size"
    RISK_ALERT = "l2.risk_alert"
    STOP_LOSS_UPDATE = "l2.stop_loss_update"
    
    # Bidirectional
    EXECUTION_REPORT = "l1.execution_report"
    POSITION_UPDATE = "l1.position_update"
    MARKET_DATA_UPDATE = "data.market_update"
    
    # Internal L2
    SIGNAL_GENERATED = "l2.signal_generated"
    RISK_CHECK_COMPLETED = "l2.risk_check_completed"
    SIZING_COMPLETED = "l2.sizing_completed"


@dataclass
class L2Message:
    """Mensaje estándar para L2"""
    message_type: MessageType
    timestamp: datetime
    source: str = "l2_tactical"
    correlation_id: Optional[str] = None
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def to_bus_message(self) -> Message:
        """Convertir a formato MessageBus"""
        return Message(
            topic=self.message_type.value,
            payload={
                "timestamp": self.timestamp.isoformat(),
                "source": self.source,
                "correlation_id": self.correlation_id,
                "data": self.data or {},
                "metadata": self.metadata or {}
            }
        )
    
    @classmethod
    def from_bus_message(cls, message: Message) -> 'L2Message':
        """Crear desde Message del bus"""
        payload = message.payload
        return cls(
            message_type=MessageType(message.topic),
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            source=payload.get("source", "unknown"),
            correlation_id=payload.get("correlation_id"),
            data=payload.get("data", {}),
            metadata=payload.get("metadata", {})
        )


class L2BusAdapter:
    """Adaptador principal para comunicación con MessageBus"""
    
    def __init__(
        self,
        message_bus: MessageBus,
        config: L2Config,
        signal_generator: Optional[TacticalSignalGenerator] = None,
        position_sizer: Optional[PositionSizerManager] = None,
        risk_manager: Optional[RiskControlManager] = None
    ):
        self.bus = message_bus
        self.config = config
        
        # Core L2 components
        self.signal_generator = signal_generator or TacticalSignalGenerator(config)
        self.position_sizer = position_sizer or PositionSizerManager(config)
        self.risk_manager = risk_manager or RiskControlManager(config)
        
        # State management
        self.l2_state = L2State()
        self.pending_decisions: Dict[str, StrategicDecision] = {}
        self.active_correlations: Dict[str, str] = {}  # correlation_id -> original_message_id
        
        # Processing flags
        self.is_running = False
        self.processing_lock = asyncio.Lock()
        
        # Metrics and monitoring
        self.message_counts: Dict[str, int] = {}
        self.processing_times: Dict[str, float] = {}
        
        logger.info("Initialized L2BusAdapter")
    
    async def start(self):
        """Iniciar el adaptador y suscribirse a topics"""
        
        if self.is_running:
            logger.warning("L2BusAdapter already running")
            return
        
        self.is_running = True
        
        # Subscribe to incoming topics
        await self._subscribe_to_topics()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_task())
        asyncio.create_task(self._cleanup_task())
        
        logger.info("L2BusAdapter started successfully")
    
    async def stop(self):
        """Detener el adaptador"""
        
        self.is_running = False
        logger.info("L2BusAdapter stopped")
    
    async def _subscribe_to_topics(self):
        """Suscribirse a topics relevantes"""
        
        # Topics de L3 (incoming)
        await self.bus.subscribe(
            MessageType.STRATEGIC_DECISION.value,
            self._handle_strategic_decision
        )
        
        await self.bus.subscribe(
            MessageType.MARKET_REGIME_UPDATE.value,
            self._handle_regime_update
        )
        
        await self.bus.subscribe(
            MessageType.PORTFOLIO_ALLOCATION.value,
            self._handle_portfolio_allocation
        )
        
        # Topics de L1 (bidirectional)
        await self.bus.subscribe(
            MessageType.EXECUTION_REPORT.value,
            self._handle_execution_report
        )
        
        await self.bus.subscribe(
            MessageType.POSITION_UPDATE.value,
            self._handle_position_update
        )
        
        # Market data
        await self.bus.subscribe(
            MessageType.MARKET_DATA_UPDATE.value,
            self._handle_market_data_update
        )
        
        logger.info("Subscribed to all L2 topics")
    
    async def _handle_strategic_decision(self, message: Message):
        """Procesar decisión estratégica de L3"""
        
        start_time = datetime.now()
        correlation_id = self._generate_correlation_id()
        
        try:
            # Convert message
            l2_message = L2Message.from_bus_message(message)
            self._update_message_counts(l2_message.message_type)
            
            # Extract strategic decision
            decision_data = l2_message.data
            strategic_decision = StrategicDecision(
                regime=decision_data.get("regime", "neutral"),
                target_exposure=decision_data.get("target_exposure", 0.5),
                risk_appetite=decision_data.get("risk_appetite", "moderate"),
                preferred_assets=decision_data.get("preferred_assets", ["BTC/USDT"]),
                time_horizon=decision_data.get("time_horizon", "1h"),
                metadata=decision_data.get("metadata", {})
            )
            
            logger.info(
                f"Received strategic decision: regime={strategic_decision.regime}, "
                f"exposure={strategic_decision.target_exposure:.2f}"
            )
            
            # Store decision for processing
            self.pending_decisions[correlation_id] = strategic_decision
            self.active_correlations[correlation_id] = l2_message.correlation_id or correlation_id
            
            # Process decision asynchronously
            asyncio.create_task(self._process_strategic_decision(strategic_decision, correlation_id))
            
        except Exception as e:
            logger.error(f"Error handling strategic decision: {e}")
            await self._send_error_response(correlation_id, str(e))
        
        finally:
            # Update processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_processing_times("strategic_decision", processing_time)
    
    async def _process_strategic_decision(
        self,
        decision: StrategicDecision,
        correlation_id: str
    ):
        """Procesar decisión estratégica y generar señales tácticas"""
        
        async with self.processing_lock:
            try:
                # Update L2 state
                self.l2_state.current_regime = decision.regime
                self.l2_state.target_exposure = decision.target_exposure
                self.l2_state.risk_appetite = decision.risk_appetite
                
                # Generate tactical signals for each preferred asset
                for symbol in decision.preferred_assets:
                    await self._generate_tactical_signal_for_asset(
                        symbol, decision, correlation_id
                    )
                
                # Send processing completion message
                await self._send_processing_complete(correlation_id)
                
            except Exception as e:
                logger.error(f"Error processing strategic decision {correlation_id}: {e}")
                await self._send_error_response(correlation_id, str(e))
            
            finally:
                # Cleanup
                self.pending_decisions.pop(correlation_id, None)
    
    async def _generate_tactical_signal_for_asset(
        self,
        symbol: str,
        decision: StrategicDecision,
        correlation_id: str
    ):
        """Generar señal táctica para un activo específico"""
        
        try:
            # Get current market features
            market_features = await self._get_market_features(symbol)
            if not market_features:
                logger.warning(f"No market features available for {symbol}")
                return
            
            # Generate signal
            signals = await self.signal_generator.generate_signals(
                symbol=symbol,
                market_features=market_features,
                regime_context={
                    "regime": decision.regime,
                    "risk_appetite": decision.risk_appetite,
                    "target_exposure": decision.target_exposure
                }
            )
            
            if not signals:
                logger.info(f"No signals generated for {symbol}")
                return
            
            # Process each signal
            for signal in signals:
                await self._process_tactical_signal(signal, market_features, correlation_id)
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
    
    async def _process_tactical_signal(
        self,
        signal: TacticalSignal,
        market_features: MarketFeatures,
        correlation_id: str
    ):
        """Procesar una señal táctica completa (sizing + risk)"""
        
        try:
            # Calculate position size
            position_size = await self._calculate_position_size(signal, market_features)
            if not position_size:
                logger.info(f"Position sizing rejected for {signal.symbol}")
                return
            
            # Risk evaluation
            risk_metrics = await self._get_risk_metrics(signal.symbol)
            portfolio_state = self._get_current_portfolio_state()
            
            allow_trade, risk_alerts, adjusted_position = self.risk_manager.evaluate_pre_trade_risk(
                signal=signal,
                position_size=position_size,
                market_features=market_features,
                portfolio_state=portfolio_state
            )
            
            # Send risk alerts if any
            for alert in risk_alerts:
                await self._send_risk_alert(alert, correlation_id)
            
            if not allow_trade or not adjusted_position:
                logger.warning(f"Trade blocked by risk controls for {signal.symbol}")
                return
            
            # Send final tactical signal to L1
            await self._send_tactical_signal(signal, adjusted_position, correlation_id)
            
        except Exception as e:
            logger.error(f"Error processing tactical signal for {signal.symbol}: {e}")
    
    async def _send_tactical_signal(self, signal: TacticalSignal, position_size: PositionSize, correlation_id: str):
        """Enviar señal táctica a L1"""
        logger.info(f"Sending tactical signal for {signal.symbol}")
        message = L2Message(
            message_type=MessageType.TACTICAL_SIGNAL,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            data={
                "signal": asdict(signal),
                "position_size": asdict(position_size)
            }
        )
        await self.bus.publish(message.to_bus_message())
    
    async def _send_risk_alert(self, alert: RiskAlert, correlation_id: str):
        """Enviar alerta de riesgo"""
        logger.info(f"Sending risk alert for {alert.symbol}")
        message = L2Message(
            message_type=MessageType.RISK_ALERT,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            data=asdict(alert)
        )
        await self.bus.publish(message.to_bus_message())
    
    async def _handle_execution_report(self, message: Message):
        """Manejar report de ejecución de L1"""
        
        try:
            l2_message = L2Message.from_bus_message(message)
            exec_data = l2_message.data
            symbol = exec_data.get("symbol")
            status = exec_data.get("status")
            filled_size = exec_data.get("filled_size", 0)
            
            logger.info(f"Execution report for {symbol}: status={status}, filled={filled_size}")
            
            # Update position in risk manager
            if status == "filled" and symbol in self.risk_manager.current_positions:
                position = self.risk_manager.current_positions[symbol]
                position.size = filled_size
            
        except Exception as e:
            logger.error(f"Error handling execution report: {e}")
    
    async def _handle_position_update(self, message: Message):
        """Manejar actualización de posición de L1"""
        
        try:
            l2_message = L2Message.from_bus_message(message)
            position_data = l2_message.data
            
            symbol = position_data.get("symbol")
            current_size = position_data.get("current_size", 0)
            avg_price = position_data.get("avg_price", 0)
            unrealized_pnl = position_data.get("unrealized_pnl", 0)
            
            # Update risk manager with current position
            if symbol in self.risk_manager.current_positions:
                position = self.risk_manager.current_positions[symbol]
                position.current_price = position_data.get("current_price", position.current_price)
                position.unrealized_pnl = unrealized_pnl
                position.unrealized_pnl_pct = unrealized_pnl / (avg_price * abs(current_size)) if current_size != 0 else 0
                
            # Check for position close
            if current_size == 0 and symbol in self.risk_manager.current_positions:
                self.risk_manager.remove_position(symbol)
                self.l2_state.active_signals.pop(symbol, None)
                logger.info(f"Position closed for {symbol}")
                
        except Exception as e:
            logger.error(f"Error handling position update: {e}")
    
    async def _handle_market_data_update(self, message: Message):
        """Manejar actualización de datos de mercado"""
        
        try:
            l2_message = L2Message.from_bus_message(message)
            market_data = l2_message.data
            
            # Monitor existing positions for risk alerts
            if self.risk_manager.current_positions:
                price_data = {
                    symbol: data.get("price", 0) 
                    for symbol, data in market_data.items()
                }
                
                portfolio_value = market_data.get("portfolio_value", 100000)
                
                risk_alerts = self.risk_manager.monitor_existing_positions(
                    price_data, portfolio_value
                )
                
                # Send any triggered alerts
                for alert in risk_alerts:
                    await self._send_risk_alert(alert, self._generate_correlation_id())
                    
        except Exception as e:
            logger.error(f"Error handling market data update: {e}")
    
    async def _handle_regime_update(self, message: Message):
        """Manejar actualización de régimen de L3"""
        
        try:
            l2_message = L2Message.from_bus_message(message)
            regime_data = l2_message.data
            
            new_regime = regime_data.get("regime")
            confidence = regime_data.get("confidence", 0.5)
            
            if new_regime and new_regime != self.l2_state.current_regime:
                logger.info(f"Regime changed: {self.l2_state.current_regime} -> {new_regime}")
                
                self.l2_state.current_regime = new_regime
                self.l2_state.regime_confidence = confidence
                self.l2_state.last_regime_update = datetime.now()
                
                # Could trigger strategy rebalancing here
                
        except Exception as e:
            logger.error(f"Error handling regime update: {e}")
    
    async def _handle_portfolio_allocation(self, message: Message):
        """Manejar actualización de asignación de portfolio"""
        
        try:
            l2_message = L2Message.from_bus_message(message)
            allocation_data = l2_message.data
            
            new_target_exposure = allocation_data.get("target_exposure")
            risk_budget = allocation_data.get("risk_budget")
            
            if new_target_exposure is not None:
                self.l2_state.target_exposure = new_target_exposure
                logger.info(f"Updated target exposure: {new_target_exposure:.2f}")
            
            if risk_budget is not None:
                self.l2_state.risk_budget = risk_budget
                logger.info(f"Updated risk budget: {risk_budget:.2f}")
                
        except Exception as e:
            logger.error(f"Error handling portfolio allocation: {e}")
    
    # Helper methods
    async def _get_market_features(self, symbol: str) -> Optional[MarketFeatures]:
        """Obtener features de mercado para un símbolo"""
        
        # This would typically fetch from data layer
        # For now, return dummy data
        return MarketFeatures(
            volatility=0.25,
            volume_ratio=1.2,
            price_momentum=0.05,
            rsi=55.0,
            macd_signal="bullish"
        )
    
    async def _get_risk_metrics(self, symbol: str) -> RiskMetrics:
        """Obtener métricas de riesgo para un símbolo"""
        
        # This would typically calculate from historical data
        return RiskMetrics(
            var_95=-0.025,
            expected_shortfall=-0.035,
            max_drawdown=0.12,
            sharpe_ratio=1.8,
            volatility=0.25
        )
    
    def _get_current_portfolio_state(self) -> Dict:
        """Obtener estado actual del portfolio"""
        
        # This would typically come from portfolio manager
        return {
            "total_capital": 100000.0,
            "available_capital": 80000.0,
            "daily_pnl": 150.0,
            "portfolio_heat": 0.3
        }
    
    async def _send_processing_complete(self, correlation_id: str):
        """Enviar mensaje de procesamiento completo"""
        
        message = L2Message(
            message_type=MessageType.SIGNAL_GENERATED,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            data={"status": "completed"},
            metadata={"processing_time": datetime.now().isoformat()}
        )
        
        await self.bus.publish(message.to_bus_message())
    
    async def _send_error_response(self, correlation_id: str, error_message: str):
        """Enviar respuesta de error"""
        
        message = L2Message(
            message_type=MessageType.SIGNAL_GENERATED,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            data={
                "status": "error",
                "error_message": error_message
            },
            metadata={"error_time": datetime.now().isoformat()}
        )
        
        await self.bus.publish(message.to_bus_message())
    
    def _generate_correlation_id(self) -> str:
        """Generar correlation ID único"""
        return f"l2_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def _update_message_counts(self, message_type: MessageType):
        """Actualizar contadores de mensajes"""
        key = message_type.value
        self.message_counts[key] = self.message_counts.get(key, 0) + 1
    
    def _update_processing_times(self, operation: str, processing_time: float):
        """Actualizar tiempos de procesamiento"""
        if operation not in self.processing_times:
            self.processing_times[operation] = []
        
        self.processing_times[operation].append(processing_time)
        
        # Keep only last 100 measurements
        if len(self.processing_times[operation]) > 100:
            self.processing_times[operation] = self.processing_times[operation][-100:]
    
    async def _heartbeat_task(self):
        """Task de heartbeat para monitoreo"""
        
        while self.is_running:
            try:
                heartbeat_msg = L2Message(
                    message_type=MessageType.SIGNAL_GENERATED,  # Using as generic status
                    timestamp=datetime.now(),
                    data={
                        "status": "healthy",
                        "active_signals": len(self.l2_state.active_signals),
                        "pending_decisions": len(self.pending_decisions),
                        "message_counts": self.message_counts.copy()
                    },
                    metadata={"heartbeat": True}
                )
                
                await self.bus.publish(heartbeat_msg.to_bus_message())
                
                # Wait 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_task(self):
        """Task de limpieza periódica"""
        
        while self.is_running:
            try:
                # Cleanup old correlations
                now = datetime.now()
                cutoff = now - timedelta(hours=1)
                
                # This would need timestamp tracking for correlations
                # For now, just clean up old alerts
                if hasattr(self.risk_manager, 'cleanup_old_alerts'):
                    self.risk_manager.cleanup_old_alerts(max_age_hours=24)
                
                # Wait 10 minutes
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(600)
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado actual del adaptador"""
        
        avg_processing_times = {}
        for operation, times in self.processing_times.items():
            if times:
                avg_processing_times[operation] = {
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }
        
        return {
            "is_running": self.is_running,
            "current_regime": self.l2_state.current_regime,
            "target_exposure": self.l2_state.target_exposure,
            "active_signals": len(self.l2_state.active_signals),
            "pending_decisions": len(self.pending_decisions),
            "message_counts": self.message_counts.copy(),
            "avg_processing_times": avg_processing_times,
            "last_signal_time": self.l2_state.last_signal_time.isoformat() if self.l2_state.last_signal_time else None,
            "risk_summary": self.risk_manager.get_portfolio_risk_summary() if self.risk_manager else {}
        }


# Factory function for easy initialization
def create_l2_bus_adapter(
    message_bus: MessageBus,
    config_path: Optional[str] = None
) -> L2BusAdapter:
    """Factory para crear L2BusAdapter configurado"""
    
    # Load config
    if config_path:
        config = L2Config.from_file(config_path)
    else:
        config = L2Config.from_env()
    
    # Create adapter
    adapter = L2BusAdapter(message_bus, config)
    
    return adapter


# Example usage
if __name__ == "__main__":
    async def main():
        # Create message bus (this would come from main system)
        from comms.message_bus import MessageBus
        
        bus = MessageBus()
        
        # Create L2 adapter
        adapter = create_l2_bus_adapter(bus)
        
        # Start adapter
        await adapter.start()
        
        # Send test strategic decision
        test_decision = L2Message(
            message_type=MessageType.STRATEGIC_DECISION,
            timestamp=datetime.now(),
            data={
                "regime": "trending",
                "target_exposure": 0.7,
                "risk_appetite": "aggressive",
                "preferred_assets": ["BTC/USDT"],
                "time_horizon": "4h"
            }
        )
        
        await bus.publish(test_decision.to_bus_message())
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # Check status
        status = adapter.get_status()
        print("L2 Adapter Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Stop adapter
        await adapter.stop()
    
    # Run example
    asyncio.run(main())