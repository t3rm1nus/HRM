"""
Selling Strategy - Four-Level Hierarchical Exit System

Implements the comprehensive selling strategy with four priority levels:

🔴 PRIORITY 1: SELL OBLIGATORIO — Protección (L1)
   - Stop-loss at 1% loss (immediate, no opinions needed)
   - Triggers automatically on position management

🟠 PRIORITY 2: SELL TÁCTICO — El edge murió (L2)
   - When original BUY signal disappears
   - Momentum down, RSI divergence, volume dry
   - L2 tactical assessment

🟡 PRIORITY 3: SELL ESTRATÉGICO — El régimen cambió (L3)
   - Regime changes from TRENDING to RANGE/BEAR
   - Confidence < 0.5
   - L3 strategic assessment

🔵 PRIORITY 4: SELL POR TIEMPO — Nada pasa (Timeout)
   - After 120 cycles (~2 hours) if trade doesn't progress
   - Max holding time or regime timeout

Priority execution order: 🔴 > 🟠 > 🟡 > 🔵
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from core.logging import logger
from l2_tactic.technical.multi_timeframe import MultiTimeframeTechnical


@dataclass
class SellSignal:
    """Represents a sell signal with priority and reason"""
    symbol: str
    priority: int  # 1=Stop-loss, 2=Tactical, 3=Strategic, 4=Timeout
    reason: str
    confidence: float
    quantity: float  # Position size to sell (negative for sell)
    price: float
    timestamp: datetime
    source: str  # 'L1_STOP_LOSS', 'L2_TACTICAL', 'L3_STRATEGIC', 'TIMEOUT'
    metadata: Dict[str, Any] = None


class SellingStrategy:
    """
    Four-level hierarchical selling system.

    Priority hierarchy ensures risk management (stop-loss) always comes first,
    followed by tactical edge assessment, strategic regime changes, and timeout.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Stop-loss configuration (Priority 1)
        self.stop_loss_config = {
            'pct_threshold': 0.01,  # 1% loss triggers stop-loss
            'immediate_execution': True,
            'no_opinion_required': True
        }

        # Tactical edge configuration (Priority 2)
        self.tactical_config = {
            'momentum_threshold': 0.3,  # Momentum drop threshold
            'rsi_extreme_threshold': 75,  # RSI extreme zone
            'volume_drop_threshold': 0.7,  # Volume drop ratio
            'edge_fade_timeout': 300  # 5 minutes after edge fades
        }

        # Strategic regime configuration (Priority 3)
        self.strategic_config = {
            'regime_change_timeout': 1800,  # 30 minutes for regime assessment
            'confidence_threshold': 0.5,  # Confidence drop threshold
            'regime_change_signals': ['TRENDING→RANGE', 'TRENDING→BEAR', 'TRENDING→VOLATILE']
        }

        # Timeout configuration (Priority 4)
        self.timeout_config = {
            'max_cycles': 120,  # ~2 hours assuming 1min cycles
            'max_holding_time': 7200,  # 2 hours in seconds
            'progress_threshold': 0.005,  # 0.5% minimum progress
            'regime_timeout_multiplier': 2.0  # Double timeout in ranging markets
        }

        # Position tracking
        self.active_positions = {}  # symbol -> position data
        self.position_entry_signals = {}  # symbol -> original BUY signal data
        self.last_edge_assessment = {}  # symbol -> timestamp of last edge check

        # Initialize technical analysis with minimal config
        from l2_tactic.technical.multi_timeframe import MultiTimeframeTechnical
        minimal_config = type('Config', (), {
            'signals': type('Signals', (), {})()
        })()
        self.technical_analyzer = MultiTimeframeTechnical(minimal_config)

        logger.info("✅ SellingStrategy initialized with 4-level hierarchy")

    def assess_sell_opportunities(self, symbol: str, current_price: float,
                                market_data: pd.DataFrame, l3_context: Dict[str, Any],
                                position_data: Dict[str, Any]) -> Optional[SellSignal]:
        """
        Assess all four selling levels and return highest priority sell signal.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            market_data: Recent market data (DataFrame)
            l3_context: L3 strategic context
            position_data: Current position information

        Returns:
            SellSignal if selling opportunity found, None otherwise
        """

        # ========================================================================================
        # 🔴 PRIORITY 1: SELL OBLIGATORIO — Stop-Loss Protection (L1)
        # ========================================================================================
        stop_loss_signal = self._assess_stop_loss_protection(symbol, current_price, position_data)
        if stop_loss_signal:
            logger.warning(f"🚨 STOP-LOSS TRIGGERED: {symbol} {stop_loss_signal.reason}")
            return stop_loss_signal

        # ========================================================================================
        # 🟠 PRIORITY 2: SELL TÁCTICO — Edge Disappearance (L2)
        # ========================================================================================
        tactical_signal = self._assess_tactical_edge(symbol, market_data, position_data, l3_context)
        if tactical_signal:
            logger.info(f"🎯 TACTICAL SELL: {symbol} {tactical_signal.reason}")
            return tactical_signal

        # ========================================================================================
        # 🟡 PRIORITY 3: SELL ESTRATÉGICO — Regime Change (L3)
        # ========================================================================================
        strategic_signal = self._assess_strategic_regime(symbol, l3_context, position_data)
        if strategic_signal:
            logger.info(f"🌟 STRATEGIC SELL: {symbol} {strategic_signal.reason}")
            return strategic_signal

        # ========================================================================================
        # 🔵 PRIORITY 4: SELL POR TIEMPO — Timeout (System)
        # ========================================================================================
        timeout_signal = self._assess_timeout_exit(symbol, position_data, l3_context)
        if timeout_signal:
            logger.info(f"⏰ TIMEOUT SELL: {symbol} {timeout_signal.reason}")
            return timeout_signal

        # No sell signal at any level
        return None

    def _assess_stop_loss_protection(self, symbol: str, current_price: float,
                                   position_data: Dict[str, Any]) -> Optional[SellSignal]:
        """
        🔴 PRIORITY 1: Stop-loss at 1% loss - immediate execution, no opinions needed.

        This is the most critical protection mechanism.
        """
        try:
            entry_price = position_data.get('entry_price', 0)
            position_qty = position_data.get('quantity', 0)

            if entry_price <= 0 or position_qty <= 0:
                return None

            # Calculate current P&L percentage
            if position_qty > 0:  # Long position
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # Short position (though we focus on longs)
                pnl_pct = (entry_price - current_price) / entry_price

            # Check stop-loss threshold (1% loss)
            if pnl_pct <= -self.stop_loss_config['pct_threshold']:
                return SellSignal(
                    symbol=symbol,
                    priority=1,
                    reason=f"Stop-loss triggered: {pnl_pct:.2f}% loss (threshold: -{self.stop_loss_config['pct_threshold']*100:.1f}%)",
                    confidence=1.0,  # Maximum confidence for risk management
                    quantity=-abs(position_qty),  # Sell entire position
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    source='L1_STOP_LOSS',
                    metadata={
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'pnl_pct': pnl_pct,
                        'threshold_pct': -self.stop_loss_config['pct_threshold'],
                        'immediate_execution': True
                    }
                )

        except Exception as e:
            logger.error(f"❌ Error assessing stop-loss for {symbol}: {e}")

        return None

    def _assess_tactical_edge(self, symbol: str, market_data: pd.DataFrame,
                            position_data: Dict[str, Any], l3_context: Dict[str, Any]) -> Optional[SellSignal]:
        """
        🟠 PRIORITY 2: Sell when original BUY signal disappears.

        Monitor for: momentum down, RSI divergence, volume dry up.
        """
        try:
            if market_data is None or market_data.empty:
                return None

            # Get original BUY signal conditions
            entry_signal = self.position_entry_signals.get(symbol, {})
            entry_price = position_data.get('entry_price', 0)
            position_qty = position_data.get('quantity', 0)

            if not entry_signal or entry_price <= 0:
                return None

            # Calculate current technical indicators
            indicators = self.technical_analyzer.calculate_technical_indicators(market_data)

            # Check for edge disappearance conditions
            edge_gone = False
            reasons = []

            # Condition 1: Momentum reversal
            current_momentum = indicators.get('momentum', 0)
            original_momentum = entry_signal.get('momentum', 0)

            if original_momentum > self.tactical_config['momentum_threshold'] and \
               current_momentum < -self.tactical_config['momentum_threshold']:
                edge_gone = True
                reasons.append(f"Momentum reversed: {original_momentum:.2f} → {current_momentum:.2f}")

            # Condition 2: RSI divergence (extreme levels)
            rsi = indicators.get('rsi', 50)
            if rsi >= self.tactical_config['rsi_extreme_threshold'] or rsi <= (100 - self.tactical_config['rsi_extreme_threshold']):
                edge_gone = True
                reasons.append(f"RSI in extreme zone: {rsi:.1f}")

            # Condition 3: Volume drying up
            current_volume = indicators.get('volume', 0)
            avg_volume = indicators.get('volume_sma', 1)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            if volume_ratio < self.tactical_config['volume_drop_threshold']:
                edge_gone = True
                reasons.append(f"Volume dried up: {volume_ratio:.2f} of average")

            # Condition 4: Price action against position
            if position_qty > 0 and current_price < entry_price * 0.98:  # 2% below entry
                edge_gone = True
                reasons.append(f"Price action turned: {((current_price/entry_price)-1)*100:.1f}% from entry")

            if edge_gone:
                confidence = min(0.8, len(reasons) * 0.2)  # Higher confidence with more reasons

                return SellSignal(
                    symbol=symbol,
                    priority=2,
                    reason=f"Tactical edge disappeared: {'; '.join(reasons[:2])}",  # Limit to top 2 reasons
                    confidence=confidence,
                    quantity=-abs(position_qty),  # Sell entire position
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    source='L2_TACTICAL',
                    metadata={
                        'reasons': reasons,
                        'indicators': indicators,
                        'entry_signal': entry_signal,
                        'edge_fade_detected': True
                    }
                )

        except Exception as e:
            logger.error(f"❌ Error assessing tactical edge for {symbol}: {e}")

        return None

    def _assess_strategic_regime(self, symbol: str, l3_context: Dict[str, Any],
                               position_data: Dict[str, Any]) -> Optional[SellSignal]:
        """
        🟡 PRIORITY 3: Sell when regime changes from TRENDING.

        Monitor for: TRENDING→RANGE, TRENDING→BEAR, confidence < 0.5
        """
        try:
            current_regime = l3_context.get('regime', 'unknown')
            regime_confidence = l3_context.get('confidence', 0.5)
            position_qty = position_data.get('quantity', 0)

            if position_qty <= 0:
                return None

            # Get original entry regime
            entry_regime = position_data.get('entry_regime', 'unknown')
            regime_change_detected = False
            reasons = []

            # Condition 1: TRENDING regime lost
            if entry_regime == 'TRENDING' and current_regime in ['RANGE', 'BEAR', 'VOLATILE']:
                regime_change_detected = True
                reasons.append(f"Regime changed: TRENDING → {current_regime}")

            # Condition 2: Confidence collapse
            if regime_confidence < self.strategic_config['confidence_threshold']:
                regime_change_detected = True
                reasons.append(f"Confidence collapsed: {regime_confidence:.2f} < {self.strategic_config['confidence_threshold']}")

            # Condition 3: Strategic signal changed to SELL
            l3_signal = l3_context.get('signal', 'hold')
            if l3_signal == 'sell':
                regime_change_detected = True
                reasons.append(f"L3 strategic signal: {l3_signal.upper()}")

            if regime_change_detected:
                return SellSignal(
                    symbol=symbol,
                    priority=3,
                    reason=f"Strategic regime change: {'; '.join(reasons[:2])}",
                    confidence=min(0.9, regime_confidence + 0.2),  # Boost confidence slightly
                    quantity=-abs(position_qty),  # Sell entire position
                    price=0,  # Will be filled with current price at execution
                    timestamp=datetime.utcnow(),
                    source='L3_STRATEGIC',
                    metadata={
                        'reasons': reasons,
                        'current_regime': current_regime,
                        'entry_regime': entry_regime,
                        'regime_confidence': regime_confidence,
                        'l3_signal': l3_signal,
                        'regime_change_detected': True
                    }
                )

        except Exception as e:
            logger.error(f"❌ Error assessing strategic regime for {symbol}: {e}")

        return None

    def _assess_timeout_exit(self, symbol: str, position_data: Dict[str, Any],
                           l3_context: Dict[str, Any]) -> Optional[SellSignal]:
        """
        🔵 PRIORITY 4: Sell after timeout if trade doesn't progress.

        After 120 cycles (~2 hours) or when progress < 0.5%
        """
        try:
            entry_time = position_data.get('entry_timestamp')
            entry_price = position_data.get('entry_price', 0)
            position_qty = position_data.get('quantity', 0)
            current_regime = l3_context.get('regime', 'TRENDING')

            if not entry_time or entry_price <= 0 or position_qty <= 0:
                return None

            # Calculate holding time
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            elif isinstance(entry_time, (int, float)):
                entry_time = datetime.fromtimestamp(entry_time)

            holding_time = (datetime.utcnow() - entry_time).total_seconds()

            # Adjust timeout based on regime (longer in ranging markets)
            base_timeout = self.timeout_config['max_holding_time']
            if current_regime == 'RANGE':
                timeout_limit = base_timeout * self.timeout_config['regime_timeout_multiplier']
            else:
                timeout_limit = base_timeout

            # Check timeout condition
            if holding_time > timeout_limit:
                return SellSignal(
                    symbol=symbol,
                    priority=4,
                    reason=f"Timeout exit: held {holding_time/3600:.1f}h (limit: {timeout_limit/3600:.1f}h)",
                    confidence=0.6,  # Moderate confidence for timeout exits
                    quantity=-abs(position_qty),  # Sell entire position
                    price=0,  # Will be filled with current price at execution
                    timestamp=datetime.utcnow(),
                    source='TIMEOUT',
                    metadata={
                        'holding_time_seconds': holding_time,
                        'timeout_limit_seconds': timeout_limit,
                        'current_regime': current_regime,
                        'entry_time': entry_time.isoformat(),
                        'timeout_triggered': True
                    }
                )

            # Check progress condition (no meaningful movement)
            # This would need current_price passed in, but for now we rely on time-based timeout

        except Exception as e:
            logger.error(f"❌ Error assessing timeout exit for {symbol}: {e}")

        return None

    def register_position_entry(self, symbol: str, entry_data: Dict[str, Any],
                              market_data: pd.DataFrame, l3_context: Dict[str, Any]):
        """
        Register a new position entry for sell strategy tracking.

        Args:
            symbol: Trading symbol
            entry_data: Position entry information
            market_data: Market data at entry time
            l3_context: L3 context at entry time
        """
        try:
            # Calculate entry indicators
            indicators = {}
            if market_data is not None and not market_data.empty:
                indicators = self.technical_analyzer.calculate_technical_indicators(market_data)

            # Store position data
            self.active_positions[symbol] = {
                'entry_price': entry_data.get('price', 0),
                'quantity': entry_data.get('quantity', 0),
                'entry_timestamp': datetime.utcnow(),
                'entry_indicators': indicators,
                'entry_regime': l3_context.get('regime', 'unknown'),
                'entry_confidence': l3_context.get('confidence', 0.5)
            }

            # Store original BUY signal data
            self.position_entry_signals[symbol] = {
                'momentum': indicators.get('momentum', 0),
                'rsi': indicators.get('rsi', 50),
                'volume': indicators.get('volume', 0),
                'l3_context': l3_context.copy(),
                'timestamp': datetime.utcnow()
            }

            logger.info(f"📝 Position entry registered for {symbol}: price=${entry_data.get('price', 0):.2f}, qty={entry_data.get('quantity', 0):.6f}")

        except Exception as e:
            logger.error(f"❌ Error registering position entry for {symbol}: {e}")

    def close_position(self, symbol: str):
        """Remove position from tracking when closed."""
        try:
            self.active_positions.pop(symbol, None)
            self.position_entry_signals.pop(symbol, None)
            self.last_edge_assessment.pop(symbol, None)
            logger.info(f"✅ Position tracking closed for {symbol}")
        except Exception as e:
            logger.error(f"❌ Error closing position tracking for {symbol}: {e}")

    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently tracked positions."""
        return self.active_positions.copy()

    def get_sell_priority_name(self, priority: int) -> str:
        """Convert priority number to descriptive name."""
        priority_names = {
            1: "🔴 STOP-LOSS (L1)",
            2: "🟠 TÁCTICO (L2)",
            3: "🟡 ESTRATÉGICO (L3)",
            4: "🔵 TIMEOUT"
        }
        return priority_names.get(priority, f"UNKNOWN ({priority})")


# Global instance
selling_strategy = SellingStrategy()


def get_selling_strategy() -> SellingStrategy:
    """Get the global selling strategy instance."""
    return selling_strategy
