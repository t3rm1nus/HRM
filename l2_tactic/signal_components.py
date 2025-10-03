"""
Shared Signal Processing Components - Common functionality mixins

Contains common functionality shared across signal generators:
- BTC/ETH synchronization mixin
- Weight calculator integration mixin
- Risk overlay mixin
- Post-processing utilities
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from core.logging import logger
from core.models import TacticalSignal
import pandas as pd
from utils import safe_float


class BTCEthereumSynchronizationMixin:
    """
    Mixin for BTC/ETH synchronization functionality
    """

    async def apply_btc_eth_synchronization_async(self, signals: List[TacticalSignal],
                                               market_data: Dict[str, pd.DataFrame],
                                               state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Apply BTC/ETH synchronization to signals
        Delegates to the btc_eth_synchronizer module
        """
        try:
            # Import the synchronizer module
            from .btc_eth_synchronizer import btc_eth_synchronizer

            # Convert TacticalSignal objects to dictionaries for compatibility
            signal_dicts = [signal.asdict() for signal in signals]

            # Apply synchronization
            synchronized_dicts = await btc_eth_synchronizer.apply_btc_eth_synchronization(
                signal_dicts, market_data, state
            )

            # Convert back to TacticalSignal objects (simplified - assuming same structure)
            synchronized_signals = []
            for sig_dict in synchronized_dicts:
                signal = TacticalSignal.from_dict(sig_dict) if hasattr(TacticalSignal, 'from_dict') else \
                        self._dict_to_tactical_signal(sig_dict)
                synchronized_signals.append(signal)

            return synchronized_signals

        except Exception as e:
            logger.error(f"❌ Error in BTC/ETH synchronization mixin: {e}")
            return signals  # Return original signals on error

    def _dict_to_tactical_signal(self, signal_dict: Dict) -> TacticalSignal:
        """Convert signal dictionary to TacticalSignal object (fallback method)"""
        # This is a simplified conversion - in practice would need full mapping
        return TacticalSignal(
            symbol=signal_dict.get('symbol', 'UNKNOWN'),
            side=signal_dict.get('side', 'hold'),
            confidence=signal_dict.get('confidence', 0.5),
            strength=signal_dict.get('strength', 0.5),
            signal_type=signal_dict.get('signal_type', 'unknown'),
            source=signal_dict.get('source', 'unknown'),
            timestamp=signal_dict.get('timestamp', pd.Timestamp.now()),
            features=signal_dict.get('features', {}),
            metadata=signal_dict.get('metadata', {})
        )


class WeightCalculatorIntegrationMixin:
    """
    Mixin for weight calculator integration functionality
    """

    async def apply_weight_calculator_async(self, signals: List[TacticalSignal],
                                         market_data: Dict[str, pd.DataFrame],
                                         state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Apply weight calculator integration to signals
        Delegates to the weight_calculator_integration module
        """
        try:
            # Import the weight calculator integration
            from .weight_calculator_integration import weight_calculator_integrator

            # Convert to dictionaries for compatibility (weight calculator expects dict format)
            signal_dicts = [signal.asdict() for signal in signals]

            # Apply weight calculator
            weighted_dicts = await weight_calculator_integrator.apply_weight_calculator_integration(
                signal_dicts, market_data, state
            )

            # Convert back to TacticalSignal objects
            weighted_signals = []
            for sig_dict in weighted_dicts:
                signal = TacticalSignal.from_dict(sig_dict) if hasattr(TacticalSignal, 'from_dict') else \
                        self._dict_to_tactical_signal(sig_dict)
                weighted_signals.append(signal)

            return weighted_signals

        except Exception as e:
            logger.error(f"❌ Error in weight calculator integration mixin: {e}")
            return signals  # Return original signals on error

    def _dict_to_tactical_signal(self, signal_dict: Dict) -> TacticalSignal:
        """Convert signal dictionary to TacticalSignal object (fallback method)"""
        return TacticalSignal(
            symbol=signal_dict.get('symbol', 'UNKNOWN'),
            side=signal_dict.get('side', 'hold'),
            confidence=signal_dict.get('confidence', 0.5),
            strength=signal_dict.get('strength', 0.5),
            signal_type=signal_dict.get('signal_type', 'unknown'),
            source=signal_dict.get('source', 'unknown'),
            timestamp=signal_dict.get('timestamp', pd.Timestamp.now()),
            features=signal_dict.get('features', {}),
            metadata=signal_dict.get('metadata', {})
        )


class RiskOverlayMixin:
    """
    Mixin for risk overlay functionality
    """

    async def apply_risk_overlay_async(self, signals: List[TacticalSignal],
                                    market_data: Dict[str, pd.DataFrame],
                                    portfolio_state: Dict[str, Any],
                                    l3_context: Optional[Dict] = None) -> List[TacticalSignal]:
        """
        Apply risk overlay to signals
        Delegates to the risk_overlay module
        """
        try:
            # Import the risk overlay module
            from .risk_overlay import risk_overlay

            # Apply risk signals (this method expects specific parameters)
            risk_signals = await risk_overlay.generate_risk_signals(
                market_data=market_data,
                portfolio_data=portfolio_state
                # Note: risk_overlay.generate_risk_signals might have a different signature
            )

            # Apply risk adjustments to each signal
            adjusted_signals = []
            for signal in signals:
                adjusted_signal = self._apply_risk_adjustment(signal, risk_signals, l3_context)
                adjusted_signals.append(adjusted_signal)

            return adjusted_signals

        except Exception as e:
            logger.error(f"❌ Error in risk overlay mixin: {e}")
            return signals  # Return original signals on error

    def _apply_risk_adjustment(self, signal: TacticalSignal, risk_signals: List[TacticalSignal],
                              l3_context: Optional[Dict]) -> TacticalSignal:
        """
        Apply risk adjustment logic (simplified version from original code)
        """
        try:
            if not risk_signals:
                return signal

            risk_appetite = l3_context.get('risk_appetite', 0.5) if l3_context else 0.5
            regime = l3_context.get('regime', 'neutral').lower() if l3_context else 'neutral'

            # Check for critical risk signals
            has_close_all = any(getattr(r, 'side', '') == 'close_all' for r in risk_signals)
            has_reduce = any(getattr(r, 'side', '') == 'reduce' for r in risk_signals)

            original_side = getattr(signal, 'side', 'hold')
            original_confidence = getattr(signal, 'confidence', 0.5)

            # CRITICAL RISK: Close all positions
            if has_close_all:
                logger.warning(f"🚨 CRITICAL RISK: Converting {original_side} to HOLD due to close_all signal")
                signal.side = 'hold'
                signal.confidence = min(original_confidence, 0.3)

            # HIGH RISK: Reduce positions
            elif has_reduce and risk_appetite < 0.5:
                # Allow high-confidence L2 signals to bypass some risk reduction
                l2_confidence = getattr(signal, 'confidence', 0.5)
                source = getattr(signal, 'source', '')

                if source.startswith('l2') and l2_confidence > 0.75:
                    # Bypass with strong L2 signal
                    signal.confidence = max(0.65, signal.confidence * 0.95)
                    signal.strength *= 0.9
                else:
                    # Standard risk reduction
                    signal.confidence *= 0.8
                    signal.strength *= 0.7

            return signal

        except Exception as e:
            logger.error(f"❌ Error applying risk adjustment: {e}")
            return signal


class SignalValidationMixin:
    """
    Mixin for signal validation and filtering
    """

    def validate_signal(self, signal: TacticalSignal, portfolio_state: Dict[str, Any]) -> bool:
        """
        Validate a signal against portfolio state and business rules
        """
        try:
            # Check if it's a valid TacticalSignal
            if not hasattr(signal, 'side') or not hasattr(signal, 'confidence'):
                return False

            side = getattr(signal, 'side', 'hold')
            confidence = getattr(signal, 'confidence', 0.5)
            symbol = getattr(signal, 'symbol', '')

            # Basic validations
            if side not in ['buy', 'sell', 'hold']:
                logger.warning(f"⚠️ Invalid signal side: {side}")
                return False

            if not (0.0 <= confidence <= 1.0):
                logger.warning(f"⚠️ Invalid confidence range: {confidence}")
                return False

            if not symbol or symbol not in ['BTCUSDT', 'ETHUSDT']:
                logger.warning(f"⚠️ Invalid or unsupported symbol: {symbol}")
                return False

            # Portfolio validation: No SELL if no position (simplified)
            if side == 'sell':
                portfolio = portfolio_state.get("portfolio", {})
                position = portfolio.get(symbol, {}).get("position", 0.0)
                if abs(position) < 0.0001:  # No position
                    logger.warning(f"⚠️ SELL signal for {symbol} but no position ({position:.6f})")
                    return False

            return True

        except Exception as e:
            logger.error(f"❌ Error validating signal: {e}")
            return False

    def filter_signals(self, signals: List[TacticalSignal], portfolio_state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Filter and validate a list of signals
        """
        valid_signals = []
        for signal in signals:
            if self.validate_signal(signal, portfolio_state):
                valid_signals.append(signal)
            else:
                logger.warning(f"🚨 Signal validation failed, skipping: {signal}")
        return valid_signals


class PostProcessingMixin(BTCEthereumSynchronizationMixin, WeightCalculatorIntegrationMixin,
                        RiskOverlayMixin, SignalValidationMixin):
    """
    Combined mixin with all post-processing capabilities
    """

    async def apply_post_processing(self, signals: List[TacticalSignal],
                                  market_data: Dict[str, pd.DataFrame],
                                  state: Dict[str, Any],
                                  l3_context: Optional[Dict] = None) -> List[TacticalSignal]:
        """
        Apply all post-processing steps in correct order
        """
        try:
            logger.info(f"🔄 Applying post-processing to {len(signals)} signals")

            # 1. Signal validation and filtering
            valid_signals = self.filter_signals(signals, state)
            logger.info(f"✓ Validation: {len(signals)} → {len(valid_signals)} signals")

            # 2. Risk overlay
            risk_filtered_signals = await self.apply_risk_overlay_async(
                valid_signals, market_data, state, l3_context
            )
            logger.info(f"✓ Risk overlay: {len(valid_signals)} → {len(risk_filtered_signals)} signals")

            # 3. BTC/ETH synchronization
            synchronized_signals = await self.apply_btc_eth_synchronization_async(
                risk_filtered_signals, market_data, state
            )
            logger.info(f"✓ BTC/ETH sync: {len(risk_filtered_signals)} → {len(synchronized_signals)} signals")

            # 4. Weight calculator integration
            final_signals = await self.apply_weight_calculator_async(
                synchronized_signals, market_data, state
            )
            logger.info(f"✓ Weight calculator: {len(synchronized_signals)} → {len(final_signals)} signals")

            return final_signals

        except Exception as e:
            logger.error(f"❌ Error in post-processing: {e}")
            return signals  # Return original signals on error
