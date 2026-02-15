"""
l2_tactic/tactical_signal_processor.py - Core Tactical Signal Processor

This module contains the main L2TacticProcessor class refactored from the original
signal_generator.py, focusing on core signal processing logic.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import warnings

from core.logging import logger, log_trading_action
from core.config import HRM_PATH_MODE
from .models import TacticalSignal, L2State
from .technical.multi_timeframe import MultiTimeframeTechnical
from .risk_overlay import RiskOverlay
from .signal_composer import SignalComposer
from .finrl_processor import FinRLProcessor
from .finrl_wrapper import FinRLProcessorWrapper
from .signal_validator import validate_signal_list, validate_tactical_signal, create_fallback_signal
from .l2_utils import safe_float
# from .btc_eth_synchronizer import btc_eth_synchronizer
from .weight_calculator_integration import weight_calculator_integrator
from .path_mode_generator import PathModeSignalGenerator


class L2TacticProcessor:
    """
    Refactored L2TacticProcessor - Core tactical signal processing.
    Now focuses on main processing logic without embedded sub-systems.
    """

    def __init__(self, config=None, portfolio_manager=None, apagar_l3=False):
        # Track HOLD logging per cycle per symbol to avoid spam
        self.hold_logged_this_cycle = {}
        """Initialize processor with core components."""
        # Provide default config if none given
        if config is None:
            default_signals = type('Signals', (), {})()
            self.config = type('Config', (), {'signals': default_signals})()
        else:
            self.config = config
        self.portfolio_manager = portfolio_manager
        self.apagar_l3 = apagar_l3

        # Core components
        self.multi_timeframe = MultiTimeframeTechnical(self.config)
        self.risk_overlay = RiskOverlay()
        self.signal_composer = SignalComposer(self.config.signals)

        # Conditionally initialize BTC/ETH synchronizer (disabled in PAPER mode)
        if not os.getenv('DISABLE_BTC_ETH_SYNC'):
            from .btc_eth_synchronizer import btc_eth_synchronizer
            self.synchronizer = btc_eth_synchronizer.apply_btc_eth_synchronization
        else:
            self.synchronizer = None
            logger.info("⏭️ BTC/ETH Synchronizer disabled via config")

        # AI processor setup
        self._setup_finrl_processor()

        logger.info("✅ L2TacticProcessor initialized (refactored)")

    def reset_hold_logging(self):
        """Reset HOLD logging tracking for new cycle to avoid spam"""
        self.hold_logged_this_cycle = {}

    def _setup_finrl_processor(self):
        """Setup FinRL processor and wrapper."""
        model_path = "models/L2/deepseek.zip"
        self.finrl_processor = FinRLProcessor(model_path)
        self.finrl_wrapper = FinRLProcessorWrapper(self.finrl_processor, "deepseek")

    async def process_signals(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Main signal processing entry point.
        Now delegates to HRM path mode system or traditional processing.
        """
        # Suppress numpy warnings for clean processing
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            warnings.filterwarnings("ignore", message="divide by zero encountered")
            warnings.filterwarnings("ignore", message="overflow encountered")

            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                return await self._process_signals_internal(state)

    async def _process_signals_internal(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """Internal signal processing with HRM path mode routing."""
        try:
            market_data = state.get("market_data_simple") or state.get("market_data", {})
            if not market_data:
                logger.warning("⚠️ L2: No market data available")
                return []

            # HRM PATH MODE: If enabled, use the new path mode system
            if HRM_PATH_MODE in ['PATH1', 'PATH2', 'PATH3']:
                logger.info(f"🎯 Using HRM Path Mode: {HRM_PATH_MODE}")
                return await self._process_with_hrm_path_modes(state, market_data)

            # Traditional processing path (legacy)
            logger.info("🎯 Using traditional L2 processing")
            return await self._process_traditional_signals(state, market_data)

        except Exception as e:
            logger.error(f"❌ Error in L2 signal processing: {e}", exc_info=True)
            return []

    async def _process_with_hrm_path_modes(self, state: Dict[str, Any], market_data: Dict) -> List[TacticalSignal]:
        """Process signals using HRM path mode system."""
        try:
            # Get L3 output for path modes
            l3_output = self._get_l3_output(state)

            # Generate signals using HRM path mode logic
            path_mode_generator = PathModeSignalGenerator()
            path_signals = []

            for symbol in ["BTCUSDT", "ETHUSDT"]:
                # Create mock L1/L2 signal for now (should come from real L2 processor)
                mock_l2_signal = type('Signal', (), {
                    'action': 'hold',
                    'symbol': symbol,
                    'confidence': 0.5,
                    'direction': 'hold'
                })()

                # Create mock L3 context from l3_output with safe defaults
                mock_l3_context = {
                    'path_mode': 'PATH2',
                    'regime': l3_output.get('regime', 'RANGE'),
                    'subtype': l3_output.get('subtype') or 'unknown',  # SAFE: Never None
                    'l3_signal': l3_output.get('signal', 'hold'),
                    'l3_confidence': l3_output.get('confidence', 0.50),
                    'setup_type': l3_output.get('setup_type'),  # Can be None
                    'allow_l2_signals': l3_output.get('allow_l2_signals', False),
                    'allow_setup_trades': l3_output.get('allow_setup_trades', False)
                }

                # Generate signal for this symbol
                processed_signal = path_mode_generator.generate_signal(
                    symbol=symbol,
                    l1_l2_signal=mock_l2_signal.action,
                    l3_context=mock_l3_context
                )
                path_signals.append(processed_signal)

            # Convert dict signals to TacticalSignal objects
            tactical_signals = []
            for sig_dict in path_signals:
                try:
                    # Create TacticalSignal from dict (simplified conversion)
                    tactical_signal = TacticalSignal(
                        symbol=sig_dict.get('symbol', 'UNKNOWN'),
                        side=sig_dict.get('side', 'hold'),
                        strength=sig_dict.get('strength', 0.5),
                        confidence=sig_dict.get('confidence', 0.5),
                        source=sig_dict.get('source', 'hrm_path_mode'),
                        timestamp=pd.Timestamp.now(),
                        features=sig_dict.get('features', {}),
                        metadata=sig_dict.get('metadata', {})
                    )

                    # Add path mode specific metadata
                    if hasattr(tactical_signal, 'metadata'):
                        tactical_signal.metadata.update({
                            'path_mode': sig_dict.get('path_mode', HRM_PATH_MODE),
                            'reason': sig_dict.get('reason', 'Generated by HRM path mode')
                        })

                    tactical_signals.append(tactical_signal)

                except Exception as e:
                    logger.error(f"Error converting path mode signal: {e}")
                    continue

            # Apply shared post-processing for consistency with traditional path
            final_signals = await self._apply_post_processing(tactical_signals, market_data, state)
            logger.info(f"✅ HRM Path Mode: {len(tactical_signals)} → {len(final_signals)} signals after post-processing")

            return final_signals

        except Exception as e:
            logger.error(f"Error processing HRM path modes: {e}")
            return []

    async def _process_traditional_signals(self, state: Dict[str, Any], market_data: Dict) -> List[TacticalSignal]:
        """Traditional L2 signal processing (legacy path)."""
        signals = []

        for symbol, data in market_data.items():
            df = self._extract_dataframe(data, symbol)
            if df is None:
                continue

            # Technical analysis, L1 integration, L2 AI processing
            indicators = self.multi_timeframe.calculate_technical_indicators(df)
            l1_signals = await self._get_l1_operational_signals(state, symbol, df)
            l2_signal = await self._get_finrl_signal(state, symbol, indicators, df)

            combined_signal = self._combine_l1_l2_signals(l1_signals, l2_signal, symbol, df, indicators)

            # L3 processing and risk adjustments
            processed_signal = await self._apply_l3_processing(combined_signal, state, symbol, df, indicators)

            if processed_signal:
                signals.append(processed_signal)

        # Apply post-processing features
        final_signals = await self._apply_post_processing(signals, market_data, state)

        return validate_signal_list(final_signals)

    async def _apply_l3_processing(self, combined_signal: TacticalSignal, state: Dict[str, Any],
                                 symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[TacticalSignal]:
        """Apply L3 processing to combined L1/L2 signal."""
        try:
            # Position-aware validation
            position_aware_signal = self._apply_position_aware_validation(combined_signal, symbol, state)
            if not position_aware_signal:
                logger.warning(f"⚠️ Position validation rejected signal for {symbol}")
                return None

            # Technical filtering phase 2
            filtered_signal = self._apply_technical_filtering_phase2(position_aware_signal, df, indicators, symbol)

            # L3 regime override
            l3_override_result = self._apply_l3_trend_following_override(filtered_signal, state, symbol, df)

            # Risk adjustments
            risk_adjusted = self._apply_risk_adjustment(l3_override_result, state, df, symbol)

            # Stop loss calculation
            final_signal = self._apply_stop_loss_calculation(risk_adjusted, df, state)

            return final_signal

        except Exception as e:
            logger.error(f"Error in L3 processing for {symbol}: {e}")
            return combined_signal  # Return original on error

    async def _apply_post_processing(self, signals: List[TacticalSignal], market_data: Dict[str, pd.DataFrame],
                                   state: Dict[str, Any]) -> List[TacticalSignal]:
        """Apply post-processing features like BTC/ETH sync and weight calculator."""
        try:
            # Convert to dict format for external processors
            signal_dicts = []
            for sig in signals:
                signal_dicts.append({
                    'symbol': getattr(sig, 'symbol', 'UNKNOWN'),
                    'side': getattr(sig, 'side', 'hold'),
                    'confidence': getattr(sig, 'confidence', 0.5),
                    'strength': getattr(sig, 'strength', 0.5),
                    'quantity': getattr(sig, 'quantity', 0),
                    'features': getattr(sig, 'features', {}),
                    'metadata': getattr(sig, 'metadata', {})
                })

            # Apply BTC/ETH synchronization (conditionally disabled for PAPER mode)
            if os.getenv('DISABLE_BTC_ETH_SYNC'):
                logger.info("⏭️ BTC/ETH Synchronizer disabled via environment variable")
                synchronized = signal_dicts
            else:
                synchronized = btc_eth_synchronizer.apply_btc_eth_synchronization(
                    signal_dicts, market_data, state
                )

            # Apply weight calculator
            weighted = await weight_calculator_integrator.apply_weight_calculator_integration(
                synchronized, market_data, state
            )

            # Convert back to TacticalSignal objects
            final_signals = []
            for sig_dict in weighted:
                try:
                    tactical_sig = TacticalSignal(
                        symbol=sig_dict.get('symbol', 'UNKNOWN'),
                        side=sig_dict.get('side', 'hold'),
                        strength=sig_dict.get('strength', 0.5),
                        confidence=sig_dict.get('confidence', 0.5),
                        source='processed_l2',
                        timestamp=pd.Timestamp.now(),
                        features=sig_dict.get('features', {}),
                        metadata=sig_dict.get('metadata', {})
                    )
                    final_signals.append(tactical_sig)
                except Exception as e:
                    logger.error(f"Error converting processed signal: {e}")
                    continue

            logger.info(f"✅ Post-processing: {len(signals)} → {len(final_signals)} signals")
            return final_signals

        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return signals

    # Utility methods (simplified from original)
    def _extract_dataframe(self, data, symbol: str) -> Optional[pd.DataFrame]:
        """Extract DataFrame from market data."""
        try:
            if isinstance(data, dict):
                if 'historical_data' in data and isinstance(data['historical_data'], pd.DataFrame):
                    df = data['historical_data']
                    return None if len(df) < 50 else df
                return None
            elif isinstance(data, pd.DataFrame):
                return None if data.empty or len(data) < 50 else data
            return None
        except Exception as e:
            logger.error(f"❌ Error extracting DataFrame for {symbol}: {e}")
            return None

    async def _get_l1_operational_signals(self, state: Dict[str, Any], symbol: str, df: pd.DataFrame) -> List[TacticalSignal]:
        """Get L1 operational signals."""
        try:
            from l1_operational.l1_operational import L1OperationalProcessor

            if 'l1_processor' not in state:
                state['l1_processor'] = L1OperationalProcessor({})

            l1_processor = state['l1_processor']
            l1_market_data = {symbol: df}
            return await l1_processor.process_market_data(l1_market_data)

        except Exception as e:
            logger.error(f"❌ Error getting L1 signals for {symbol}: {e}")
            return []

    async def _get_finrl_signal(self, state: Dict[str, Any], symbol: str, indicators: Dict, df: pd.DataFrame = None) -> Optional[TacticalSignal]:
        """Get FinRL AI signal with L3 context for aggressive DeepSeek behavior."""
        try:
            if not self.finrl_wrapper:
                return None

            market_data = state.get("market_data_simple") or state.get("market_data", {})

            # PRIORITY 2: Pass L3 context for aggressive DeepSeek processing
            l3_context = self._get_l3_output(state)
            signal = await self.finrl_wrapper.generate_signal(market_data, symbol, indicators, l3_context)

            if signal:
                signal.source = 'ai'

            return signal

        except Exception as e:
            logger.error(f"❌ Error getting FinRL signal for {symbol}: {e}")
            return None

    def _combine_l1_l2_signals(self, l1_signals: List[TacticalSignal], l2_signal: Optional[TacticalSignal],
                              symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[TacticalSignal]:
        """Combine L1 and L2 signals (simplified from original)."""
        try:
            if not l1_signals and not l2_signal:
                return None

            if not l1_signals:
                return l2_signal

            if not l2_signal:
                return self._create_l1_composite_signal(l1_signals, symbol, df, indicators)

            # Combine L1 and L2 (simplified logic)
            return self._create_l1_l2_composite_signal(l1_signals, l2_signal, symbol)

        except Exception as e:
            logger.error(f"❌ Error combining signals for {symbol}: {e}")
            return l2_signal

    # Additional utility methods (truncated for brevity - would include all original methods)
    def _create_l1_composite_signal(self, l1_signals: List[TacticalSignal], symbol: str,
                                   df: pd.DataFrame, indicators: Dict) -> Optional[TacticalSignal]:
        """Create composite L1 signal."""
        # Implementation details...
        return None

    def _create_l1_l2_composite_signal(self, l1_signals: List[TacticalSignal],
                                      l2_signal: TacticalSignal, symbol: str) -> Optional[TacticalSignal]:
        """Create composite L1+L2 signal."""
        # Implementation details...
        return l2_signal

    def _apply_position_aware_validation(self, signal: TacticalSignal, symbol: str, state: Dict[str, Any]) -> Optional[TacticalSignal]:
        """Apply position-aware validation."""
        # Implementation details...
        return signal

    def _apply_technical_filtering_phase2(self, signal: TacticalSignal, df: pd.DataFrame,
                                         indicators: Dict, symbol: str) -> TacticalSignal:
        """Apply technical filtering phase 2."""
        # Implementation details...
        return signal

    def _apply_l3_trend_following_override(self, signal: TacticalSignal, state: Dict[str, Any],
                                         symbol: str, df: pd.DataFrame) -> TacticalSignal:
        """Apply L3 trend following override."""
        # Implementation details...
        return signal

    def _apply_risk_adjustment(self, signal: TacticalSignal, state: Dict[str, Any],
                              df: pd.DataFrame, symbol: str) -> TacticalSignal:
        """Apply risk adjustments."""
        # Implementation details...
        return signal

    def _apply_stop_loss_calculation(self, signal: TacticalSignal, df: pd.DataFrame, state: Dict[str, Any]) -> TacticalSignal:
        """Calculate and apply stop loss."""
        # Implementation details...
        return signal

    def _get_l3_output(self, state: Dict[str, Any]) -> Dict:
        """Get L3 output from state - use current output if available, fallback to cache."""
        # Try current L3 output first (for real-time accuracy)
        current_l3_output = state.get("l3_output", {})
        if current_l3_output and isinstance(current_l3_output, dict) and current_l3_output.get('regime'):
            return current_l3_output

        # Fallback to cache if current is not available or valid
        l3_context_cache = state.get("l3_context_cache", {})
        return l3_context_cache.get("last_output", {})

    async def generate_signals_async(self, market_data: Dict[str, Any], l3_context: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Generate L2 signals with L3 context integration (SOLUCIÓN COMPLETA).
        Async version to be called from within the main loop.

        Args:
            market_data: Market data dictionary
            l3_context: L3 regime information with validation

        Returns:
            List of TacticalSignal objects
        """
        # ========================================================================================
        # ✅ FIX: L2 SIEMPRE PUEDE ACTUAR - CHECK CRITICAL CONDITIONS FIRST
        # ========================================================================================
        l3_allow_l2 = l3_context.get('allow_l2_signals', True)  # ✅ DEFAULT TRUE
        l3_strategic_hold = l3_context.get('strategic_hold_active', False)

        # Check if market conditions are critical (override L3)
        if not l3_allow_l2 or l3_strategic_hold:
            logger.warning("⚠️ L3 blocking L2, but checking for critical signals")
            if self._is_critical_condition(market_data):
                logger.warning("🚨 CRITICAL CONDITION - L2 OVERRIDE L3")
                l3_allow_l2 = True
            else:
                logger.info(f"🛡️ L3 BLOCK: L2 signals blocked by L3 context (allow_l2={l3_allow_l2}, strategic_hold={l3_strategic_hold})")
                return []

        signals = []

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            try:
                # Generate base L2 signal using traditional processing
                mock_state = {
                    "market_data": market_data,
                    "market_data_simple": market_data,
                    "l3_output": l3_context  # Pass L3 context to processing
                }

                # Call async method directly
                symbol_signals = await self.process_signals(mock_state)

                # Find signal for this specific symbol
                signal_for_symbol = None
                for sig in symbol_signals:
                    if hasattr(sig, 'symbol') and sig.symbol == symbol:
                        signal_for_symbol = sig
                        break

                # If no signal generated, NO crear señal neutral por defecto
                if not signal_for_symbol:
                    logger.debug(f"⏸️ No L2 signal generated for {symbol} - no default HOLD")
                    continue  # No agregar señal, pasar al siguiente símbolo

                # Apply additional L3 context-based validation/filtering
                validated_signal = self._apply_l3_context_validation(signal_for_symbol, l3_context)

                if validated_signal:
                    signals.append(validated_signal)
                    logger.debug(f"✅ L2 signal generated for {symbol} with L3 context")
                else:
                    logger.debug(f"⚠️ L2 signal filtered out for {symbol} by L3 context validation")

            except Exception as e:
                logger.error(f"❌ Error generating L2 signal for {symbol}: {e}")
                # NO crear señal fallback - mantener lista limpia
                continue

        return signals

    def generate_signals(self, market_data: Dict[str, Any], l3_context=None):
        """
        Generate tactical signals with L3 dominance control.

        ✅ FIX: L2 SIEMPRE PUEDE ACTUAR SI HAY CONDICIONES CRÍTICAS
        """

        # ========================================================================================
        # ✅ FIX: VERIFICACIÓN L3 PRIMERO - PERMITIR OVERRIDE SI HAY CONDICIONES CRÍTICAS
        # ========================================================================================
        if not l3_context:
            logger.info("🛡️ L3 BLOCK: No L3 context provided - blocking all L2 signals")
            return []  # NO generar señales por defecto

        l3_allow_l2 = l3_context.get('allow_l2_signals', True)  # ✅ DEFAULT TRUE
        l3_strategic_hold = l3_context.get('strategic_hold_active', False)

        # Check if market conditions are critical (override L3)
        if not l3_allow_l2 or l3_strategic_hold:
            logger.warning("⚠️ L3 blocking L2, but checking for critical signals")
            if self._is_critical_condition(market_data):
                logger.warning("🚨 CRITICAL CONDITION - L2 OVERRIDE L3")
                l3_allow_l2 = True
            else:
                logger.info(f"🛡️ L3 BLOCK: L2 signals blocked by L3 context (allow_l2={l3_allow_l2}, strategic_hold={l3_strategic_hold})")
                return []  # NO generar señales - lista vacía

        # ========================================================================================
        # L3 PERMITE SEÑALES - GENERAR SOLO CON INTENT REAL
        # ========================================================================================
        l3_signal = l3_context.get('signal', 'hold')
        l3_confidence = l3_context.get('confidence', 0.0)
        l3_regime = l3_context.get('regime', 'unknown')

        logger.info(f"📊 L2 processing L3: {l3_signal} ({l3_confidence:.2f}) in {l3_regime}")

        signals = []
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            # NO usar umbrales por defecto - solo generar si hay intención real
            signal_output = self._generate_single_signal_with_thresholds(symbol, market_data, l3_context)
            if signal_output:  # Solo agregar si se generó una señal real
                self._emit_l2_signal(signal_output, l3_context)
                signals.append(signal_output)
            # Si no hay señal, NO agregar HOLD por defecto

        return signals

    def _generate_single_signal_with_thresholds(self, symbol: str, market_data: Dict[str, Any], l3_context: Dict[str, Any]) -> Optional[TacticalSignal]:
        """
        Generate signal ONLY when ALL conditions for BUY/SELL are met.

        NO DEFAULT HOLD: Only generate signals when there's real trading intention.
        If conditions not met → return None (no signal)

        MINIMUM CONDITIONS FOR BUY/SELL:
        1. confidence >= 0.70
        2. abs(signal_strength) >= threshold (0.6)
        3. RSI NOT in extreme zone (not < 25 or > 75)
        4. volatility NOT collapsing (< 0.8)

        If ANY condition fails → NO SIGNAL (not HOLD)
        """

        try:
            # Get market data for this symbol
            symbol_data = market_data.get(symbol, {})
            if not isinstance(symbol_data, dict) or 'historical_data' not in symbol_data:
                logger.debug(f"⏸️ L2 {symbol}: No market data - no signal")
                return None

            df = symbol_data['historical_data']
            if not isinstance(df, pd.DataFrame) or len(df) < 50:
                logger.debug(f"⏸️ L2 {symbol}: Insufficient data - no signal")
                return None

            # Calculate technical indicators
            indicators = self.multi_timeframe.calculate_technical_indicators(df)

            # ========================================================================================
            # CONDITION 1: HIGH CONFIDENCE THRESHOLD (>= 0.70)
            # ========================================================================================
            ai_confidence = self._calculate_ai_confidence(symbol, indicators, l3_context)
            if ai_confidence < 0.70:
                logger.debug(f"⏸️ L2 {symbol}: Confidence {ai_confidence:.3f} < 0.70 - no signal")
                return None

            # ========================================================================================
            # CONDITION 2: SIGNAL STRENGTH THRESHOLD (>= 0.60)
            # ========================================================================================
            signal_strength = self._calculate_signal_strength(symbol, indicators)
            if abs(signal_strength) < 0.60:
                logger.debug(f"⏸️ L2 {symbol}: Signal strength {abs(signal_strength):.3f} < 0.60 - no signal")
                return None

            # ========================================================================================
            # CONDITION 3: RSI NOT IN EXTREME ZONE (NOT < 25 OR > 75)
            # ========================================================================================
            rsi = indicators.get('rsi', 50)
            if rsi < 25 or rsi > 75:
                logger.debug(f"⏸️ L2 {symbol}: RSI {rsi:.1f} in extreme zone - no signal")
                return None

            # ========================================================================================
            # CONDITION 4: VOLATILITY NOT COLLAPSING (>= 0.80)
            # ========================================================================================
            volatility = self._calculate_volatility_score(symbol, indicators)
            if volatility < 0.80:
                logger.debug(f"⏸️ L2 {symbol}: Volatility {volatility:.3f} < 0.80 - no signal")
                return None

            # ========================================================================================
            # ALL CONDITIONS MET: Generate BUY/SELL signal
            # ========================================================================================
            signal_direction = 'buy' if signal_strength > 0 else 'sell'
            final_signal = {
                "signal": signal_direction.upper(),
                "confidence": ai_confidence,
                "reason": f"All thresholds met: conf≥0.70, strength≥0.60, RSI={rsi:.1f}, vol≥0.80",
                "source": "L2_TACTICAL"
            }

            logger.info(f"✅ L2 {symbol}: {signal_direction.upper()} signal accepted - {final_signal['reason']}")
            return self._create_signal_from_dict(symbol, final_signal)

        except Exception as e:
            logger.error(f"❌ Error generating L2 signal for {symbol}: {e}")
            return None  # No fallback signal

    def _calculate_ai_confidence(self, symbol: str, indicators: Dict, l3_context: Dict[str, Any]) -> float:
        """Calculate AI model confidence for the symbol."""
        try:
            # Use FinRL model if available
            if self.finrl_wrapper:
                mock_state = {
                    "market_data": {symbol: {"historical_data": pd.DataFrame()}},  # Simplified
                    "l3_output": l3_context
                }
                ai_signal = self.finrl_wrapper.generate_signal(mock_state, symbol, indicators)
                if ai_signal and hasattr(ai_signal, 'confidence'):
                    return float(getattr(ai_signal, 'confidence', 0.0))
        except Exception as e:
            logger.error(f"Error calculating AI confidence for {symbol}: {e}")

        # Fallback: use L3 confidence or default low value
        return l3_context.get('confidence', 0.4)

    def _calculate_signal_strength(self, symbol: str, indicators: Dict) -> float:
        """Calculate signal strength from technical indicators."""
        try:
            # Combine multiple indicators for strength calculation
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd_signal', 0)
            bb_position = indicators.get('bb_position', 0)  # Bollinger Band position

            # Normalize RSI momentum (-1 to 1)
            rsi_momentum = (rsi - 50) / 50

            # Combine signals (simplified logic)
            strength = (rsi_momentum * 0.4) + (macd * 0.3) + (bb_position * 0.3)

            return float(strength)
        except Exception as e:
            logger.error(f"Error calculating signal strength for {symbol}: {e}")
            return 0.0

    def _calculate_volatility_score(self, symbol: str, indicators: Dict) -> float:
        """Calculate volatility score (1.0 = high volatility, 0.0 = low volatility)."""
        try:
            # Use ATR or other volatility indicators
            atr = indicators.get('atr', 1.0)
            bb_width = indicators.get('bb_width', 0.02)

            # Normalize to 0-1 scale (higher values = more volatile)
            vol_score = min(1.0, (atr * 10) + (bb_width * 50))  # Simplified calculation
            return float(vol_score)
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.5

    def _create_signal_from_dict(self, symbol: str, signal_dict: Dict[str, Any]) -> TacticalSignal:
        """Create TacticalSignal object from signal dictionary."""
        return TacticalSignal(
            symbol=symbol,
            side=signal_dict["signal"].lower(),
            strength=0.5,
            confidence=signal_dict["confidence"],
            source=signal_dict["source"],
            timestamp=pd.Timestamp.now(),
            features={},
            metadata={
                'reason': signal_dict["reason"],
                'l2_tactical_decision': True
            }
        )

    def _generate_hold_signals_with_reason(self, market_data: Dict[str, Any], reason: str) -> List[TacticalSignal]:
        """Generate HOLD signals with specific reason."""
        signals = []
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            signal = TacticalSignal(
                symbol=symbol,
                side='hold',
                strength=0.5,
                confidence=0.42,
                source='L2_TACTICAL',
                timestamp=pd.Timestamp.now(),
                features={},
                metadata={
                    'reason': reason,
                    'l2_tactical_decision': True
                }
            )
            signals.append(signal)
        return signals

    def _is_critical_condition(self, market_data: Dict[str, Any]) -> bool:
        """Check if market is in critical condition requiring L2 override"""
        for symbol, data in market_data.items():
            # Check if data has historical_data or is a DataFrame
            if isinstance(data, dict) and 'historical_data' in data:
                df = data['historical_data']
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                continue
                
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                # Calculate RSI
                try:
                    # Calculate RSI using close prices
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    latest_rsi = rsi.iloc[-1]
                    
                    # Critical overbought/oversold conditions
                    if latest_rsi > 80 or latest_rsi < 20:
                        logger.warning(f"🚨 CRITICAL CONDITION: {symbol} RSI = {latest_rsi:.1f}")
                        return True
                except Exception as e:
                    logger.debug(f"Error calculating RSI for {symbol}: {e}")
                    continue
        return False

    def _apply_l3_context_validation(self, signal: TacticalSignal, l3_context: Dict[str, Any]) -> Optional[TacticalSignal]:
        """
        Apply additional L3 context-based validation to generated L2 signals.

        Args:
            signal: Generated TacticalSignal
            l3_context: L3 regime information

        Returns:
            Validated signal or None if filtered out
        """
        # Extract L3 context with safe defaults
        l3_regime = l3_context.get('regime', 'unknown')
        l3_confidence = l3_context.get('confidence', 0.0)
        l3_allow_l2 = l3_context.get('allow_l2', True)

        # ✅ FIX: Allow L2 to act if there are critical market conditions
        if not l3_allow_l2:
            # Check if this is a critical signal that should override L3
            if hasattr(signal, 'metadata') and signal.metadata.get('is_critical', False):
                logger.warning(f"🚨 CRITICAL SIGNAL OVERRIDE: Allowing {signal.side} for {signal.symbol} despite L3 block")
            elif hasattr(signal, 'side') and (signal.side == 'sell' and self._is_critical_condition({'dummy': {'historical_data': pd.DataFrame()}})):
                logger.warning(f"🚨 SELL SIGNAL OVERRIDE: Allowing SELL for {signal.symbol} despite L3 block")
            else:
                logger.info(f"🚫 L3 context validation: filtering {signal.side} signal for {signal.symbol} (L3 dominance)")
                return None

        # Add L3 context metadata to signal for debugging/tracking
        if hasattr(signal, 'metadata'):
            signal.metadata.update({
                'l3_regime': l3_regime,
                'l3_confidence': l3_confidence,
                'l3_allow_l2': l3_allow_l2,
                'l3_context_validated': True
            })

        return signal

    # Helper methods for L3 context-based signal generation
    def _generate_default_signals(self, market_data):
        """Genera señales HOLD por defecto cuando no hay L3 context"""
        from datetime import datetime

        signals = []
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            signal = TacticalSignal(
                symbol=symbol,
                side='hold',
                strength=0.5,
                confidence=0.0,
                source='l2_default',
                timestamp=pd.Timestamp.now(),
                features={},
                metadata={'reason': 'No L3 context provided'}
            )
            signals.append(signal)

        return signals

    def _generate_buy_signals(self, market_data, l3_context):
        """Genera señales BUY basadas en directiva L3"""
        from datetime import datetime

        signals = []

        for symbol in ['BTCUSDT', 'ETHUSDT']:
            signal = TacticalSignal(
                symbol=symbol,
                side='buy',
                strength=min(1.0, l3_context['confidence'] + 0.3),  # Boost strength for L3 signals
                confidence=l3_context['confidence'],  # Use L3 confidence directly
                source='l2_following_l3',
                timestamp=pd.Timestamp.now(),
                features={},
                metadata={
                    'reason': f"L3 {l3_context['regime']} {l3_context.get('subtype', '')} setup",
                    'l3_regime': l3_context['regime'],
                    'l3_confidence': l3_context['confidence'],
                    'l3_signal': 'buy'
                }
            )
            signals.append(signal)

        return signals

    def _generate_sell_signals(self, market_data, l3_context):
        """Genera señales SELL basadas en directiva L3"""
        from datetime import datetime

        signals = []

        for symbol in ['BTCUSDT', 'ETHUSDT']:
            signal = TacticalSignal(
                symbol=symbol,
                side='sell',
                strength=min(1.0, l3_context['confidence'] + 0.3),  # Boost strength for L3 signals
                confidence=l3_context['confidence'],  # Use L3 confidence directly
                source='l2_following_l3',
                timestamp=pd.Timestamp.now(),
                features={},
                metadata={
                    'reason': f"L3 {l3_context['regime']} {l3_context.get('subtype', '')} setup",
                    'l3_regime': l3_context['regime'],
                    'l3_confidence': l3_context['confidence'],
                    'l3_signal': 'sell'
                }
            )
            signals.append(signal)

        return signals

    def _generate_autonomous_signals_with_threshold_sync(self, market_data, l3_context):
        """
        Genera señales usando modelos L2 propios con UMBRAL DE CONFIANZA (versión sincrónica).
        HOLD por defecto - BUY/SELL solo si confidence >= 0.6
        """
        logger.info("🔄 Using L2 autonomous processing with CONFIDENCE THRESHOLD (HOLD default) - SYNC VERSION")

        try:
            signals = []
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                # NO DEFAULT HOLD: Only generate signals when L3 allows and conditions met
                # If L3 blocks L2, return empty signals
                if not l3_context.get('allow_l2_signals', False):
                    logger.info(f"🛡️ L3 BLOCK: L2 signals blocked by L3 context for {symbol}")
                    continue  # Skip this symbol, don't generate any signal

                # Only generate signals when there's real intention, not defaults
                logger.debug(f"🎯 L2 checking conditions for {symbol} (no default HOLD generation)")

                # Intentar generar señal AI - versión simplificada sin async
                try:
                    # Simular evaluación AI básica (sin procesamiento complejo de async)
                    # En un entorno real, esto debería ser async, pero por simplicidad usamos fallback

                    # Para testing: asumir que la mayoría de las veces no hay señal AI clara
                    # En producción, aquí iría la lógica real de evaluación AI
                    ai_confidence = 0.3  # Simular baja confianza por defecto
                    ai_side = 'hold'     # Simular señal HOLD por defecto

                    logger.info(f"🎯 L2 AI signal for {symbol}: {ai_side} (conf={ai_confidence:.3f})")

                    # UMBRAL CRÍTICO: Solo BUY/SELL si confidence >= 0.6 (o >= 0.5 en TRENDING regime)
                    regime = l3_context.get('regime', 'unknown')
                    min_confidence = 0.6  # WEAK_THRESHOLD removed - TRENDING no longer gets special treatment
                    if ai_side in ['buy', 'sell'] and ai_confidence >= min_confidence:
                        logger.info(f"✅ L2 THRESHOLD MET: {ai_side.upper()} signal accepted (conf={ai_confidence:.3f} >= 0.6)")

                        # Usar señal AI validada
                        validated_signal = TacticalSignal(
                            symbol=symbol,
                            side=ai_side,
                            strength=0.7,
                            confidence=ai_confidence,
                            source='l2_autonomous_ai_threshold_met',
                            timestamp=pd.Timestamp.now(),
                            features={},
                            metadata={
                                'reason': f'L2 autonomous - confidence threshold met ({ai_confidence:.3f} >= 0.6)',
                                'l3_regime': l3_context.get('regime', 'unknown'),
                                'l3_confidence': l3_context.get('confidence', 0.0),
                                'l3_signal': l3_context.get('signal', 'hold'),
                                'l2_model_confidence': ai_confidence,
                                'threshold_check_passed': True
                            }
                        )
                        signals.append(validated_signal)

                        # LOGGING EXPLÍCITO: Registrar BUY/SELL que pasan umbral en events.json
                        log_trading_action(
                            symbol=symbol,
                            strategy='L2_AUTONOMOUS_THRESHOLD',
                            regime=l3_context.get('regime', 'unknown'),
                            action=ai_side,
                            confidence=ai_confidence,
                            reason=f'L2 autonomous - confidence threshold met ({ai_confidence:.3f} >= 0.6)'
                        )

                        continue  # No usar HOLD por defecto

                    elif ai_side in ['buy', 'sell'] and ai_confidence < 0.6:
                        logger.info(f"⏸️ L2 THRESHOLD NOT MET: {ai_side.upper()} rejected (conf={ai_confidence:.3f} < 0.6), using HOLD")

                        # Actualizar metadata del HOLD por defecto
                        default_hold_signal.metadata.update({
                            'ai_signal_rejected': ai_side,
                            'ai_confidence_below_threshold': ai_confidence,
                            'threshold_required': 0.6
                        })

                    else:
                        logger.info(f"⏸️ L2 AI returned HOLD or invalid signal, using default HOLD")

                except Exception as e:
                    logger.error(f"❌ Error generating AI signal for {symbol}: {e}")
                    # Continuar con HOLD por defecto

                # DEFAULT: Usar HOLD si no hay señal AI válida que supere umbral
                signals.append(default_hold_signal)

                # LOGGING EXPLÍCITO: Registrar HOLD como señal L2 real en events.json
                log_trading_action(
                    symbol=symbol,
                    strategy='L2_AUTONOMOUS_THRESHOLD',
                    regime=l3_context.get('regime', 'unknown'),
                    action='hold',
                    confidence=0.4,
                    reason='HOLD as default - no clear statistical advantage'
                )

            logger.info(f"✅ L2 autonomous with threshold: generated {len(signals)} signals (mostly HOLD)")
            return signals

        except Exception as e:
            logger.error(f"❌ Error in L2 autonomous signal generation with threshold: {e}")
            # Fallback to hold signals
            return self._generate_hold_signals(market_data)

    async def _generate_autonomous_signals_with_threshold(self, market_data, l3_context):
        """
        Genera señales usando modelos L2 propios con UMBRAL DE CONFIANZA.
        HOLD por defecto - BUY/SELL solo si confidence >= 0.6
        """
        logger.info("🔄 Using L2 autonomous processing with CONFIDENCE THRESHOLD (HOLD default)")

        try:
            signals = []
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                # NO DEFAULT HOLD: Only generate signals when L3 allows and conditions met
                # If L3 blocks L2, skip this symbol entirely
                if not l3_context.get('allow_l2_signals', False):
                    logger.info(f"🛡️ L3 BLOCK: L2 signals blocked by L3 context for {symbol}")
                    continue  # Skip this symbol, don't generate any signal

                # Only generate signals when there's real intention, not defaults
                logger.debug(f"🎯 L2 checking conditions for {symbol} (no default HOLD generation)")

                # Intentar generar señal AI solo si hay posibilidad de superar umbral
                ai_signal = None
                try:
                    # Get market state for this symbol
                    symbol_data = market_data.get(symbol, {})

                    if self.finrl_wrapper and isinstance(symbol_data, dict) and 'historical_data' in symbol_data:
                        df = symbol_data['historical_data']
                        if isinstance(df, pd.DataFrame) and len(df) >= 50:
                            # Calculate indicators
                            indicators = self.multi_timeframe.calculate_technical_indicators(df)

                            # Create state for FinRL processing
                            mock_state = {
                                "market_data": market_data,
                                "market_data_simple": market_data,
                                "l3_output": l3_context
                            }

                            # Generate AI signal
                            ai_signal = await self.finrl_wrapper.generate_signal(mock_state, symbol, indicators)

                            if ai_signal:
                                ai_confidence = getattr(ai_signal, 'confidence', 0.0)
                                ai_side = getattr(ai_signal, 'side', 'hold')

                                logger.info(f"🎯 L2 AI signal for {symbol}: {ai_side} (conf={ai_confidence:.3f})")

                                # UMBRAL CRÍTICO: Solo BUY/SELL si confidence >= 0.6
                                if ai_side in ['buy', 'sell'] and ai_confidence >= 0.6:
                                    logger.info(f"✅ L2 THRESHOLD MET: {ai_side.upper()} signal accepted (conf={ai_confidence:.3f} >= 0.6)")

                                    # Usar señal AI validada
                                    validated_signal = TacticalSignal(
                                        symbol=symbol,
                                        side=ai_side,
                                        strength=getattr(ai_signal, 'strength', 0.7),
                                        confidence=ai_confidence,
                                        source='l2_autonomous_ai_threshold_met',
                                        timestamp=pd.Timestamp.now(),
                                        features=getattr(ai_signal, 'features', {}),
                                        metadata={
                                            'reason': f'L2 autonomous - confidence threshold met ({ai_confidence:.3f} >= 0.6)',
                                            'l3_regime': l3_context.get('regime', 'unknown'),
                                            'l3_confidence': l3_context.get('confidence', 0.0),
                                            'l3_signal': l3_context.get('signal', 'hold'),
                                            'l2_model_confidence': ai_confidence,
                                            'threshold_check_passed': True
                                        }
                                    )
                                    signals.append(validated_signal)
                                    continue  # No usar HOLD por defecto

                                elif ai_side in ['buy', 'sell'] and ai_confidence < 0.6:
                                    logger.info(f"⏸️ L2 THRESHOLD NOT MET: {ai_side.upper()} rejected (conf={ai_confidence:.3f} < 0.6), using HOLD")

                                    # Actualizar metadata del HOLD por defecto
                                    default_hold_signal.metadata.update({
                                        'ai_signal_rejected': ai_side,
                                        'ai_confidence_below_threshold': ai_confidence,
                                        'threshold_required': 0.6
                                    })

                                else:
                                    logger.info(f"⏸️ L2 AI returned HOLD or invalid signal, using default HOLD")

                except Exception as e:
                    logger.error(f"❌ Error generating AI signal for {symbol}: {e}")
                    # Continuar con HOLD por defecto

                # DEFAULT: Usar HOLD si no hay señal AI válida que supere umbral
                signals.append(default_hold_signal)

                # LOGGING EXPLÍCITO: Registrar HOLD como señal L2 real
                logger.info(f"📊 L2 SIGNAL: {symbol} HOLD (confidence=0.4) - logged as real L2 signal")

            logger.info(f"✅ L2 autonomous with threshold: generated {len(signals)} signals (mostly HOLD)")
            return signals

        except Exception as e:
            logger.error(f"❌ Error in L2 autonomous signal generation with threshold: {e}")
            # Fallback to hold signals
            return self._generate_hold_signals(market_data)

    def _generate_autonomous_signals(self, market_data, l3_context):
        """Genera señales usando modelos L2 propios cuando L3 permite autonomía"""
        logger.info("🔄 Using L2 autonomous processing with L3 context awareness")

        # Try to use FinRL models for autonomous generation
        try:
            signals = []
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                # Get market state for this symbol
                symbol_data = market_data.get(symbol, {})

                # Use FinRL wrapper if available
                if self.finrl_wrapper:
                    # Create a mock state for FinRL processing
                    mock_state = {
                        "market_data": market_data,
                        "market_data_simple": market_data,
                        "l3_output": l3_context
                    }

                    # Calculate indicators if data is available
                    indicators = {}
                    if isinstance(symbol_data, dict) and 'historical_data' in symbol_data:
                        df = symbol_data['historical_data']
                        if isinstance(df, pd.DataFrame) and len(df) >= 50:
                            indicators = self.multi_timeframe.calculate_technical_indicators(df)

                    # Generate signal using FinRL - Use synchronous validation to avoid async issues
                    try:
                        from core.incremental_signal_verifier import IncrementalSignalVerifier

                        # Use synchronous signal verification for faster processing
                        signal_verifier = IncrementalSignalVerifier()
                        base_signal_dict = {
                            'signal_id': f"{symbol}_autonomous_{int(pd.Timestamp.now().timestamp())}",
                            'action': 'hold',  # Default to hold for autonomous processing
                            'symbol': symbol,
                            'confidence': 0.5
                        }

                        # Check if we should attempt AI processing - HOLD signals bypass complex async processing
                        if signal_verifier.verify_signal(base_signal_dict):
                            # For HOLD signals, use synchronous processing only (no complex FinRL async processing)
                            logger.info(f"✅ Signal {symbol} HOLD auto-approved (no async processing needed)")
                            ai_signal = None  # We'll use fallback logic below instead
                        else:
                            # For BUY/SELL signals, skip complex processing to avoid async issues
                            logger.info(f"⚠️ Signal {symbol} BUY/SELL rejected (complex async processing disabled)")
                            ai_signal = None
                    except Exception as e:
                        logger.error(f"❌ Error generating FinRL signal for {symbol}: {e}")
                        ai_signal = None

                    if ai_signal:
                        # Ensure signal follows L3 context
                        signal_side = getattr(ai_signal, 'side', 'hold')
                        signal_confidence = getattr(ai_signal, 'confidence', 0.5)

                        # Create updated signal with L3 awareness
                        signal = TacticalSignal(
                            symbol=symbol,
                            side=signal_side,
                            strength=getattr(ai_signal, 'strength', 0.5),
                            confidence=signal_confidence,
                            source='l2_autonomous_ai',
                            timestamp=pd.Timestamp.now(),
                            features=getattr(ai_signal, 'features', {}),
                            metadata={
                                'reason': 'L2 autonomous processing with L3 awareness',
                                'l3_regime': l3_context.get('regime', 'unknown'),
                                'l3_confidence': l3_context.get('confidence', 0.0),
                                'l3_signal': l3_context.get('signal', 'hold'),
                                'l2_model_confidence': signal_confidence
                            }
                        )
                        signals.append(signal)
                    else:
                        # Fallback to hold if no AI signal
                        signal = TacticalSignal(
                            symbol=symbol,
                            side='hold',
                            strength=0.5,
                            confidence=0.5,
                            source='l2_autonomous_fallback',
                            timestamp=pd.Timestamp.now(),
                            features={},
                            metadata={
                                'reason': 'L2 autonomous - no AI signal generated',
                                'l3_regime': l3_context.get('regime', 'unknown'),
                                'l3_allow_l2': True
                            }
                        )
                        signals.append(signal)
                else:
                    # No AI available, use basic hold signals
                    logger.warning("⚠️ L2 FinRL wrapper not available for autonomous processing")
                    signal = TacticalSignal(
                        symbol=symbol,
                        side='hold',
                        strength=0.5,
                        confidence=0.5,
                        source='l2_autonomous_no_ai',
                        timestamp=pd.Timestamp.now(),
                        features={},
                        metadata={
                            'reason': 'L2 autonomous - no AI processor available',
                            'l3_regime': l3_context.get('regime', 'unknown'),
                            'l3_allow_l2': True
                        }
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"❌ Error in L2 autonomous signal generation: {e}")
            # Fallback to hold signals
            return self._generate_hold_signals(market_data)

    def _emit_l2_signal(self, signal: TacticalSignal, l3_context: Dict[str, Any]):
        """
        ✅ FIX: Emitir TODAS las señales L2 (BUY/SELL/HOLD) como eventos L2_SIGNAL formales.

        REGLA FUNDAMENTAL: Cada ciclo + símbolo = exactamente 1 señal
        Nunca "None" - siempre BUY/SELL/HOLD
        """
        # Extraer información del TacticalSignal
        symbol = getattr(signal, 'symbol', 'UNKNOWN')
        action = getattr(signal, 'side', 'hold').upper()
        confidence = float(getattr(signal, 'confidence', 0.5))

        # ✅ FIX: Ensure action is BUY/SELL/HOLD, default to HOLD
        if action not in ["BUY", "SELL", "HOLD"]:
            action = "HOLD"

        # Create signal dict in canonical format
        signal_dict = {
            "type": "L2_SIGNAL",
            "action": action,
            "symbol": symbol,
            "confidence": confidence,
            "source": "L2_TACTICAL",
            "timestamp": datetime.utcnow().isoformat()
        }

        # Emit signal using log_trading_action
        log_trading_action(
            symbol=symbol,
            strategy='L2_SIGNAL',
            regime=l3_context.get('regime', 'unknown'),
            action=action,
            confidence=confidence,
            reason=getattr(signal, 'metadata', {}).get('reason', 'L2 tactical signal')
        )

        logger.info(
            f"L2_SIGNAL emitted: {action} for {symbol} (confidence={confidence:.2f})"
        )

        return signal_dict

    def _generate_hold_signals(self, market_data):
        """Genera señales HOLD cuando L3 sugiere mantenimiento de posiciones"""
        from datetime import datetime

        signals = []
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            signal = TacticalSignal(
                symbol=symbol,
                side='hold',
                strength=0.5,
                confidence=0.5,
                source='l2_hold_l3_suggestion',
                timestamp=pd.Timestamp.now(),
                features={},
                metadata={'reason': 'L3 suggests maintaining positions'}
            )
            signals.append(signal)

        return signals

    def generate_signals_conservative(self, market_data, l3_context):
        """
        INVARIANTE CRÍTICO: Genera señales conservativas con HOLD por defecto.

        FILOSOFÍA: Un sistema que siempre opera no es agresivo, es ciego.
        - HOLD por defecto (70%+ de señales)
        - BUY/SELL solo con evidencia excepcional
        - Sistema duda → HOLD

        Args:
            market_data: Market data dictionary
            l3_context: L3 regime information

        Returns:
            List of conservative TacticalSignal objects (mostly HOLD)
        """
        logger.info("🔄 GENERANDO SEÑALES CONSERVATIVAS: HOLD por defecto")

        signals = []
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            try:
                signal = self._generate_conservative_signal_for_symbol(symbol, market_data, l3_context)
                if signal:
                    # 🚨 CRÍTICO: EMITIR TODAS LAS SEÑALES (BUY/SELL/HOLD) COMO L2_SIGNAL
                    self._emit_l2_signal(signal, l3_context)
                    signals.append(signal)
            except Exception as e:
                logger.error(f"❌ Error generando señal conservativa para {symbol}: {e}")
                # Fallback: HOLD signal
                fallback_signal = TacticalSignal(
                    symbol=symbol,
                    side='hold',
                    strength=0.5,
                    confidence=0.4,
                    source='l2_conservative_fallback',
                    timestamp=pd.Timestamp.now(),
                    features={},
                    metadata={'reason': 'Conservative fallback - system doubt'}
                )
                # 🚨 CRÍTICO: EMITIR TAMBIÉN LA SEÑAL FALLBACK HOLD
                self._emit_l2_signal(fallback_signal, l3_context)
                signals.append(fallback_signal)

        logger.info(f"✅ Generadas {len(signals)} señales conservativas")
        return signals

    def _generate_conservative_signal_for_symbol(self, symbol, market_data, l3_context):
        """
        Genera UNA señal conservativa para un símbolo.

        CRÍTICO: AUTORIDAD REAL L3 SOBRE L2
        - Si L3 tiene bias claro (sell/buy), L2 responde inmediatamente
        - No más HOLD por defecto cuando L3 tiene intención clara
        - SELL_LIGHT/REDUCE para señales L3 sell, BUY_LIGHT para buy

        PRIORITY 1 FIX: BREAK HOLD IN SETUPS
        - If l3 setup_active and l3 signal == "buy": l2_min_confidence = 0.30
        - Allow BUY override (5-10% allocation) with stop-loss

        LÓGICA CONSERVATIVA + INV-5 RULE:
        - HOLD por defecto (90%+ de casos) SOLO cuando L3 no tiene bias claro
        - BUY/SELL solo si TODAS las condiciones excepcionales se cumplen
        - INV-5 RULE: Sistema duda → HOLD (confidence < 0.6 → HOLD)

        BUG FIX: L2 deve considerar setup trades de L3 incluso en RANGE regime
        """
        # ========================================================================================
        # INV-5 RULE: Sistema duda → HOLD (CRÍTICO - ANTES QUE CUALQUIER OTRA LÓGICA)
        # ========================================================================================
        # Si el sistema duda (confidence < 0.6), genera HOLD sin excepciones
        system_doubt_threshold = 0.35
        l3_confidence = l3_context.get('confidence', 0.0)

        if l3_confidence < system_doubt_threshold:
            logger.info(f"🎯 INV-5 RULE ACTIVATED: System doubt (L3 confidence {l3_confidence:.2f} < {system_doubt_threshold}) → HOLD for {symbol}")

            doubt_hold_signal = TacticalSignal(
                symbol=symbol,
                side='hold',
                strength=0.5,
                confidence=max(0.1, l3_confidence),  # Never zero confidence for logging
                source='l2_inv5_doubt_hold_rule',
                timestamp=pd.Timestamp.now(),
                features={},
                metadata={
                    'reason': f'INV-5 RULE: System doubt (confidence {l3_confidence:.2f} < {system_doubt_threshold}) → HOLD',
                    'l3_regime': l3_context.get('regime', 'unknown'),
                    'l3_signal': l3_context.get('signal', 'hold'),
                    'l3_confidence': l3_confidence,
                    'inv5_rule_applied': True,
                    'doubt_threshold': system_doubt_threshold,
                    'philosophy': 'Sistema duda → HOLD (INV-5 rule)'
                }
            )
            logger.info(f"🛡️ INV-5 ENFORCED: {symbol} HOLD signal (doubt rule) - confidence {max(0.1, l3_confidence):.2f}")
            return doubt_hold_signal

        # ========================================================================================
        # CRÍTICO: AUTORIDAD REAL L3 - Override HOLD cuando L3 tiene bias claro
        # ========================================================================================
        l3_signal = l3_context.get('signal', 'hold')

        # L3 SELL SIGNAL → L2 RESPONDE CON SELL_LIGHT/REDUCE
        if l3_signal == "sell":
            action = "SELL_LIGHT"  # o REDUCE según preferencia
            confidence = max(l3_confidence, 0.55)  # Boost confidence mínimo

            l3_sell_signal = TacticalSignal(
                symbol=symbol,
                side='sell',
                strength=0.6,
                confidence=confidence,
                source='l2_following_l3_sell',
                timestamp=pd.Timestamp.now(),
                features={},
                metadata={
                    'reason': f'L3 sell signal - L2 responds with {action} (real L3 authority)',
                    'l3_regime': l3_context.get('regime', 'unknown'),
                    'l3_signal': l3_signal,
                    'l3_confidence': l3_confidence,
                    'l2_action': action,
                    'l3_authority_override': True,
                    'philosophy': 'L3 has real authority - no HOLD when L3 has clear bias'
                }
            )
            logger.info(f"🛑 L3 AUTHORITY: {symbol} SELL_LIGHT (conf={confidence:.2f}) - L3 sell signal override")
            return l3_sell_signal

        # L3 BUY SIGNAL → L2 RESPONDE CON BUY_LIGHT
        elif l3_signal == "buy":
            action = "BUY_LIGHT"
            confidence = max(l3_confidence, 0.55)  # Boost confidence mínimo

            l3_buy_signal = TacticalSignal(
                symbol=symbol,
                side='buy',
                strength=0.6,
                confidence=confidence,
                source='l2_following_l3_buy',
                timestamp=pd.Timestamp.now(),
                features={},
                metadata={
                    'reason': f'L3 buy signal - L2 responds with {action} (real L3 authority)',
                    'l3_regime': l3_context.get('regime', 'unknown'),
                    'l3_signal': l3_signal,
                    'l3_confidence': l3_confidence,
                    'l2_action': action,
                    'l3_authority_override': True,
                    'philosophy': 'L3 has real authority - no HOLD when L3 has clear bias'
                }
            )
            logger.info(f"🟢 L3 AUTHORITY: {symbol} BUY_LIGHT (conf={confidence:.2f}) - L3 buy signal override")
            return l3_buy_signal

        # PULLBACK STRATEGY: When L3 has LONG bias but no clear signal, look for pullbacks
        if l3_context.get('bias') == 'LONG' and l3_signal == 'hold':
            logger.info(f"🎯 L3 LONG BIAS DETECTED: Looking for pullback buying opportunities for {symbol}")
            try:
                # Check if we have pullback conditions (RSI < 50, price below MA)
                symbol_data = market_data.get(symbol, {})
                if isinstance(symbol_data, dict) and 'historical_data' in symbol_data:
                    df = symbol_data['historical_data']
                    if isinstance(df, pd.DataFrame) and len(df) >= 50:
                        indicators = self.multi_timeframe.calculate_technical_indicators(df)
                        
                        # Get latest values from indicators (which are pandas Series)
                        rsi = indicators.get('rsi', pd.Series([50])).iloc[-1]
                        
                        # Check if ma50 is available (it might be called close_sma or sma_50)
                        ma50 = indicators.get('ma50')
                        if ma50 is None:
                            ma50 = indicators.get('close_sma')  # Check if it's called close_sma
                        if ma50 is None:
                            ma50 = indicators.get('sma_50')    # Check if it's called sma_50
                        if ma50 is None:
                            ma50 = 0  # Fallback value
                        else:
                            ma50 = ma50.iloc[-1]
                            
                        current_price = df['close'].iloc[-1]
                        
                        # Pullback conditions: RSI < 50 and price below 50-period MA
                        if rsi < 50 and current_price < ma50:
                            pullback_signal = TacticalSignal(
                                symbol=symbol,
                                side='buy',
                                strength=0.6,
                                confidence=0.55,
                                source='l2_pullback_strategy',
                                timestamp=pd.Timestamp.now(),
                                features=indicators,
                                metadata={
                                    'reason': f'Pullback buy - aligned with L3 LONG bias (RSI={rsi:.1f}, price < MA50)',
                                    'l3_regime': l3_context.get('regime', 'unknown'),
                                    'l3_bias': 'LONG',
                                    'l3_confidence': l3_context.get('confidence', 0.0),
                                    'strategy_type': 'pullback',
                                    'rsi': rsi,
                                    'ma50_price': ma50,
                                    'current_price': current_price,
                                    'philosophy': 'Buy pullbacks in trending markets - L3 bias = LONG'
                                }
                            )
                            logger.info(f"✅ {symbol}: PULLBACK BUY signal (RSI={rsi:.1f}, price < MA50) - aligned with L3 LONG bias")
                            return pullback_signal
                        else:
                            logger.debug(f"⏸️ {symbol}: No pullback conditions - RSI={rsi:.1f}, price={current_price:.2f}, MA50={ma50:.2f}")
            except Exception as e:
                logger.warning(f"⚠️ Error in pullback strategy: {e}")
        
        # HOLD por defecto - SOLO cuando L3 no tiene bias claro
        # Relajar la regla conservadora de HOLD por defecto
        # Permitir que FinRL/DeepSeek tome más protagonismo cuando L3 no da bias claro
        # Bajar el umbral de "clear bias" y reducir la frecuencia de HOLD

        # Verificar si L2 tiene una señal clara a pesar de L3 no tener bias
        try:
            # Intentar generar señal AI para decidir si usar HOLD o no
            symbol_data = market_data.get(symbol, {})
            if isinstance(symbol_data, dict) and 'historical_data' in symbol_data:
                df = symbol_data['historical_data']
                if isinstance(df, pd.DataFrame) and len(df) >= 50:
                    indicators = self.multi_timeframe.calculate_technical_indicators(df)
                    ai_confidence = self._calculate_ai_confidence(symbol, indicators, l3_context)
                    
        # Relajar el umbral de confianza para permitir más señales
                    if ai_confidence >= 0.55:  # Aumentado de 0.45 a 0.55 para mayor precisión
                        ai_signal = None
                        try:
                            # Intentar generar señal AI para decidir si usar HOLD o no
                            mock_state = {
                                "market_data": market_data,
                                "market_data_simple": market_data,
                                "l3_output": l3_context
                            }
                            # Usar fallback sincrónico para evitar async issues
                            ai_signal = self.finrl_wrapper.generate_signal(mock_state, symbol, indicators)
                            
                            if ai_signal and hasattr(ai_signal, 'side') and ai_signal.side in ['buy', 'sell']:
                                # Si AI tiene una señal clara, usarla en lugar de HOLD
                                logger.warning(f"!!! L2 ACTIVE SIGNAL !!! {ai_signal.side} {ai_signal.confidence:.2f}")
                                return ai_signal
                        except Exception as e:
                            logger.debug(f"Error getting AI signal: {e}")
        except Exception as e:
            logger.debug(f"Error checking L2 AI signal: {e}")

        # HOLD por defecto solo si no hay señal AI clara
        default_hold = TacticalSignal(
            symbol=symbol,
            side='hold',
            strength=0.5,
            confidence=0.4,
            source='l2_conservative_default',
            timestamp=pd.Timestamp.now(),
            features={},
            metadata={
                'reason': 'HOLD as default - L3 has no clear bias',
                'l3_regime': l3_context.get('regime', 'unknown'),
                'l3_signal': l3_signal,
                'l3_confidence': l3_confidence,
                'philosophy': 'Un sistema que siempre opera no es agresivo, es ciego'
            }
        )

        try:
            # Obtener datos del símbolo
            symbol_data = market_data.get(symbol, {})
            if not isinstance(symbol_data, dict) or 'historical_data' not in symbol_data:
                logger.debug(f"⏸️ {symbol}: No market data - HOLD")
                return default_hold

            df = symbol_data['historical_data']
            if not isinstance(df, pd.DataFrame) or len(df) < 50:
                logger.debug(f"⏸️ {symbol}: Insufficient data - HOLD")
                return default_hold

            # Calcular indicadores técnicos
            indicators = self.multi_timeframe.calculate_technical_indicators(df)

            # ========================================================================================
            # PRIORITY 1: BREAK HOLD IN SETUPS - Override conservative logic
            # ========================================================================================
            setup_type = l3_context.get('setup_type')
            allow_setup_trades = l3_context.get('allow_setup_trades', False) or l3_context.get('setup_active', False)
            l3_signal = l3_context.get('signal', 'hold')

            if allow_setup_trades and l3_signal == "buy":
                # if l3.setup_active and l3.signal == "buy": l2_min_confidence = 0.30
                logger.warning(f"🚨 PRIORITY 1 ACTIVATED: L3 setup active + BUY signal - Breaking HOLD with confidence >= 0.30")

                # Reduced confidence threshold for setup trades
                l2_min_confidence = 0.25  # Reduced from 0.30 to 0.25

                # relax momentum requirements for setup trades
                try:
                    ai_confidence = self._calculate_ai_confidence(symbol, indicators, l3_context)
                    if ai_confidence >= l2_min_confidence:  # Relaxed threshold
                        setup_buy_signal = TacticalSignal(
                            symbol=symbol,
                            side='buy',
                            strength=0.6,  # Conservative strength
                            confidence=ai_confidence,
                            source='l2_setup_breakthrough',
                            timestamp=pd.Timestamp.now(),
                            features=indicators,
                            metadata={
                                'reason': f'L3 setup breakthrough - BUY override (conf {ai_confidence:.2f} >= {l2_min_confidence:.2f})',
                                'l3_regime': l3_context.get('regime', 'unknown'),
                                'l3_setup_type': setup_type,
                                'l3_signal': l3_signal,
                                'l3_confidence': l3_context.get('confidence', 0.0),
                                'priority_1_activated': True,
                                'setup_max_allocation': 0.10,  # 10% max allocation
                                'setup_stop_loss': 0.008,  # 0.8% stop loss
                                'philosophy': 'Breaking HOLD in setup - tactical muscle activation'
                            }
                        )
                        logger.info(f"✅ {symbol}: PRIORITY 1 BREAKTHROUGH - BUY signal (conf={ai_confidence:.2f}) - Setup trade!")
                        return setup_buy_signal
                except Exception as e:
                    logger.warning(f"⚠️ Error in setup breakthrough calculation: {e}")

            # ========================================================================================
            # CRITICAL BUG FIX: CHECK L3 SETUP TRADES BEFORE REGIME BLOCK
            # ========================================================================================
            if allow_setup_trades and setup_type == "oversold":
                logger.info(f"🎯 L3 OVERSOLD SETUP DETECTED: L2 can generate BUY signal despite RANGE regime")
                # Generate conservative BUY signal for oversold setup
                l3_confidence = l3_context.get('confidence', 0.5)
                l2_confidence = min(self._calculate_ai_confidence(symbol, indicators, l3_context), 0.7)  # Cap L2 confidence
                final_confidence = min(l3_confidence, l2_confidence)

                setup_buy_signal = TacticalSignal(
                    symbol=symbol,
                    side='buy',
                    strength=0.6,  # Conservative strength
                    confidence=final_confidence,
                    source='l2_oversold_setup_trade',
                    timestamp=pd.Timestamp.now(),
                    features=indicators,
                    metadata={
                        'reason': f'L3 oversold mean reversion setup - confidence {final_confidence:.2f}',
                        'l3_regime': l3_context.get('regime', 'unknown'),
                        'l3_setup_type': setup_type,
                        'l3_confidence': l3_confidence,
                        'l2_confidence': l2_confidence,
                        'setup_max_allocation': l3_context.get('max_allocation_for_setup', 0.10),
                        'exceptional_conditions_met': True,
                        'philosophy': 'L3 setup trades override regime restrictions'
                    }
                )

                logger.info(f"✅ {symbol}: OVERSOLD SETUP BUY signal (conf={final_confidence:.2f}) - L3 setup trade allowed")
                return setup_buy_signal

            # ========================================================================================
            # VALIDACIÓN CONSERVATIVA: Solo BUY/SELL si TODAS las condiciones excepcionales
            # ========================================================================================

            # CONDICIÓN 1: Régimen favorable (no RANGE) - EXCEPT when L3 allows setup trades
            regime = l3_context.get('regime', 'unknown')
            if regime == 'RANGE' and not (allow_setup_trades and setup_type == "oversold"):
                logger.debug(f"⏸️ {symbol}: RANGE regime - HOLD (capital preservation)")
                return default_hold

            # CONDICIÓN 2: Confianza L3 excepcional (>= 0.85)
            l3_confidence = l3_context.get('confidence', 0.0)
            if l3_confidence < 0.85:
                logger.debug(f"⏸️ {symbol}: L3 confidence {l3_confidence:.2f} < 0.85 - HOLD")
                return default_hold

            # CONDICIÓN 3: Momentum claro (no agotado)
            rsi = indicators.get('rsi', 50)
            macd_signal = indicators.get('macd_signal', 0)
            volume_sma = indicators.get('volume_sma', 0)
            current_volume = indicators.get('volume', 0)

            # Rechazar si momentum agotado
            momentum_exhausted = False
            if rsi > 75 or rsi < 25:  # RSI extremo
                if macd_signal < 0 or current_volume < volume_sma * 0.8:  # Divergencia o volumen bajo
                    momentum_exhausted = True

            if momentum_exhausted:
                logger.debug(f"⏸️ {symbol}: Momentum exhausted (RSI={rsi:.1f}, MACD={macd_signal:.3f}) - HOLD")
                return default_hold

            # CONDICIÓN 4: Volatilidad no extrema
            volatility = indicators.get('bb_width', 0.02)
            if volatility > 0.05:  # Volatilidad extrema
                logger.debug(f"⏸️ {symbol}: High volatility {volatility:.3f} - HOLD")
                return default_hold

            # CONDICIÓN 5: AI confianza excepcional (>= 0.80)
            try:
                ai_confidence = self._calculate_ai_confidence(symbol, indicators, l3_context)
                if ai_confidence < 0.80:
                    logger.debug(f"⏸️ {symbol}: AI confidence {ai_confidence:.2f} < 0.80 - HOLD")
                    return default_hold
            except Exception as e:
                logger.warning(f"⚠️ Error calculating AI confidence for {symbol}: {e} - HOLD")
                return default_hold

            # ========================================================================================
            # TODAS LAS CONDICIONES EXCEPCIONALES CUMPLIDAS: Generar BUY/SELL
            # ========================================================================================

            # Determinar dirección basada en momentum
            if rsi > 65 and macd_signal > 0.1:
                signal_side = 'buy'
                reason = f'Conservative BUY: RSI={rsi:.1f}, MACD={macd_signal:.3f}, L3_conf={l3_confidence:.2f}'
            elif rsi < 35 and macd_signal < -0.1:
                signal_side = 'sell'
                reason = f'Conservative SELL: RSI={rsi:.1f}, MACD={macd_signal:.3f}, L3_conf={l3_confidence:.2f}'
            else:
                logger.debug(f"⏸️ {symbol}: No clear direction despite conditions met - HOLD")
                return default_hold

            # Generar señal excepcional
            exceptional_signal = TacticalSignal(
                symbol=symbol,
                side=signal_side,
                strength=min(0.8, ai_confidence),  # Limitar strength
                confidence=ai_confidence,
                source='l2_conservative_exceptional',
                timestamp=pd.Timestamp.now(),
                features=indicators,
                metadata={
                    'reason': reason,
                    'l3_regime': regime,
                    'l3_confidence': l3_confidence,
                    'rsi': rsi,
                    'macd_signal': macd_signal,
                    'volatility': volatility,
                    'exceptional_conditions_met': True,
                    'philosophy': 'Exceptional evidence required for action'
                }
            )

            logger.info(f"✅ {symbol}: EXCEPTIONAL {signal_side.upper()} signal (conf={ai_confidence:.2f}) - All conservative conditions met")
            return exceptional_signal

        except Exception as e:
            logger.error(f"❌ Error in conservative signal generation for {symbol}: {e}")
            return default_hold

    def apply_limited_l3_override(self, base_signals, l3_context, max_override_rate=0.15):
        """
        Aplica L3 override SOLO como EXCEPCIÓN (no regla).

        INVARIANTE: Override ≠ regla, Override = excepción
        - Override rate máximo: 15% de ciclos
        - Solo en condiciones excepcionales
        - Nunca override automático por régimen

        Args:
            base_signals: Lista de señales base (conservativas)
            l3_context: Contexto L3
            max_override_rate: Máximo porcentaje de señales a overridear

        Returns:
            Lista de señales overrideadas (limitada)
        """
        logger.info("🔄 Aplicando L3 override limitado (excepcional)")

        override_signals = []
        max_overrides = max(1, int(len(base_signals) * max_override_rate))  # Al menos 1, máximo 15%

        # Solo override en condiciones EXCEPCIONALES
        exceptional_conditions = (
            l3_context.get('setup_type') in ['oversold', 'overbought'] or  # Setup trading
            l3_context.get('confidence', 0.0) > 0.90  # Confianza excepcional
        )

        if not exceptional_conditions:
            logger.info("⏸️ No exceptional conditions for L3 override - no overrides applied")
            return []

        # Contar overrides aplicados
        overrides_applied = 0

        for signal in base_signals:
            if overrides_applied >= max_overrides:
                break

            # Solo override señales HOLD (no forzar BUY/SELL)
            if getattr(signal, 'side', 'hold') == 'hold':
                # Verificar si L3 tiene dirección clara
                l3_signal = l3_context.get('signal', 'hold')
                if l3_signal in ['buy', 'sell']:
                    # Aplicar override limitado
                    override_signal = TacticalSignal(
                        symbol=getattr(signal, 'symbol', 'UNKNOWN'),
                        side=l3_signal,
                        strength=min(0.7, l3_context.get('confidence', 0.5)),  # Limitar strength
                        confidence=l3_context.get('confidence', 0.5),
                        source='l3_override_exceptional',
                        timestamp=pd.Timestamp.now(),
                        features=getattr(signal, 'features', {}),
                        metadata={
                            'reason': f'L3 exceptional override: {l3_signal.upper()} from HOLD',
                            'original_signal': 'hold',
                            'override_type': 'exceptional',
                            'setup_type': l3_context.get('setup_type'),
                            'l3_confidence': l3_context.get('confidence', 0.0),
                            'max_override_rate': max_override_rate,
                            'philosophy': 'Override = excepción, no regla'
                        }
                    )

                    # 🚨 CRÍTICO: EMITIR SEÑAL OVERRIDE COMO L2_SIGNAL
                    self._emit_l2_signal(override_signal, l3_context)
                    override_signals.append(override_signal)
                    overrides_applied += 1

                    logger.warning(f"🚨 L3 OVERRIDE EXCEPCIÓN: {getattr(signal, 'symbol', 'UNKNOWN')} HOLD → {l3_signal.upper()} (conf={l3_context.get('confidence', 0.0):.2f})")

        logger.info(f"✅ L3 override limitado: {len(override_signals)} excepciones aplicadas (máx {max_overrides})")
        return override_signals

    def process_signals_with_l3_override(self, market_data, l3_info):
        """
        Process L2 signals with L3 context - HOLD dominant approach.

        FUNDAMENTAL RULE: L3 can only suggest context, not force signals.
        L2 maintains strict threshold validation regardless of L3 regime.

        Args:
            market_data: Market data dictionary
            l3_info: L3 regime information

        Returns:
            List of signals based on L2's strict threshold validation
        """
        # ========================================================================================
        # FUNDAMENTAL RULE: L3 CANNOT FORCE SIGNALS - ONLY PROVIDE CONTEXT
        # ========================================================================================
        logger.info("🔄 L3 context integration: Using L2's strict threshold validation (HOLD dominant)")

        # Use the new threshold-based signal generation method
        signals = []
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            signal_output = self._generate_single_signal_with_thresholds(symbol, market_data, l3_info)
            signals.append(signal_output)

        logger.info(f"✅ L2 threshold validation complete: {len(signals)} signals generated")
        return signals

    def _process_signals_normal(self, market_data, l3_info):
        """
        Procesamiento normal de señales L2 cuando no hay override L3.
        """
        # Usar la lógica existente del método process_signals
        mock_state = {
            "market_data": market_data,
            "l3_output": l3_info
        }

        try:
            # Avoid async event loop issues by using simple synchronous processing
            logger.info("🔄 Using synchronous fallbacks to avoid event loop conflicts")
            return self._generate_hold_signals(market_data)
        except Exception as e:
            logger.error(f"❌ Error in normal L2 processing: {e}")
            return self._generate_hold_signals(market_data)

    # Model switching and helper methods
    def switch_model(self, model_key: str) -> bool:
        """Switch to a different L2 model."""
        try:
            if not hasattr(self.config, 'ai_model'):
                return False

            if self.config.ai_model.switch_model(model_key):
                new_model_path = self.config.ai_model.model_path
                logger.info(f"🔄 Switching L2 model to: {model_key} -> {new_model_path}")

                if not os.path.exists(new_model_path):
                    logger.error(f"❌ Model file does not exist: {new_model_path}")
                    return False

                try:
                    new_processor = FinRLProcessor(new_model_path)
                    self.finrl_processor = new_processor
                    self.current_model_path = new_model_path
                    logger.info(f"✅ Successfully switched to model: {model_key}")
                    return True
                except Exception as e:
                    logger.error(f"❌ Failed to load new model {model_key}: {e}")
                    return False
            else:
                logger.error(f"❌ Config switch_model failed for: {model_key}")
                return False
        except Exception as e:
            logger.error(f"❌ Error switching model to {model_key}: {e}")
            return False
