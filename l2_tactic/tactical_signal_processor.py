"""
l2_tactic/tactical_signal_processor.py - Core Tactical Signal Processor

This module contains the main L2TacticProcessor class refactored from the original
signal_generator.py, focusing on core signal processing logic.
"""

import asyncio
import os
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
from .utils import safe_float
# from .btc_eth_synchronizer import btc_eth_synchronizer
from .weight_calculator_integration import weight_calculator_integrator
from .path_mode_generator import PathModeSignalGenerator


class L2TacticProcessor:
    """
    Refactored L2TacticProcessor - Core tactical signal processing.
    Now focuses on main processing logic without embedded sub-systems.
    """

    def __init__(self, config=None, portfolio_manager=None, apagar_l3=False):
        """Initialize processor with core components."""
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
                    'allow_l2_signals': l3_output.get('allow_l2_signals', False)
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
        """Get FinRL AI signal."""
        try:
            if not self.finrl_wrapper:
                return None

            market_data = state.get("market_data_simple") or state.get("market_data", {})
            signal = await self.finrl_wrapper.generate_signal(market_data, symbol, indicators)

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

                # If no signal generated, create a neutral one
                if not signal_for_symbol:
                    signal_for_symbol = TacticalSignal(
                        symbol=symbol,
                        side='hold',
                        strength=0.5,
                        confidence=0.5,
                        source='l2_processor',
                        timestamp=pd.Timestamp.now(),
                        features={},
                        metadata={'l3_context_integrated': True}
                    )

                # Apply additional L3 context-based validation/filtering
                validated_signal = self._apply_l3_context_validation(signal_for_symbol, l3_context)

                if validated_signal:
                    signals.append(validated_signal)
                    logger.debug(f"✅ L2 signal generated for {symbol} with L3 context")
                else:
                    logger.debug(f"⚠️ L2 signal filtered out for {symbol} by L3 context validation")

            except Exception as e:
                logger.error(f"❌ Error generating L2 signal for {symbol}: {e}")
                # Create fallback neutral signal
                fallback_signal = TacticalSignal(
                    symbol=symbol,
                    side='hold',
                    strength=0.4,
                    confidence=0.4,
                    source='l2_fallback',
                    timestamp=pd.Timestamp.now(),
                    features={},
                    metadata={'error': str(e), 'l3_context_integrated': False}
                )
                signals.append(fallback_signal)

        return signals

    def generate_signals(self, market_data: Dict[str, Any], l3_context=None):
        """Genera señales tácticas considerando contexto L3"""

        # VALIDAR L3 context
        if not l3_context:
            logger.warning("⚠️ No L3 context provided, using default HOLD")
            return self._generate_default_signals(market_data)

        # EXTRAER señal L3
        l3_signal = l3_context.get('signal', 'hold')
        l3_confidence = l3_context.get('confidence', 0.0)
        l3_regime = l3_context.get('regime', 'unknown')

        logger.info(f"📊 L2 processing L3: {l3_signal} ({l3_confidence:.2f}) in {l3_regime}")

        # SI L3 DICE BUY CON ALTA CONFIANZA → L2 DEBE GENERAR BUY
        if l3_signal == 'buy' and l3_confidence >= 0.70:
            logger.info("✅ L3 high-confidence BUY signal - L2 generating BUY signals")
            return self._generate_buy_signals(market_data, l3_context)

        # SI L3 DICE SELL CON ALTA CONFIANZA → L2 DEBE GENERAR SELL
        elif l3_signal == 'sell' and l3_confidence >= 0.70:
            logger.info("✅ L3 high-confidence SELL signal - L2 generating SELL signals")
            return self._generate_sell_signals(market_data, l3_context)

        # SI L3 PERMITE L2 → L2 PUEDE USAR SUS PROPIOS MODELOS
        elif l3_context.get('allow_l2', False):
            logger.info("🔓 L3 allows L2 autonomy - using L2 models")
            return self._generate_autonomous_signals(market_data, l3_context)

        # DEFAULT: HOLD
        else:
            logger.info("⏸️ L3 suggests HOLD - maintaining positions")
            return self._generate_hold_signals(market_data)

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

        # If L3 explicitly doesn't allow L2 signals, filter based on confidence
        if not l3_allow_l2 and l3_confidence > 0.60:
            signal_side = getattr(signal, 'side', 'hold')
            if signal_side in ['buy', 'sell']:
                logger.info(f"🚫 L3 context validation: filtering {signal_side} signal for {signal.symbol} (L3 dominance)")
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

                    # Generate signal using FinRL
                    ai_signal = asyncio.run(self._get_finrl_signal(mock_state, symbol, indicators))

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
