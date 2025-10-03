"""
L2 Tactical Signal Generator - Refactored Architecture (PATH MODE ORCHESTRATOR)

Clean orchestrator that delegates to dedicated path-mode generators.
Split from monolithic structure into focused, maintainable components.
"""

import asyncio
from typing import Dict, List, Any
import pandas as pd
from core.logging import logger
from core.config import HRM_PATH_MODE

# Import the new dedicated components
from .signal_generators import (
    BaseSignalGenerator,
    PureTrendFollowingGenerator,
    HybridModeGenerator,
    FullL3DominanceGenerator,
    create_generator
)
from .signal_components import PostProcessingMixin


class L2TacticProcessor(PostProcessingMixin):
    """
    Refactored L2TacticProcessor - Clean orchestrator for HRM path modes.

    Architecture:
    - Delegates path-specific logic to dedicated generator classes
    - Applies post-processing (BTC/ETH sync, weight calculator, risk overlay)
    - Maintains backward compatibility and clean interfaces
    """

    def __init__(self, config=None, portfolio_manager=None, apagar_l3=False):
        """
        Initialize L2TacticProcessor with configuration and path mode generator

        Args:
            config: L2 configuration object
            portfolio_manager: Portfolio manager instance
            apagar_l3: Whether to disable L3 processing
        """
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.apagar_l3 = apagar_l3

        # Initialize path mode generator based on HRM_PATH_MODE
        self.path_generator = create_generator(
            HRM_PATH_MODE,
            config={'max_contra_allocation': 0.80}  # Configurable through new system
        )

        # Initialize L3 cache for regime information
        self.l3_context_cache = {}

        logger.info(f"✅ L2TacticProcessor initialized - Path Mode: {HRM_PATH_MODE} (refactored architecture)")

    def switch_model(self, model_key: str) -> bool:
        """Switch to a different L2 model dynamically"""
        try:
            if not hasattr(self.config, 'ai_model'):
                logger.error("❌ No ai_model config available for switching")
                return False

            # Try to switch model in config
            if self.config.ai_model.switch_model(model_key):
                new_model_path = self.config.ai_model.model_path
                logger.info(f"🔄 Switching L2 model to: {model_key} -> {new_model_path}")

                # Check if file exists
                import os
                if not os.path.exists(new_model_path):
                    logger.error(f"❌ Model file does not exist: {new_model_path}")
                    return False

                logger.info(f"✅ Successfully switched to model: {model_key}")
                return True
            else:
                logger.error(f"❌ Config switch_model failed for: {model_key}")
                return False
        except Exception as e:
            logger.error(f"❌ Error switching model to {model_key}: {e}")
            return False

    async def process_signals(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Main orchestrator for L2 signal processing.
        Delegates to path generator and applies post-processing.

        Args:
            state: System state containing market data, portfolio info, etc.

        Returns:
            List of final tactical signals
        """
        try:
            logger.info(f"🎯 L2TACTIC ORCHESTRATOR: Processing signals with {HRM_PATH_MODE}")

            # Extract symbols from market data
            market_data = state.get("market_data", {}) or state.get("market_data_simple", {})
            if not market_data:
                logger.warning("⚠️ L2: No market data available")
                return []

            symbols = list(market_data.keys())[:2]  # Support max 2 symbols for now
            logger.info(f"🎯 Processing symbols: {symbols}")

            # Generate initial signals using path-specific logic
            initial_signals = await self._generate_path_signals(symbols, state)

            # Apply post-processing pipeline
            final_signals = await self.apply_post_processing(
                initial_signals, market_data, state
            )

            logger.info(f"✅ L2TACTIC ORCHESTRATOR: Generated {len(final_signals)} final signals")
            for signal in final_signals:
                side = getattr(signal, 'side', 'unknown')
                conf = getattr(signal, 'confidence', 0.0)
                source = getattr(signal, 'source', 'unknown')
                logger.info(f"   {signal.symbol}: {side.upper()} (conf={conf:.3f}, source={source})")

            return final_signals

        except Exception as e:
            logger.error(f"❌ Error in L2TacticProcessor orchestrator: {e}")
            return []

    async def _generate_path_signals(self, symbols: List[str], state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Generate signals using the dedicated path mode generator

        Args:
            symbols: List of symbols to process
            state: System state

        Returns:
            List of tactical signals
        """
        try:
            signals = []
            market_data = state.get("market_data", {}) or state.get("market_data_simple", {})

            # Get L3 regime information
            l3_regime = await self._extract_l3_regime(state)

            for symbol in symbols:
                try:
                    # Get symbol-specific market data
                    symbol_data = market_data.get(symbol, {})
                    if not isinstance(symbol_data, dict):
                        symbol_data = {'historical_data': symbol_data}

                    # Get L1 signals (simplified for refactoring)
                    l1_signals = await self._get_l1_signals_simple(symbol, symbol_data, state)

                    # Get L2 output (simplified for refactoring)
                    l2_output = await self._get_l2_output_simple(symbol, symbol_data, l1_signals, state)

                    # Generate signal using dedicated path generator
                    signal = await self.path_generator.generate_signal(
                        symbol=symbol,
                        market_data=symbol_data,
                        l1_signals=l1_signals,
                        l2_output=l2_output,
                        l3_regime=l3_regime,
                        portfolio_state=state.get("portfolio", {})
                    )

                    signals.append(signal)

                except Exception as e:
                    logger.error(f"❌ Error generating signal for {symbol}: {e}")
                    # Create fallback signal
                    from core.models import TacticalSignal
                    fallback_signal = TacticalSignal(
                        symbol=symbol,
                        side='hold',
                        confidence=0.4,
                        strength=0.3,
                        signal_type='fallback_error',
                        source='error_fallback',
                        timestamp=pd.Timestamp.now(),
                        features={'error': str(e)},
                        metadata={}
                    )
                    signals.append(fallback_signal)

            return signals

        except Exception as e:
            logger.error(f"❌ Error in path signal generation: {e}")
            return []

    async def _extract_l3_regime(self, state: Dict[str, Any]) -> str:
        """
        Extract L3 regime information from state

        Returns:
            L3 regime string ('bull', 'bear', 'neutral', etc.)
        """
        try:
            # Try multiple sources for L3 regime info
            l3_context_cache = state.get("l3_context_cache", {})
            l3_output = l3_context_cache.get("last_output", {}) or state.get("l3_output", {})

            regime = l3_output.get('regime', 'neutral')
            return regime.lower()

        except Exception as e:
            logger.warning(f"❌ Error extracting L3 regime, using neutral: {e}")
            return 'neutral'

    async def _get_l1_signals_simple(self, symbol: str, market_data: Dict, state: Dict[str, Any]) -> List[Dict]:
        """
        Simplified L1 signal getter for refactored architecture
        """
        try:
            df = market_data.get('historical_data')
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return []

            signals = []

            # RSI-based signals
            if len(df) >= 14:
                rsi = self._calculate_rsi_simple(df)
                if rsi < 30:
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'side': 'buy',
                        'confidence': 0.6,
                        'source': 'l1_rsi_oversold'
                    })
                elif rsi > 70:
                    signals.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'side': 'sell',
                        'confidence': 0.6,
                        'source': 'l1_rsi_overbought'
                    })

            # MACD-based signals
            if len(df) >= 26:
                macd, signal_line = self._calculate_macd_simple(df)
                macd_diff = macd - signal_line
                if macd_diff > 5:
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'side': 'buy',
                        'confidence': 0.55,
                        'source': 'l1_macd_bullish'
                    })
                elif macd_diff < -5:
                    signals.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'side': 'sell',
                        'confidence': 0.55,
                        'source': 'l1_macd_bearish'
                    })

            return signals

        except Exception as e:
            logger.warning(f"Error getting L1 signals for {symbol}: {e}")
            return []

    async def _get_l2_output_simple(self, symbol: str, market_data: Dict, l1_signals: List[Dict], state: Dict[str, Any]) -> Dict:
        """
        Simplified L2 output getter for refactored architecture
        """
        try:
            if not l1_signals:
                return {
                    'side': 'hold',
                    'confidence': 0.5,
                    'source': 'l2_fallback'
                }

            # Aggregate L1 signals
            buy_count = sum(1 for s in l1_signals if s.get('side') == 'buy')
            sell_count = sum(1 for s in l1_signals if s.get('side') == 'sell')

            if buy_count > sell_count:
                return {
                    'side': 'buy',
                    'confidence': min(0.9, 0.5 + (buy_count - sell_count) * 0.1),
                    'source': 'l2_l1_aggregated'
                }
            elif sell_count > buy_count:
                return {
                    'side': 'sell',
                    'confidence': min(0.9, 0.5 + (sell_count - buy_count) * 0.1),
                    'source': 'l2_l1_aggregated'
                }
            else:
                return {
                    'side': 'hold',
                    'confidence': 0.5,
                    'source': 'l2_l1_aggregated'
                }

        except Exception as e:
            logger.warning(f"Error getting L2 output for {symbol}: {e}")
            return {'side': 'hold', 'confidence': 0.4, 'source': 'l2_error_fallback'}

    def _calculate_rsi_simple(self, df, period=14):
        """Simple RSI calculation"""
        try:
            if len(df) < period + 1:
                return 50.0

            prices = df['close'].values[-period-1:]
            gains = [max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))]
            losses = [max(0, prices[i-1] - prices[i]) for i in range(1, len(prices))]

            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        except Exception:
            return 50.0

    def _calculate_macd_simple(self, df):
        """Simple MACD calculation"""
        try:
            if len(df) < 26:
                return 0.0, 0.0

            prices = df['close'].values

            # Simple approximation
            ema12 = prices[-13:].mean() if len(prices) >= 13 else prices[-1]
            ema26 = prices[-27:].mean() if len(prices) >= 27 else prices[-1]

            macd = ema12 - ema26
            signal_line = macd * 0.8  # Approximation

            return macd, signal_line

        except Exception:
            return 0.0, 0.0

    # Legacy methods for backward compatibility
    def apply_l3_trend_following_override(self, signal, l3_regime: str, symbol: str, state: Dict[str, Any]):
        """
        Legacy method - now handled by path generators
        """
        logger.warning("⚠️ apply_l3_trend_following_override is now handled by path generators")
        return signal

    def _check_portfolio_has_positions(self, symbol: str, state: Dict[str, Any]) -> bool:
        """
        Legacy method - moved to signal_components
        """
        logger.warning("⚠️ _check_portfolio_has_positions moved to signal_components")
        return False


# Import core models for signal creation
from core.models import TacticalSignal
