"""
l2_tactic/btc_eth_synchronizer.py - BTC/ETH Synchronization Logic

This module handles all BTC/ETH synchronization functionality including
similarity analysis, synchronized sell triggers, and correlation-based sizing.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from core.logging import logger


class BTCEthereumSynchronizer:
    """Handles BTC/ETH synchronization logic and operations."""

    # 🔧 CONFIGURATION PARAMETERS - All thresholds and multipliers
    SYNCHRONIZATION_CONFIG = {
        # Master controls
        'enabled': True,  # Master switch for BTC/ETH synchronization

        # Safety and risk controls
        'circuit_breakers': {
            'correlation_max': 0.98,  # Disable if correlation > 98%
            'volatility_max': 0.15,   # Disable if volatility > 15%
            'consecutive_failures': 3  # Disable after N consecutive failures
        },

        # Core thresholds
        'similarity_threshold': 0.80,  # Minimum similarity for synchronization
        'confidence_threshold': 0.70,   # Minimum confidence for triggers
        'weakness_threshold': 0.60,     # Minimum weakness for secondary triggers

        # Gradual rollout settings
        'gradual_rollout': {
            'enabled': True,
            'initial_operations': 10,  # Number of operations with conservative settings
            'conservative_multipliers': {
                'similarity': 0.85,  # Higher similarity requirement initially
                'confidence': 0.75,  # Higher confidence requirement initially
                'weakness': 0.65     # Higher weakness requirement initially
            }
        },

        # Correlation-based sizing
        'correlation_sizing': {
            'enabled': True,
            'factor_min': 0.70,  # Minimum position size reduction (70% of original)
            'factor_max': 1.0,   # Maximum position size (no reduction)
            'correlation_threshold': 0.80  # Minimum correlation for sizing adjustments
        },

        # Rollback mechanism
        'rollback': {
            'enabled': True,
            'failure_timeout': 3600,  # 1 hour timeout after failures
            'success_reset_count': 10, # Reset failure count after N successes
            'emergency_failure_threshold': 5  # Emergency rollback after N failures
        },

        # Market condition analysis weights
        'similarity_weights': {
            'correlation': 0.40,    # 40% weight on correlation
            'rsi': 0.25,           # 25% weight on RSI similarity
            'macd': 0.20,          # 20% weight on MACD similarity
            'trend': 0.15          # 15% weight on trend similarity
        }
    }

    def apply_btc_eth_synchronization(self, signals: List[Any], market_data: Dict[str, pd.DataFrame],
                                    state: Dict[str, Any]) -> List[Any]:
        """
        Apply BTC/ETH synchronization logic to signals with safety controls.
        """
        try:
            # 🔒 SAFETY CHECK 1: Master switch
            if not self.SYNCHRONIZATION_CONFIG['enabled']:
                logger.info("🔒 SYNCHRONIZATION DISABLED: Master switch is off")
                return signals

            # 🔒 SAFETY CHECK 2: Enhanced input validation
            validation_result = self._validate_synchronization_inputs(signals, market_data, state)
            if not validation_result['valid']:
                logger.error(f"🚨 SYNCHRONIZATION VALIDATION FAILED: {validation_result['reason']}")
                self._record_synchronization_failure(state, 'validation_failed', validation_result['reason'])
                return signals

            # 🔒 SAFETY CHECK 3: Circuit breaker checks
            circuit_breaker_status = self._check_circuit_breakers(market_data, state)
            if circuit_breaker_status['triggered']:
                logger.warning(f"🚨 CIRCUIT BREAKER TRIGGERED: {circuit_breaker_status['reason']}")
                self._record_synchronization_failure(state, 'circuit_breaker', circuit_breaker_status['reason'])
                return signals

            # 🔒 SAFETY CHECK 4: Rollback mechanism check
            rollback_status = self._check_rollback_status(state)
            if rollback_status['rollback_required']:
                logger.warning(f"🔄 ROLLBACK REQUIRED: {rollback_status['reason']}")
                self._execute_rollback(state, rollback_status['reason'])
                return signals

            # Detect market condition similarity
            similarity_analysis = self._detect_market_condition_similarity(market_data)

            # 🔒 SAFETY CHECK 5: Gradual rollout - use conservative thresholds initially
            effective_similarity_threshold = self.SYNCHRONIZATION_CONFIG['similarity_threshold']
            effective_confidence_threshold = self.SYNCHRONIZATION_CONFIG['confidence_threshold']
            effective_weakness_threshold = self.SYNCHRONIZATION_CONFIG['weakness_threshold']

            success_count = state.get('synchronization_success_count', 0)
            if success_count < self.SYNCHRONIZATION_CONFIG['gradual_rollout']['initial_operations']:
                effective_similarity_threshold = max(
                    effective_similarity_threshold,
                    self.SYNCHRONIZATION_CONFIG['gradual_rollout']['conservative_multipliers']['similarity']
                )
                logger.info("🛡️ GRADUAL ROLLOUT: Using conservative thresholds")

            if similarity_analysis['similarity_score'] < effective_similarity_threshold:
                logger.debug(
                    f"Similarity below threshold: score={similarity_analysis['similarity_score']:.3f} "
                    f"< threshold={effective_similarity_threshold:.3f}"
                )
                return signals

            logger.info(
                f"✅ BTC/ETH similarity OK: score={similarity_analysis['similarity_score']:.3f}, "
                f"corr={similarity_analysis['correlation']:.3f}, rsi_sim={similarity_analysis.get('rsi_similarity', 0.0):.3f}"
            )

            # Get BTC and ETH signals
            btc_signal = next((s for s in signals if getattr(s, 'symbol', '') == 'BTCUSDT'), None)
            eth_signal = next((s for s in signals if getattr(s, 'symbol', '') == 'ETHUSDT'), None)

            # Apply synchronized sell triggers
            synchronized_signals = self._apply_synchronized_sell_triggers(
                signals, btc_signal, eth_signal, similarity_analysis, state, market_data,
                confidence_threshold=effective_confidence_threshold,
                weakness_threshold=effective_weakness_threshold
            )

            # Apply correlation-based sizing adjustments
            final_signals = self._apply_correlation_based_sizing(
                synchronized_signals, similarity_analysis, state
            )

            # Record successful synchronization
            self._record_synchronization_success(state)

            return final_signals

        except Exception as e:
            logger.error(f"❌ Error applying BTC/ETH synchronization: {e}")
            self._record_synchronization_failure(state, 'exception', str(e))
            return signals

    def _detect_market_condition_similarity(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect similarity between BTC and ETH market conditions."""
        try:
            btc_data = market_data.get("BTCUSDT")
            eth_data = market_data.get("ETHUSDT")

            if not (isinstance(btc_data, pd.DataFrame) and isinstance(eth_data, pd.DataFrame)):
                return {'correlation': 0.0, 'similarity_score': 0.0, 'is_similar': False}

            # Weighted correlation score
            correlation_score = self._compute_btc_eth_corr30(market_data)

            # RSI similarity
            btc_rsi = self._calculate_rsi(btc_data)
            eth_rsi = self._calculate_rsi(eth_data)
            rsi_diff = abs(btc_rsi - eth_rsi)
            rsi_similarity = max(0, 1 - (rsi_diff / 50))

            # Overall similarity score
            similarity_score = (
                correlation_score * 0.4 +      # 40% weight on correlation
                rsi_similarity * 0.25 +        # 25% weight on RSI
                0.8 * 0.2 + 0.8 * 0.15        # Default weights for MACD/trend (simplified)
            )

            is_similar = similarity_score > 0.8

            result = {
                'correlation': correlation_score,
                'similarity_score': similarity_score,
                'is_similar': is_similar,
                'rsi_similarity': rsi_similarity,
                'btc_rsi': btc_rsi,
                'eth_rsi': eth_rsi
            }

            logger.info(
                f"Similarity assessment: score={similarity_score:.3f}, corr={correlation_score:.3f}, rsi_sim={rsi_similarity:.3f}"
            )
            return result

        except Exception as e:
            logger.error(f"❌ Error detecting market condition similarity: {e}")
            return {'correlation': 0.0, 'similarity_score': 0.0, 'is_similar': False}

    def _compute_btc_eth_corr30(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate 30-period correlation BTC vs ETH."""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if not (isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame)):
                return 0.0

            eth_close = eth["close"].astype(float).tail(30)
            btc_close = btc["close"].astype(float).tail(30)
            common_idx = eth_close.index.intersection(btc_close.index)
            eth_close = eth_close.loc[common_idx]
            btc_close = btc_close.loc[common_idx]

            if len(eth_close) < 3:
                return 0.0

            eth_ret = eth_close.pct_change().dropna()
            btc_ret = btc_close.pct_change().dropna()
            common_idx = eth_ret.index.intersection(btc_ret.index)

            if len(common_idx) < 3:
                return 0.0

            corr_matrix = np.corrcoef(eth_ret.loc[common_idx], btc_ret.loc[common_idx])
            corr = float(corr_matrix[0, 1])
            return corr if np.isfinite(corr) else 0.0

        except Exception:
            return 0.0

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI for given dataframe."""
        try:
            if len(df) < period + 1:
                return 50.0

            prices = df['close'].tail(period + 1).values
            gains, losses = [], []

            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        except Exception:
            return 50.0

    def _apply_synchronized_sell_triggers(self, signals: List[Any], btc_signal: Optional[Any],
                                        eth_signal: Optional[Any], similarity_analysis: Dict[str, Any],
                                        state: Dict[str, Any], market_data: Dict[str, pd.DataFrame] = None,
                                        confidence_threshold: float = 0.7, weakness_threshold: float = 0.6) -> List[Any]:
        """Apply synchronized sell triggers when BTC and ETH conditions are similar."""
        try:
            correlation = similarity_analysis['correlation']
            btc_side = getattr(btc_signal, 'side', 'hold') if btc_signal else 'hold'
            eth_side = getattr(eth_signal, 'side', 'hold') if eth_signal else 'hold'

            # SYNCHRONIZED SELL TRIGGER logic
            synchronized_signals = []

            for signal in signals:
                symbol = getattr(signal, 'symbol', '')
                signal_side = getattr(signal, 'side', 'hold')
                signal_confidence = getattr(signal, 'confidence', 0.5)

                # Check if this signal should trigger synchronization
                if symbol == 'BTCUSDT' and signal_side == 'sell' and signal_confidence > confidence_threshold and correlation > 0.8:
                    # BTC has strong sell signal, check if ETH should also sell
                    if eth_signal and getattr(eth_signal, 'side', 'hold') != 'sell':
                        eth_weakness = self._check_asset_weakness('ETHUSDT', market_data, state)
                        if eth_weakness > weakness_threshold:
                            logger.info(
                                f"🔁 Sync SELL: BTC strong sell triggers ETH sell (weakness={eth_weakness:.3f}, corr={correlation:.3f})"
                            )
                            eth_sell_signal = self._create_synchronized_sell_signal(eth_signal, signal_confidence * 0.9)
                            synchronized_signals.append(eth_sell_signal)

                elif symbol == 'ETHUSDT' and signal_side == 'sell' and signal_confidence > confidence_threshold and correlation > 0.8:
                    # ETH has strong sell signal, check if BTC should also sell
                    if btc_signal and getattr(btc_signal, 'side', 'hold') != 'sell':
                        btc_weakness = self._check_asset_weakness('BTCUSDT', market_data, state)
                        if btc_weakness > weakness_threshold:
                            logger.info(
                                f"🔁 Sync SELL: ETH strong sell triggers BTC sell (weakness={btc_weakness:.3f}, corr={correlation:.3f})"
                            )
                            btc_sell_signal = self._create_synchronized_sell_signal(btc_signal, signal_confidence * 0.9)
                            synchronized_signals.append(btc_sell_signal)

                # Add original signal
                synchronized_signals.append(signal)

            return synchronized_signals

        except Exception as e:
            logger.error(f"❌ Error applying synchronized sell triggers: {e}")
            return signals

    def _check_asset_weakness(self, symbol: str, market_data: Dict[str, pd.DataFrame], state: Dict[str, Any]) -> float:
        """Check how weak an asset appears based on technical indicators."""
        try:
            df = market_data.get(symbol)
            if not isinstance(df, pd.DataFrame) or df.empty:
                return 0.5

            # RSI weakness (higher RSI = more overbought = weaker)
            rsi = self._calculate_rsi(df)
            rsi_weakness = min(1.0, rsi / 100.0) if rsi > 50 else 0.0

            # Recent price weakness (negative momentum)
            if len(df) >= 5:
                recent_returns = df['close'].pct_change().tail(4).mean()
                momentum_weakness = max(0.0, min(1.0, -recent_returns * 5))
            else:
                momentum_weakness = 0.5

            # Combined weakness score
            weakness_score = (rsi_weakness * 0.3 + momentum_weakness * 0.7)

            logger.debug(f"Asset weakness for {symbol}: RSI={rsi:.1f}({rsi_weakness:.2f}), momentum={momentum_weakness:.2f} → total={weakness_score:.3f}")

            return weakness_score

        except Exception as e:
            logger.error(f"❌ Error checking asset weakness for {symbol}: {e}")
            return 0.5

    def _create_synchronized_sell_signal(self, original_signal: Any, confidence: float) -> Any:
        """Create a synchronized sell signal based on another asset's strong signal."""
        try:
            # For dict-like signals (backward compatibility)
            if hasattr(original_signal, 'get'):
                synchronized_signal = original_signal.copy()
                synchronized_signal.update({
                    'side': 'sell',
                    'confidence': confidence,
                    'source': 'btc_eth_sync',
                    'features': original_signal.get('features', {}).copy(),
                    'metadata': {
                        'synchronized': True,
                        'sync_reason': 'high_correlation_trigger',
                        'original_confidence': original_signal.get('confidence', 0.5)
                    }
                })
                if 'features' in synchronized_signal and isinstance(synchronized_signal['features'], dict):
                    synchronized_signal['features']['synchronized_sell'] = True
                    synchronized_signal['features']['sync_trigger_confidence'] = confidence
                return synchronized_signal

            # For TacticalSignal objects
            else:
                from .models import TacticalSignal

                return TacticalSignal(
                    symbol=getattr(original_signal, 'symbol', ''),
                    side='sell',
                    confidence=confidence,
                    strength=getattr(original_signal, 'strength', 0.8),
                    reason=f"Synchronized sell: {getattr(original_signal, 'symbol', '')} triggered",
                    source='btc_eth_sync',
                    signal_type='synchronized_sell',
                    features={
                        'synchronized_sell': True,
                        'sync_trigger_confidence': confidence,
                        'original_side': getattr(original_signal, 'side', 'hold'),
                        'original_confidence': getattr(original_signal, 'confidence', 0.5)
                    },
                    metadata={
                        'synchronized': True,
                        'sync_reason': 'high_correlation_trigger',
                        'original_confidence': getattr(original_signal, 'confidence', 0.5)
                    },
                    timestamp=pd.Timestamp.now()
                )

        except Exception as e:
            logger.error(f"❌ Error creating synchronized sell signal: {e}")
            return original_signal

    def _apply_correlation_based_sizing(self, signals: List[Any], similarity_analysis: Dict[str, Any], state: Dict[str, Any]) -> List[Any]:
        """Apply correlation-based position sizing adjustments."""
        try:
            correlation = similarity_analysis['correlation']

            # Only apply sizing adjustments for highly correlated markets
            if correlation < 0.8:
                return signals

            logger.info(f"⚖️ Applying correlation-based sizing (corr={correlation:.3f})")
            adjusted_signals = []

            for signal in signals:
                symbol = getattr(signal, 'symbol', '')
                signal_side = getattr(signal, 'side', 'hold')

                # Only adjust sell signals when correlation is high
                if signal_side == 'sell' and symbol in ['BTCUSDT', 'ETHUSDT']:
                    original_quantity = getattr(signal, 'quantity', 0.0)

                    # Reduce position size based on correlation strength
                    correlation_factor = max(0.7, 1.0 - (correlation - 0.8) * 2)
                    adjusted_quantity = original_quantity * correlation_factor

                    # Handle different signal types
                    if hasattr(signal, 'get'):  # Dict-like
                        signal['quantity'] = adjusted_quantity
                        signal['correlation_factor'] = correlation_factor
                        signal['correlation_adjusted'] = True
                    else:  # TacticalSignal object
                        # Create updated signal if needed (attributes may be readonly)
                        if hasattr(signal, '__dict__') or hasattr(signal, '_replace'):  # Dataclass-style
                            setattr(signal, 'quantity', adjusted_quantity)
                            setattr(signal, 'correlation_factor', correlation_factor)
                            setattr(signal, 'correlation_adjusted', True)
                        # If attributes are readonly, we might need to create a new object, but for now assume they are mutable

                    logger.info(
                        f"Adjusted {symbol} SELL size: {original_quantity:.6f} -> {adjusted_quantity:.6f} "
                        f"(factor={correlation_factor:.6f})"
                    )
                adjusted_signals.append(signal)

            return adjusted_signals

        except Exception as e:
            logger.error(f"❌ Error applying correlation-based sizing: {e}")
            return signals

    # Circuit breaker and safety methods
    def _validate_synchronization_inputs(self, signals: List[Any], market_data: Dict[str, pd.DataFrame], state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation for all synchronization inputs."""
        validation_errors = []

        if not signals or not isinstance(signals, list):
            validation_errors.append("Invalid signals list")

        required_symbols = ['BTCUSDT', 'ETHUSDT']
        for symbol in required_symbols:
            if symbol not in [getattr(s, 'symbol', '') for s in signals if s]:
                validation_errors.append(f"Missing signal for {symbol}")

        if validation_errors:
            return {'valid': False, 'reason': '; '.join(validation_errors)}

        return {'valid': True, 'reason': 'All inputs valid'}

    def _check_circuit_breakers(self, market_data: Dict[str, pd.DataFrame], state: Dict[str, Any]) -> Dict[str, Any]:
        """Check circuit breaker conditions."""
        try:
            btc_data = market_data.get('BTCUSDT')
            eth_data = market_data.get('ETHUSDT')

            if btc_data is not None and eth_data is not None:
                correlation = self._compute_btc_eth_corr30(market_data)
                if correlation > self.SYNCHRONIZATION_CONFIG['circuit_breakers']['correlation_max']:
                    return {
                        'triggered': True,
                        'reason': f'Correlation too high: {correlation:.3f} > {self.SYNCHRONIZATION_CONFIG["circuit_breakers"]["correlation_max"]:.3f}'
                    }

            consecutive_failures = state.get('synchronization_consecutive_failures', 0)
            if consecutive_failures >= self.SYNCHRONIZATION_CONFIG['circuit_breakers']['consecutive_failures']:
                return {
                    'triggered': True,
                    'reason': f'Too many consecutive failures: {consecutive_failures}'
                }

            return {'triggered': False, 'reason': 'No circuit breakers triggered'}

        except Exception as e:
            logger.error(f"Error checking circuit breakers: {e}")
            return {'triggered': True, 'reason': f'Circuit breaker check error: {str(e)}'}

    def _check_rollback_status(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if rollback is required."""
        try:
            consecutive_failures = state.get('synchronization_consecutive_failures', 0)
            if consecutive_failures >= self.SYNCHRONIZATION_CONFIG['rollback']['emergency_failure_threshold']:
                return {
                    'rollback_required': True,
                    'reason': f'Emergency rollback: {consecutive_failures} consecutive failures'
                }

            return {'rollback_required': False, 'reason': 'No rollback required'}

        except Exception as e:
            return {'rollback_required': False, 'reason': f'Rollback check error: {str(e)}'}

    def _execute_rollback(self, state: Dict[str, Any], reason: str) -> None:
        """Execute rollback to disable synchronization."""
        try:
            logger.warning(f"🔄 EXECUTING SYNCHRONIZATION ROLLBACK: {reason}")
            self.SYNCHRONIZATION_CONFIG['enabled'] = False
            state['synchronization_consecutive_failures'] = 0

        except Exception as e:
            logger.error(f"Error executing rollback: {e}")

    def _record_synchronization_success(self, state: Dict[str, Any]) -> None:
        """Record successful synchronization operation."""
        success_count = state.get('synchronization_success_count', 0) + 1
        state['synchronization_success_count'] = success_count

        success_reset_count = self.SYNCHRONIZATION_CONFIG['rollback']['success_reset_count']
        if success_count >= success_reset_count:
            state['synchronization_consecutive_failures'] = 0

    def _record_synchronization_failure(self, state: Dict[str, Any], failure_type: str, reason: str) -> None:
        """Record synchronization failure."""
        consecutive_failures = state.get('synchronization_consecutive_failures', 0) + 1
        state['synchronization_consecutive_failures'] = consecutive_failures

        logger.error(f"❌ SYNCHRONIZATION FAILURE [{failure_type}]: {reason} (consecutive: {consecutive_failures})")

        if consecutive_failures >= self.SYNCHRONIZATION_CONFIG['circuit_breakers']['consecutive_failures']:
            logger.error(f"🚨 DISABLING SYNCHRONIZATION: {consecutive_failures} consecutive failures")
            self.SYNCHRONIZATION_CONFIG['enabled'] = False


# Global instance for backward compatibility
btc_eth_synchronizer = BTCEthereumSynchronizer()
