"""
l2_tactic/path_modes.py - HRM Path Mode Implementations

This module contains the three different path implementations for HRM_PATH_MODE:
- PATH1: Pure trend following mode
- PATH2: Hybrid mode (trend following with contra-allocation limits)
- PATH3: Full L3 dominance mode
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from core.logging import logger
from l2_tactic.tight_range_handler import PATH2TightRangeFix


class PATH2Processor:
    """Production-grade processor for HRM PATH2 hybrid mode signals"""

    def __init__(self):
        """Initialize PATH2 processor with production-grade tight range handler"""
        self.tight_range_handler = PATH2TightRangeFix()

    def process_signal(self, l1_signals: List[Dict], l2_output: Dict, l3_output: Dict, symbol: str, portfolio_state: Dict) -> Dict:
        """
        Process complete PATH2 hybrid mode signal with robust tight range handling

        Args:
            l1_signals: List of L1 signals
            l2_output: L2 AI model output
            l3_output: L3 regime analysis output
            symbol: Trading symbol
            portfolio_state: Current portfolio state

        Returns:
            Dict containing final signal with enhanced tight range processing
        """
        try:
            from core.config import MAX_CONTRA_ALLOCATION_PATH2

            # Get L3 regime analysis
            regime = l3_output.get('regime', 'neutral').lower()
            l3_confidence = l3_output.get('sentiment_score', 0.5)

            # Get L1/L2 combined signal
            l1_l2_signal = self._extract_l1_l2_signal(l1_signals, l2_output, symbol)

            max_contra_allocation = MAX_CONTRA_ALLOCATION_PATH2
            logger.info(f"🎯 PATH2 - Hybrid Mode for {symbol}:")

            # Enhanced regime coloring
            if regime.upper() in ['BULL', 'BULLISH']:
                colored_regime = f"\x1b[92m{regime.upper()}\x1b[0m"
            elif regime.upper() in ['BEAR', 'BEARISH']:
                colored_regime = f"\x1b[91m{regime.upper()}\x1b[0m"
            elif 'RANGE' in regime.upper() or regime.upper() in ['NEUTRAL', 'RANGE']:
                colored_regime = f"\x1b[95m{regime.upper()}\x1b[0m"
            else:
                colored_regime = f"\x1b[93m{regime.upper()}\x1b[0m"

            logger.info(f"   Regime: {colored_regime}, L3 Confidence: {l3_confidence:.3f}")
            logger.info(f"   L1/L2 Signal: {l1_l2_signal.get('side', 'unknown').upper()}")
            logger.info(f"   Max Contra-Allocation: {max_contra_allocation:.1%}")

            # HYBRID DECISION LOGIC - Production Grade
            regime_trend = None
            if regime in ['bull', 'bullish']:
                regime_trend = 'buy'
            elif regime in ['bear', 'bearish']:
                regime_trend = 'sell'

            if regime_trend:
                # Clear trend from L3 - handle normally
                return self._process_trending_regime(l1_l2_signal, l3_confidence, regime_trend, regime, max_contra_allocation, symbol)
            else:
                # Range/Neutral regime - focus on mean reversion opportunities
                return self._process_range_regime(l1_l2_signal, l3_output, portfolio_state, regime, max_contra_allocation, symbol, l3_confidence)

        except Exception as e:
            logger.error(f"❌ Critical error in PATH2Processor for {symbol}: {e}")
            return {
                'symbol': symbol,
                'side': 'hold',
                'confidence': 0.2,
                'strength': 0.1,
                'source': 'path2_critical_error',
                'reason': f'PATH2 critical error: {str(e)}',
                'path_mode': 'PATH2'
            }

    def _process_trending_regime(self, l1_l2_signal: Dict, l3_confidence: float, regime_trend: str,
                                regime: str, max_contra_allocation: float, symbol: str) -> Dict:
        """Handle trending regimes in PATH2 hybrid mode"""
        l1_l2_side = l1_l2_signal.get('side', 'hold')
        l1_l2_confidence = l1_l2_signal.get('confidence', 0.5)

        if l1_l2_side == regime_trend:
            # Agreement - boost confidence
            final_signal = {
                'symbol': symbol,
                'side': regime_trend,
                'confidence': min(0.95, (l3_confidence + l1_l2_confidence) / 2),
                'strength': 0.9,
                'source': 'path2_hybrid_agreement',
                'reason': f'L1/L2 agrees with {regime} trend (L3:{l3_confidence:.2f}, L1/L2:{l1_l2_confidence:.2f})',
                'path_mode': 'PATH2',
                'regime': regime,
                'max_contra_allocation': max_contra_allocation
            }
            logger.info(f"   → {regime_trend.upper()} (L1/L2 agrees with trend)")

        elif l1_l2_side in ['buy', 'sell'] and l1_l2_side != regime_trend:
            # Disagreement - apply contra-allocation limits
            if l1_l2_confidence > max_contra_allocation:
                contra_confidence = max_contra_allocation
                final_side = l1_l2_side  # Allow but limit
                reason = f'Limited contra-allocation vs {regime} trend (capped at {max_contra_allocation:.1%})'
            else:
                contra_confidence = l1_l2_confidence * (1 - max_contra_allocation)
                final_side = l1_l2_side
                reason = f'Contra-allocation within {max_contra_allocation:.1%} limit'

            final_signal = {
                'symbol': symbol,
                'side': final_side,
                'confidence': contra_confidence,
                'strength': 0.6,
                'source': 'path2_hybrid_contra_limited',
                'reason': reason,
                'path_mode': 'PATH2',
                'regime': regime,
                'max_contra_allocation': max_contra_allocation,
                'contra_allocation_applied': True
            }
            logger.info(f"   → {final_side.upper()} (Limited contra-allocation)")

        else:
            # L1/L2 unclear - follow trend conservatively
            final_signal = {
                'symbol': symbol,
                'side': regime_trend,
                'confidence': l3_confidence * 0.8,
                'strength': 0.7,
                'source': 'path2_hybrid_trend_default',
                'reason': f'Following {regime} trend (L1/L2 unclear)',
                'path_mode': 'PATH2',
                'regime': regime,
                'max_contra_allocation': max_contra_allocation
            }
            logger.info(f"   → {regime_trend.upper()} (Following trend, L1/L2 unclear)")

        return final_signal

    def _process_range_regime(self, l1_l2_signal: Dict, l3_output: Dict, portfolio_state: Dict,
                             regime: str, max_contra_allocation: float, symbol: str, l3_confidence: float) -> Dict:
        """Handle range regimes with production-grade mean reversion"""
        if regime.upper() == 'TIGHT_RANGE':
            # Use the robust production-grade tight range handler
            market_data = portfolio_state.get('market_data', {}).get(symbol)
            if market_data is not None:
                # Convert L1/L2 signal format for enhanced processing
                l1_l2_side = l1_l2_signal.get('side', 'HOLD').upper()
                range_signal = self.tight_range_handler.process_tight_range_signal(
                    symbol=symbol,
                    market_data=market_data,
                    l3_confidence=l3_confidence,
                    l1_l2_signal=l1_l2_side
                )

                if range_signal['action'] in ['BUY', 'SELL']:
                    final_signal = {
                        'symbol': symbol,
                        'side': range_signal['action'].lower(),
                        'confidence': range_signal['confidence'],
                        'strength': 0.8,
                        'source': 'path2_hybrid_tight_range_mean_reversion',
                        'reason': f'TIGHT_RANGE: {range_signal["reason"]}',
                        'path_mode': 'PATH2',
                        'regime': regime,
                        'max_contra_allocation': max_contra_allocation,
                        'range_strategy_activated': True,
                        'stop_loss_pct': range_signal.get('stop_loss_pct'),
                        'take_profit_pct': range_signal.get('take_profit_pct'),
                        'position_size_multiplier': range_signal.get('position_size_multiplier', 0.8),
                        'indicators': range_signal.get('indicators', {}),
                        'allow_partial_rebalance': range_signal.get('allow_partial_rebalance', True),
                        'market_making_enabled': range_signal.get('market_making_enabled', True)
                    }
                    logger.info(f"   → {range_signal['action']} (Production-grade tight range MR)")
                    return final_signal
                else:
                    # HOLD from tight range handler
                    return {
                        'symbol': symbol,
                        'side': 'hold',
                        'confidence': range_signal['confidence'],
                        'strength': 0.4,
                        'source': 'path2_hybrid_tight_range_hold',
                        'reason': f'TIGHT_RANGE: {range_signal["reason"]}',
                        'path_mode': 'PATH2',
                        'regime': regime,
                        'max_contra_allocation': max_contra_allocation
                    }
            else:
                # No market data available
                return {
                    'symbol': symbol,
                    'side': 'hold',
                    'confidence': 0.3,
                    'strength': 0.3,
                    'source': 'path2_hybrid_tight_range_no_data',
                    'reason': f'TIGHT_RANGE: Insufficient market data for mean reversion',
                    'path_mode': 'PATH2',
                    'regime': regime,
                    'max_contra_allocation': max_contra_allocation
                }

        else:
            # Other range/neutral regimes - rely on L1/L2 with caution
            l1_l2_side = l1_l2_signal.get('side', 'hold')
            l1_l2_confidence = l1_l2_signal.get('confidence', 0.5)

            return {
                'symbol': symbol,
                'side': l1_l2_side,
                'confidence': l1_l2_confidence * 0.85,  # Conservative in neutral regime
                'strength': 0.6,
                'source': 'path2_hybrid_neutral_l1_l2',
                'reason': f'L1/L2 signal in {regime} regime (confidence reduced for safety)',
                'path_mode': 'PATH2',
                'regime': regime,
                'max_contra_allocation': max_contra_allocation
            }

    def _extract_l1_l2_signal(self, l1_signals: List[Dict], l2_output: Dict, symbol: str) -> Dict:
        """Extract and aggregate L1/L2 signals for hybrid decision making"""
        try:
            # L1 signals analysis
            l1_weights = {'buy': 0, 'sell': 0, 'hold': 0}
            l1_count = 0

            for signal in l1_signals:
                side = signal.get('action', signal.get('side', 'hold')).lower()
                confidence = signal.get('confidence', 0.5)
                if side in ['buy', 'sell', 'hold']:
                    l1_weights[side] += confidence
                    l1_count += 1

            # L2 signal analysis
            l2_side = l2_output.get('side', 'hold') if isinstance(l2_output, dict) else 'hold'
            l2_confidence = l2_output.get('confidence', 0.5) if isinstance(l2_output, dict) else 0.5

            # Combined decision with weights
            all_weights = {
                'buy': l1_weights['buy'] + (l2_confidence if l2_side == 'buy' else 0),
                'sell': l1_weights['sell'] + (l2_confidence if l2_side == 'sell' else 0),
                'hold': l1_weights['hold'] + (l2_confidence if l2_side == 'hold' else 0)
            }

            winning_side = max(all_weights, key=all_weights.get)
            winning_weight = all_weights[winning_side]
            total_weight = sum(all_weights.values()) or 1

            return {
                'side': winning_side,
                'confidence': winning_weight / total_weight,
                'l1_count': l1_count,
                'l2_included': bool(l2_output)
            }

        except Exception as e:
            logger.error(f"Error extracting L1/L2 signal for {symbol}: {e}")
            return {'side': 'hold', 'confidence': 0.4}


# Legacy function wrapper for backward compatibility
def apply_hybrid_mode(l1_signals: List[Dict], l2_output: Dict, l3_output: Dict, symbol: str, portfolio_state: Dict) -> Dict:
    """Legacy wrapper for backward compatibility - now uses PATH2Processor"""
    processor = PATH2Processor()
    return processor.process_signal(l1_signals, l2_output, l3_output, symbol, portfolio_state)


def apply_pure_trend_following(l1_signals: List[Dict], l2_output: Dict, l3_output: Dict, symbol: str, portfolio_state: Dict) -> Dict:
    """
    PATH1: Hybrid trend following mode implementing L3 awareness and preemptive adjustment
    - L2 generates its own opinion with L3 context metadata
    - L3 adjusts signal when confidence >= 0.70 (preemptive integration)
    - Maintains L2 independence for learning purposes

    Args:
        l1_signals: List of L1 signals
        l2_output: L2 AI model output
        l3_output: L3 regime analysis output
        symbol: Trading symbol (BTCUSDT, ETHUSDT)
        portfolio_state: Current portfolio state

    Returns:
        Dict containing final signal with L3 awareness and adjustments
    """
    try:
        logger.info(f"🎯 PATH1 - Hybrid Trend Following for {symbol} (L3 Awareness)")

        # Generate raw L2 signal first (maintains independence)
        if l2_output and l2_output.get('side'):
            raw_signal = l2_output.copy()
            logger.info(f"   Raw L2: {raw_signal.get('side', 'hold').upper()} conf={raw_signal.get('confidence', 0):.3f}")
        else:
            # Fallback to L1 if no L2 output
            l1_combined = l1_signals[0] if l1_signals else {'side': 'hold', 'confidence': 0.5, 'source': 'l1_fallback'}
            raw_signal = l1_combined.copy()
            logger.info(f"   L1 Fallback: {raw_signal.get('side', 'hold').upper()} conf={raw_signal.get('confidence', 0):.3f}")

        # Add L3 context as metadata (maintains L2 independence)
        raw_signal['l3_regime'] = l3_output.get('regime', 'UNKNOWN')
        raw_signal['l3_sentiment'] = l3_output.get('sentiment_score', 0.5)
        raw_signal['l3_signal'] = l3_output.get('signal', 'UNKNOWN')
        raw_signal['l3_confidence'] = l3_output.get('confidence', 0.0)
        raw_signal['path_mode'] = 'PATH1'

        # Extract L3 context for potential adjustments
        l3_regime = l3_output.get('regime', 'neutral').lower()
        l3_confidence = l3_output.get('confidence', 0.0)

        logger.info(f"   L3 Context: regime={l3_regime}, conf={l3_confidence:.2f}")

        # APPLY L3 ADJUSTMENTS (Preemptive Integration - task requirement)
        adjusted_signal = raw_signal.copy()

        # Strong L3 HOLD dominance - force L2 to HOLD (preemptive)
        if l3_output.get('signal') == "HOLD" and l3_confidence >= 0.70:
            adjusted_signal['side'] = "HOLD"
            adjusted_signal['confidence'] = l3_confidence
            adjusted_signal['source'] = 'path1_l3_forced_hold'
            adjusted_signal['reason'] = f'Adjusted to HOLD due to L3 dominance (conf={l3_confidence:.2f})'
            adjusted_signal['l3_adjustment'] = 'forced_hold'
            logger.info("   → L3 FORCED HOLD (Strong dominance)")

        # Moderate L3 HOLD warning - reduce L2 position size
        elif l3_output.get('signal') == "HOLD" and 0.50 <= l3_confidence < 0.70:
            original_conf = raw_signal.get('confidence', 0.5)
            adjusted_signal['confidence'] *= 0.75  # Reduce confidence
            adjusted_signal['strength'] = adjusted_signal.get('strength', 0.5) * 0.75
            adjusted_signal['source'] = 'path1_l3_caution_reduced'
            adjusted_signal['reason'] = f'Position reduced due to L3 caution (conf={l3_confidence:.2f})'
            adjusted_signal['l3_adjustment'] = 'reduced_size'
            logger.info(f"   → REDUCED POSITION (L3 caution: {original_conf:.2f} → {adjusted_signal['confidence']:.2f})")

        # L3 agrees with L2 or no strong L3 signal - maintain L2 independence
        else:
            adjusted_signal['reason'] = f'L2 maintains independence - L3 context added for learning (l3_conf={l3_confidence:.2f})'
            adjusted_signal['l3_adjustment'] = 'no_adjustment'
            logger.info("   → L2 MAINTAINS INDEPENDENCE (L3 context added)")

        # Ensure consistent format
        final_signal = {
            'symbol': symbol,
            'side': adjusted_signal.get('side', 'hold').lower(),
            'confidence': float(adjusted_signal.get('confidence', 0.5)),
            'strength': float(adjusted_signal.get('strength', 0.5)),
            'source': adjusted_signal.get('source', 'path1_hybrid'),
            'reason': adjusted_signal.get('reason', 'Hybrid trend following with L3 awareness'),
            'path_mode': 'PATH1',
            'regime': l3_regime,
            # Include L3 context metadata for learning
            'l3_regime': l3_regime,
            'l3_sentiment': adjusted_signal['l3_sentiment'],
            'l3_signal': adjusted_signal['l3_signal'],
            'l3_confidence': adjusted_signal['l3_confidence'],
            'l3_adjustment': adjusted_signal.get('l3_adjustment', 'unknown'),
            # Maintain L2 independence indicators
            'l2_independence_maintained': adjusted_signal.get('l3_adjustment') != 'forced_hold',
            'l3_awareness_enabled': True
        }

        logger.info(f"   Final: {final_signal['side'].upper()} conf={final_signal['confidence']:.3f} (L3: {final_signal['l3_adjustment']})")

        return final_signal

    except Exception as e:
        logger.error(f"❌ Error in apply_pure_trend_following hybrid for {symbol}: {e}")
        # Fallback signal with L3 awareness
        return {
            'symbol': symbol,
            'side': 'hold',
            'confidence': 0.4,
            'strength': 0.3,
            'source': 'path1_error_fallback',
            'reason': f'Error in hybrid trend following: {str(e)}',
            'path_mode': 'PATH1',
            'l3_regime': l3_output.get('regime', 'unknown'),
            'l3_sentiment': l3_output.get('sentiment_score', 0.5),
            'l3_signal': l3_output.get('signal', 'unknown'),
            'l3_confidence': l3_output.get('confidence', 0.0),
            'l3_adjustment': 'error_fallback',
            'l2_independence_maintained': False,
            'l3_awareness_enabled': True
        }


def apply_hybrid_mode(l1_signals: List[Dict], l2_output: Dict, l3_output: Dict, symbol: str, portfolio_state: Dict) -> Dict:
    """
    PATH2: Hybrid mode (trend following with contra-allocation limits)
    - Mezcla L1+L2 con L3, pero limita contra-tendencia (ej. 20% contra-allocation)

    Args:
        l1_signals: List of L1 signals
        l2_output: L2 AI model output
        l3_output: L3 regime analysis output
        symbol: Trading symbol (BTCUSDT, ETHUSDT)
        portfolio_state: Current portfolio state

    Returns:
        Dict containing final signal with side, confidence, strength, etc.
    """

    try:
        from core.config import MAX_CONTRA_ALLOCATION_PATH2

        # Get L3 regime analysis
        regime = l3_output.get('regime', 'neutral').lower()
        l3_confidence = l3_output.get('sentiment_score', 0.5)

        # Get L1/L2 combined signal (simplified - in real implementation would aggregate properly)
        l1_l2_signal = _extract_l1_l2_signal(l1_signals, l2_output, symbol)

        max_contra_allocation = MAX_CONTRA_ALLOCATION_PATH2
        logger.info(f"🎯 PATH2 - Hybrid Mode for {symbol}:")
        # Colorear el régimen para PATH2
        if regime.upper() in ['BULL', 'BULLISH']:
            colored_regime = f"\x1b[92m{regime.upper()}\x1b[0m"
        elif regime.upper() in ['BEAR', 'BEARISH']:
            colored_regime = f"\x1b[91m{regime.upper()}\x1b[0m"
        elif 'RANGE' in regime.upper() or regime.upper() in ['NEUTRAL', 'RANGE']:
            colored_regime = f"\x1b[95m{regime.upper()}\x1b[0m"
        else:
            colored_regime = f"\x1b[93m{regime.upper()}\x1b[0m"

        logger.info(f"   Regime: {colored_regime}, L3 Confidence: {l3_confidence:.3f}")
        logger.info(f"   L1/L2 Signal: {l1_l2_signal.get('side', 'unknown').upper()}")
        logger.info(f"   Max Contra-Allocation: {max_contra_allocation:.1%}")

        # HYBRID DECISION LOGIC
        regime_trend = None
        if regime in ['bull', 'bullish']:
            regime_trend = 'buy'
        elif regime in ['bear', 'bearish']:
            regime_trend = 'sell'

        if regime_trend:
            # There's a clear trend from L3
            l1_l2_side = l1_l2_signal.get('side', 'hold')
            l1_l2_confidence = l1_l2_signal.get('confidence', 0.5)

            if l1_l2_side == regime_trend:
                # L1/L2 agrees with trend - full confidence
                final_signal = {
                    'symbol': symbol,
                    'side': regime_trend,
                    'confidence': min(0.95, (l3_confidence + l1_l2_confidence) / 2),
                    'strength': 0.9,
                    'source': 'path2_hybrid_agreement',
                    'reason': f'L1/L2 agrees with {regime} trend (L3_conf: {l3_confidence:.3f}, L1/L2_conf: {l1_l2_confidence:.3f})',
                    'path_mode': 'PATH2',
                    'regime': regime,
                    'max_contra_allocation': max_contra_allocation
                }
                logger.info(f"   → {regime_trend.upper()} (L1/L2 agrees with trend)")

            elif l1_l2_side in ['buy', 'sell'] and l1_l2_side != regime_trend:
                # L1/L2 disagrees - limit contra-tendency exposure
                if l1_l2_confidence > max_contra_allocation:
                    # Strong disagreement - reduce but allow limited contra-allocation
                    contra_confidence = max_contra_allocation
                    final_side = l1_l2_side  # Allow contra-allocation within limit
                    reason = f'Limited contra-allocation vs {regime} trend (L3_conf: {l3_confidence:.3f}, limited to {max_contra_allocation:.1%})'
                else:
                    # Weak disagreement - allow but reduce confidence
                    contra_confidence = l1_l2_confidence * (1 - max_contra_allocation)
                    final_side = l1_l2_side
                    reason = f'Contra-allocation within limits vs {regime} trend'

                final_signal = {
                    'symbol': symbol,
                    'side': final_side,
                    'confidence': contra_confidence,
                    'strength': 0.6,
                    'source': 'path2_hybrid_contra_limited',
                    'reason': reason,
                    'path_mode': 'PATH2',
                    'regime': regime,
                    'max_contra_allocation': max_contra_allocation,
                    'contra_allocation_applied': True
                }
                logger.info(f"   → {final_side.upper()} (Limited contra-allocation)")

            else:
                # L1/L2 is hold or unclear - follow trend but with reduced confidence
                final_signal = {
                    'symbol': symbol,
                    'side': regime_trend,
                    'confidence': l3_confidence * 0.8,
                    'strength': 0.7,
                    'source': 'path2_hybrid_trend_default',
                    'reason': f'Following {regime} trend (L1/L2 unclear, L3_conf: {l3_confidence:.3f})',
                    'path_mode': 'PATH2',
                    'regime': regime,
                    'max_contra_allocation': max_contra_allocation
                }
                logger.info(f"   → {regime_trend.upper()} (Following trend, L1/L2 unclear)")

        else:
            # Neutral/range regime handling
            if regime == 'tight_range':
                # CRITICAL FIX: Use mean reversion logic for tight ranges instead of just HOLD
                market_data = portfolio_state.get('market_data', {}).get(symbol)
                if market_data is not None and isinstance(market_data, pd.DataFrame) and len(market_data) > 20:
                    tight_range_fix = PATH2TightRangeFix()
                    range_signal = tight_range_fix.process_tight_range_signal(symbol, market_data, l3_confidence)

                    final_signal = {
                        'symbol': symbol,
                        'side': range_signal['action'].lower(),
                        'confidence': range_signal['confidence'],
                        'strength': 0.8,
                        'source': 'path2_hybrid_tight_range_mean_reversion',
                        'reason': f'TIGHT RANGE MEAN REVERSION: {range_signal["reason"]} (L3_conf: {l3_confidence:.3f})',
                        'path_mode': 'PATH2',
                        'regime': regime,
                        'max_contra_allocation': max_contra_allocation,
                        'range_strategy_activated': True,
                        'stop_loss_pct': range_signal.get('stop_loss_pct'),
                        'take_profit_pct': range_signal.get('take_profit_pct'),
                        'allow_partial_rebalance': range_signal.get('allow_partial_rebalance', True),
                        'market_making_enabled': range_signal.get('market_making_enabled', True)
                    }
                    logger.info(f"   → {range_signal['action']} (Tight range mean reversion)")
                else:
                    # Fallback if no market data
                    final_signal = {
                        'symbol': symbol,
                        'side': 'hold',
                        'confidence': 0.5,
                        'strength': 0.4,
                        'source': 'path2_hybrid_tight_range_no_data',
                        'reason': f'TIGHT RANGE - No market data for mean reversion (L3_conf: {l3_confidence:.3f})',
                        'path_mode': 'PATH2',
                        'regime': regime,
                        'max_contra_allocation': max_contra_allocation
                    }
                    logger.info("   → HOLD (Tight range - no data for mean reversion)")
            else:
                # Other neutral/range regimes - rely on L1/L2 but with caution
                l1_l2_side = l1_l2_signal.get('side', 'hold')
                l1_l2_confidence = l1_l2_signal.get('confidence', 0.5)

                final_signal = {
                    'symbol': symbol,
                    'side': l1_l2_side,
                    'confidence': l1_l2_confidence * 0.9,  # Slightly reduce confidence in neutral regime
                    'strength': 0.7,
                    'source': 'path2_hybrid_neutral_l1_l2',
                    'reason': f'L1/L2 signal in neutral regime ({regime}, L1/L2_conf: {l1_l2_confidence:.3f})',
                    'path_mode': 'PATH2',
                    'regime': regime,
                    'max_contra_allocation': max_contra_allocation
                }
                logger.info(f"   → {l1_l2_side.upper()} (L1/L2 in {regime} regime)")

        return final_signal

    except Exception as e:
        logger.error(f"❌ Error in apply_hybrid_mode for {symbol}: {e}")
        # Fallback signal
        return {
            'symbol': symbol,
            'side': 'hold',
            'confidence': 0.4,
            'strength': 0.3,
            'source': 'path2_error_fallback',
            'reason': f'Error in hybrid mode: {str(e)}',
            'path_mode': 'PATH2'
        }


def apply_full_l3_dominance(l3_output: Dict, symbol: str, portfolio_state: Dict) -> Dict:
    """
    PATH3: Full L3 dominance mode
    - L3 manda 100%, bloquea cualquier señal en contra

    Args:
        l3_output: L3 regime analysis output
        symbol: Trading symbol (BTCUSDT, ETHUSDT)
        portfolio_state: Current portfolio state

    Returns:
        Dict containing final signal with side, confidence, strength, etc.
    """

    try:
        # Get L3 regime analysis
        regime = l3_output.get('regime', 'neutral').lower()
        l3_confidence = l3_output.get('sentiment_score', 0.5)

        logger.info(f"🎯 PATH3 - Full L3 Dominance for {symbol}:")
        # Colorear el régimen para mejor visibilidad en PATH3
        if regime.upper() in ['BULL', 'BULLISH']:
            colored_regime = f"\x1b[92m{regime.upper()}\x1b[0m"  # Verde para bull
        elif regime.upper() in ['BEAR', 'BEARISH']:
            colored_regime = f"\x1b[91m{regime.upper()}\x1b[0m"  # Rojo para bear
        elif 'RANGE' in regime.upper() or regime.upper() in ['NEUTRAL', 'RANGE']:
            colored_regime = f"\x1b[95m{regime.upper()}\x1b[0m"  # Rosa/Magenta para range
        else:
            colored_regime = f"\x1b[93m{regime.upper()}\x1b[0m"  # Amarillo para otros

        logger.info(f"   Regime: {colored_regime}, L3 Confidence: {l3_confidence:.3f}")

        # FULL L3 DOMINANCE - L3 decides everything
        if regime in ['bull', 'bullish']:
            # BULL REGIME: FORCE BUY - block all opposing signals
            final_signal = {
                'symbol': symbol,
                'side': 'buy',
                'confidence': min(0.95, l3_confidence),
                'strength': 1.0,
                'source': 'path3_full_l3_dominance',
                'reason': f'FULL L3 DOMINANCE - Bull regime forces BUY (regime: {regime}, L3_conf: {l3_confidence:.3f})',
                'path_mode': 'PATH3',
                'regime': regime,
                'l3_dominance': True,
                'blocks_opposing_signals': True
            }
            logger.info("   → BUY (Full L3 dominance - Bull regime)")

        elif regime in ['bear', 'bearish']:
            # BEAR REGIME: FORCE SELL - block all opposing signals
            final_signal = {
                'symbol': symbol,
                'side': 'sell',
                'confidence': min(0.95, l3_confidence),
                'strength': 1.0,
                'source': 'path3_full_l3_dominance',
                'reason': f'FULL L3 DOMINANCE - Bear regime forces SELL (regime: {regime}, L3_conf: {l3_confidence:.3f})',
                'path_mode': 'PATH3',
                'regime': regime,
                'l3_dominance': True,
                'blocks_opposing_signals': True
            }
            logger.info("   → SELL (Full L3 dominance - Bear regime)")

        else:
            # RANGE REGIME: ACTIVAR MEAN-REVERSION STRATEGY
            # ✅ NUEVO: Implementar mean-reversion en PATH3 para régimen RANGE
            try:
                from .generators.mean_reversion import RangeBoundStrategy
                range_strategy = RangeBoundStrategy({})

                # Usar datos de mercado del portfolio_state si están disponibles
                market_data = portfolio_state.get('market_data', {}).get(symbol)
                if market_data is not None and len(market_data) > 20:
                    signal = range_strategy.generate_signal(symbol, market_data)

                    if signal['action'] in ['BUY', 'SELL']:
                        # Activar estrategia de range con confianza ajustada
                        final_signal = {
                            'symbol': symbol,
                            'side': signal['action'].lower(),
                            'confidence': signal['confidence'] * 0.8,  # Ajustar confianza para range
                            'strength': 0.7,
                            'source': 'path3_range_mean_reversion',
                            'reason': f'PATH3 RANGE MEAN-REVERSION - {signal["action"]} signal (regime: {regime}, range_position_confirmed)',
                            'path_mode': 'PATH3',
                            'regime': regime,
                            'l3_dominance': False,  # Mean-reversion toma control en ranges
                            'range_strategy_activated': True,
                            'stop_loss': signal.get('stop_loss'),
                            'take_profit': signal.get('take_profit'),
                            'position_size_multiplier': signal.get('position_size_multiplier', 0.6)
                        }
                        logger.info(f"   → {signal['action']} (RANGE MEAN-REVERSION activated - {regime})")
                    else:
                        # HOLD cuando no hay señal clara de range
                        final_signal = {
                            'symbol': symbol,
                            'side': 'hold',
                            'confidence': 0.6,
                            'strength': 0.5,
                            'source': 'path3_range_mean_reversion_hold',
                            'reason': f'PATH3 RANGE MEAN-REVERSION - HOLD (no clear range signal, regime: {regime})',
                            'path_mode': 'PATH3',
                            'regime': regime,
                            'l3_dominance': False,
                            'range_strategy_activated': True
                        }
                        logger.info("   → HOLD (RANGE MEAN-REVERSION - no signal)")
                else:
                    # Fallback si no hay datos suficientes
                    final_signal = {
                        'symbol': symbol,
                        'side': 'hold',
                        'confidence': 0.5,
                        'strength': 0.4,
                        'source': 'path3_range_fallback',
                        'reason': f'PATH3 RANGE - Insufficient data for mean-reversion (regime: {regime})',
                        'path_mode': 'PATH3',
                        'regime': regime,
                        'l3_dominance': True,
                        'range_strategy_activated': False
                    }
                    logger.info("   → HOLD (RANGE - insufficient data for mean-reversion)")

            except Exception as range_error:
                logger.error(f"❌ Error in PATH3 range mean-reversion for {symbol}: {range_error}")
                # Fallback a HOLD si hay error
                final_signal = {
                    'symbol': symbol,
                    'side': 'hold',
                    'confidence': 0.4,
                    'strength': 0.3,
                    'source': 'path3_range_error_fallback',
                    'reason': f'PATH3 RANGE ERROR - {str(range_error)}',
                    'path_mode': 'PATH3',
                    'regime': regime,
                    'l3_dominance': True,
                    'range_strategy_activated': False
                }
                logger.info("   → HOLD (RANGE ERROR - fallback)")

        return final_signal

    except Exception as e:
        logger.error(f"❌ Error in apply_full_l3_dominance for {symbol}: {e}")
        # Fallback signal
        return {
            'symbol': symbol,
            'side': 'hold',
            'confidence': 0.4,
            'strength': 0.3,
            'source': 'path3_error_fallback',
            'reason': f'Error in full L3 dominance: {str(e)}',
            'path_mode': 'PATH3'
        }





def _extract_l1_l2_signal(l1_signals: List[Dict], l2_output: Dict, symbol: str) -> Dict:
    """
    Extract combined L1/L2 signal for hybrid mode decisions
    (Simplified implementation - in real system would aggregate properly)
    """
    try:
        # L1 signals analysis
        l1_weights = {'buy': 0, 'sell': 0, 'hold': 0}
        l1_count = 0

        for signal in l1_signals:
            side = signal.get('action', signal.get('side', 'hold')).lower()
            confidence = signal.get('confidence', 0.5)
            if side in ['buy', 'sell', 'hold']:
                l1_weights[side] += confidence
                l1_count += 1

        # L2 signal analysis (simplified)
        l2_side = l2_output.get('side', 'hold') if isinstance(l2_output, dict) else 'hold'
        l2_confidence = l2_output.get('confidence', 0.5) if isinstance(l2_output, dict) else 0.5

        # Combined decision
        all_weights = {
            'buy': l1_weights['buy'] + (l2_confidence if l2_side == 'buy' else 0),
            'sell': l1_weights['sell'] + (l2_confidence if l2_side == 'sell' else 0),
            'hold': l1_weights['hold'] + (l2_confidence if l2_side == 'hold' else 0)
        }

        winning_side = max(all_weights, key=all_weights.get)
        winning_weight = all_weights[winning_side]
        total_weight = sum(all_weights.values()) or 1

        return {
            'side': winning_side,
            'confidence': winning_weight / total_weight,
            'l1_count': l1_count,
            'l2_included': bool(l2_output)
        }

    except Exception as e:
        logger.error(f"Error extracting L1/L2 signal for {symbol}: {e}")
        return {'side': 'hold', 'confidence': 0.4}
