"""
L2 Tactical Signal Generators - Dedicated Path-Mode Generators

Separate signal generator classes for different HRM path modes:
- PureTrendFollowingGenerator (PATH1)
- HybridModeGenerator (PATH2)
- FullL3DominanceGenerator (PATH3)

Each generator encapsulates its specific signal processing logic.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from core.logging import logger
from l2_tactic.models import TacticalSignal
import pandas as pd
from utils import safe_float


class BaseSignalGenerator(ABC):
    """
    Abstract base class for all signal generators.
    Provides common interface and shared functionality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    async def generate_signal(self, symbol: str, market_data: Dict[str, Any],
                            l1_signals: List[Dict], l2_output: Dict,
                            l3_regime: str, portfolio_state: Dict[str, Any]) -> TacticalSignal:
        """
        Generate signal for a specific symbol using this generator's logic.

        Args:
            symbol: Trading symbol (BTCUSDT, ETHUSDT)
            market_data: Market data for the symbol
            l1_signals: List of L1 operational signals
            l2_output: L2 AI model output
            l3_regime: Current L3 regime classification
            portfolio_state: Current portfolio state

        Returns:
            TacticalSignal object with the final signal
        """
        pass

    def _create_signal(self, symbol: str, side: str, confidence: float,
                       strength: float, source: str, reason: str,
                       **kwargs) -> TacticalSignal:
        """Helper to create standardized TacticalSignal objects"""
        signal = TacticalSignal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strength=strength,
            signal_type=f"path_generator_{source}",
            source=source,
            timestamp=pd.Timestamp.now(),
            features=kwargs.get('features', {}),
            metadata=kwargs.get('metadata', {})
        )
        signal.reason = reason  # Add reason attribute
        return signal


class PureTrendFollowingGenerator(BaseSignalGenerator):
    """
    PATH1: Pure trend following mode generator
    - Only follows L3 trends, completely ignores mean-reversion
    - L3 has complete dominance over L1/L2 signals
    """

    async def generate_signal(self, symbol: str, market_data: Dict[str, Any],
                            l1_signals: List[Dict], l2_output: Dict,
                            l3_regime: str, portfolio_state: Dict[str, Any]) -> TacticalSignal:

        try:
            logger.info(f"🎯 PURE TREND FOLLOWING ({symbol}): L3 regime = {l3_regime}")

            # Extract L3 confidence from market data or state
            l3_confidence = self._extract_l3_confidence(market_data, portfolio_state)

            # PURE TREND FOLLOWING DECISIONS
            if l3_regime.lower() in ['bull', 'bullish']:
                return self._create_signal(
                    symbol=symbol,
                    side='buy',
                    confidence=min(0.9, l3_confidence),
                    strength=0.8,
                    source='pure_trend_following',
                    reason=f'Bull trend following (L3_regime: {l3_regime}, confidence: {l3_confidence:.3f})',
                    features={
                        'path_mode': 'PATH1',
                        'l3_regime': l3_regime,
                        'trend_direction': 'bullish',
                        'l3_confidence': l3_confidence,
                        'ignores_l1_l2': True
                    }
                )

            elif l3_regime.lower() in ['bear', 'bearish']:
                return self._create_signal(
                    symbol=symbol,
                    side='sell',
                    confidence=min(0.9, l3_confidence),
                    strength=0.8,
                    source='pure_trend_following',
                    reason=f'Bear trend following (L3_regime: {l3_regime}, confidence: {l3_confidence:.3f})',
                    features={
                        'path_mode': 'PATH1',
                        'l3_regime': l3_regime,
                        'trend_direction': 'bearish',
                        'l3_confidence': l3_confidence,
                        'ignores_l1_l2': True
                    }
                )

            else:  # Neutral, range, unknown
                return self._create_signal(
                    symbol=symbol,
                    side='hold',
                    confidence=0.6,
                    strength=0.4,
                    source='pure_trend_following',
                    reason=f'Neutral/range regime - no trend to follow (L3_regime: {l3_regime}, confidence: {l3_confidence:.3f})',
                    features={
                        'path_mode': 'PATH1',
                        'l3_regime': l3_regime,
                        'trend_direction': 'neutral',
                        'l3_confidence': l3_confidence,
                        'ignores_l1_l2': True
                    }
                )

        except Exception as e:
            logger.error(f"❌ Error in PureTrendFollowingGenerator for {symbol}: {e}")
            return self._create_signal(
                symbol=symbol,
                side='hold',
                confidence=0.4,
                strength=0.3,
                source='pure_trend_following_error',
                reason=f'Error in pure trend following: {str(e)}'
            )

    def _extract_l3_confidence(self, market_data: Dict[str, Any], portfolio_state: Dict[str, Any]) -> float:
        """Extract L3 confidence score from available data"""
        # Try to get from market_data first (newer L3 integration)
        if 'l3_output' in market_data:
            l3_output = market_data['l3_output']
            if isinstance(l3_output, dict) and 'sentiment_score' in l3_output:
                return l3_output['sentiment_score']

        # Fallback to portfolio_state (legacy)
        if 'l3_output' in portfolio_state:
            l3_output = portfolio_state['l3_output']
            if isinstance(l3_output, dict) and 'sentiment_score' in l3_output:
                return l3_output['sentiment_score']

        # Default confidence
        return 0.5


class HybridModeGenerator(BaseSignalGenerator):
    """
    PATH2: Hybrid mode generator (trend following with contra-allocation limits)
    - Combines L1+L2 with L3, but limits contra-trend allocation
    - Allows limited contra-allocation when L1/L2 strongly disagree
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        from core.config import MAX_CONTRA_ALLOCATION_PATH2
        self.max_contra_allocation = config.get('max_contra_allocation', MAX_CONTRA_ALLOCATION_PATH2)

    async def generate_signal(self, symbol: str, market_data: Dict[str, Any],
                            l1_signals: List[Dict], l2_output: Dict,
                            l3_regime: str, portfolio_state: Dict[str, Any]) -> TacticalSignal:

        try:
            logger.info(f"🎯 HYBRID MODE ({symbol}): L3 regime = {l3_regime}, max contra = {self.max_contra_allocation:.1%}")

            # Extract L3 confidence
            l3_confidence = self._extract_l3_confidence(market_data, portfolio_state)

            # Get combined L1/L2 signal
            l1_l2_signal = await self._get_l1_l2_combined_signal(symbol, l1_signals, l2_output, market_data)

            # Determine L3 trend direction
            regime_trend = self._determine_regime_trend(l3_regime)

            if regime_trend:
                # There's a clear trend from L3
                final_signal = await self._process_trend_with_l1_l2(
                    symbol, regime_trend, l3_confidence, l1_l2_signal, market_data
                )
            else:
                # Neutral/range regime - rely on L1/L2 with caution
                final_signal = self._process_neutral_regime(symbol, l3_regime, l3_confidence, l1_l2_signal)

            return final_signal

        except Exception as e:
            logger.error(f"❌ Error in HybridModeGenerator for {symbol}: {e}")
            return self._create_signal(
                symbol=symbol,
                side='hold',
                confidence=0.4,
                strength=0.3,
                source='hybrid_mode_error',
                reason=f'Error in hybrid mode: {str(e)}'
            )

    async def _get_l1_l2_combined_signal(self, symbol: str, l1_signals: List[Dict],
                                       l2_output: Dict, market_data: Dict[str, Any]) -> Dict:
        """Combine L1 and L2 signals into a unified decision"""
        try:
            # Simple L1 aggregation
            l1_buy_weight = 0
            l1_sell_weight = 0

            for signal in l1_signals:
                side = signal.get('action', signal.get('side', 'hold')).lower()
                confidence = signal.get('confidence', 0.5)
                if side == 'buy':
                    l1_buy_weight += confidence
                elif side == 'sell':
                    l1_sell_weight += confidence

            # L2 signal
            l2_side = l2_output.get('side', 'hold') if isinstance(l2_output, dict) else 'hold'
            l2_confidence = l2_output.get('confidence', 0.5) if isinstance(l2_output, dict) else 0.5

            # Combined decision
            total_buy = l1_buy_weight + (l2_confidence if l2_side == 'buy' else 0)
            total_sell = l1_sell_weight + (l2_confidence if l2_side == 'sell' else 0)

            if total_buy > total_sell and total_buy > 0.3:  # Minimum threshold
                return {'side': 'buy', 'confidence': min(0.9, total_buy), 'source': 'l1_l2_combined'}
            elif total_sell > total_buy and total_sell > 0.3:
                return {'side': 'sell', 'confidence': min(0.9, total_sell), 'source': 'l1_l2_combined'}
            else:
                return {'side': 'hold', 'confidence': 0.5, 'source': 'l1_l2_combined'}

        except Exception as e:
            logger.warning(f"Error combining L1/L2 signals for {symbol}: {e}")
            return {'side': 'hold', 'confidence': 0.4, 'source': 'l1_l2_fallback'}

    def _determine_regime_trend(self, l3_regime: str) -> Optional[str]:
        """Determine trend direction from L3 regime"""
        regime_lower = l3_regime.lower()
        if regime_lower in ['bull', 'bullish']:
            return 'buy'
        elif regime_lower in ['bear', 'bearish']:
            return 'sell'
        else:
            return None

    async def _process_trend_with_l1_l2(self, symbol: str, regime_trend: str, l3_confidence: float,
                                     l1_l2_signal: Dict, market_data: Dict[str, Any]) -> TacticalSignal:
        """Process signal when there's a clear L3 trend, considering L1/L2 agreement/disagreement"""

        l1_l2_side = l1_l2_signal.get('side', 'hold')
        l1_l2_confidence = l1_l2_signal.get('confidence', 0.5)

        if l1_l2_side == regime_trend:
            # Agreement - full confidence
            combined_confidence = min(0.95, (l3_confidence + l1_l2_confidence) / 2)
            return self._create_signal(
                symbol=symbol,
                side=regime_trend,
                confidence=combined_confidence,
                strength=0.9,
                source='hybrid_agreement',
                reason=f'L1/L2 agrees with {regime_trend} trend (L3_conf: {l3_confidence:.3f}, L1/L2_conf: {l1_l2_confidence:.3f})',
                features={
                    'path_mode': 'PATH2',
                    'agreement': True,
                    'regime_trend': regime_trend,
                    'l3_confidence': l3_confidence,
                    'l1_l2_confidence': l1_l2_confidence,
                    'max_contra_allocation': self.max_contra_allocation
                }
            )

        elif l1_l2_side in ['buy', 'sell'] and l1_l2_side != regime_trend:
            # Disagreement - apply contra-allocation limits
            if l1_l2_confidence > self.max_contra_allocation:
                # Strong disagreement - allow limited contra-allocation
                contra_confidence = self.max_contra_allocation
                return self._create_signal(
                    symbol=symbol,
                    side=l1_l2_side,  # Allow contra-allocation
                    confidence=contra_confidence,
                    strength=0.6,
                    source='hybrid_contra_limited',
                    reason=f'Limited contra-allocation vs {regime_trend} trend (L3_conf: {l3_confidence:.3f}, limited to {self.max_contra_allocation:.1%})',
                    features={
                        'path_mode': 'PATH2',
                        'agreement': False,
                        'regime_trend': regime_trend,
                        'l3_confidence': l3_confidence,
                        'l1_l2_confidence': l1_l2_confidence,
                        'max_contra_allocation': self.max_contra_allocation,
                        'contra_allocation_applied': True
                    }
                )
            else:
                # Weak disagreement - allow but reduce confidence
                contra_confidence = l1_l2_confidence * (1 - self.max_contra_allocation)
                return self._create_signal(
                    symbol=symbol,
                    side=l1_l2_side,  # Allow contra-allocation
                    confidence=contra_confidence,
                    strength=0.6,
                    source='hybrid_contra_allowed',
                    reason=f'Contra-allocation within limits vs {regime_trend} trend',
                    features={
                        'path_mode': 'PATH2',
                        'agreement': False,
                        'regime_trend': regime_trend,
                        'l3_confidence': l3_confidence,
                        'l1_l2_confidence': l1_l2_confidence,
                        'max_contra_allocation': self.max_contra_allocation,
                        'contra_allocation_applied': True
                    }
                )
        else:
            # L1/L2 unclear - follow trend with reduced confidence
            return self._create_signal(
                symbol=symbol,
                side=regime_trend,
                confidence=l3_confidence * 0.8,
                strength=0.7,
                source='hybrid_trend_default',
                reason=f'Following {regime_trend} trend (L1/L2 unclear, L3_conf: {l3_confidence:.3f})',
                features={
                    'path_mode': 'PATH2',
                    'agreement': 'unclear',
                    'regime_trend': regime_trend,
                    'l3_confidence': l3_confidence,
                    'l1_l2_confidence': l1_l2_confidence,
                    'max_contra_allocation': self.max_contra_allocation
                }
            )

    def _process_neutral_regime(self, symbol: str, l3_regime: str, l3_confidence: float,
                               l1_l2_signal: Dict) -> TacticalSignal:
        """Process signal in neutral/range regime"""

        l1_l2_side = l1_l2_signal.get('side', 'hold')
        l1_l2_confidence = l1_l2_signal.get('confidence', 0.5)

        return self._create_signal(
            symbol=symbol,
            side=l1_l2_side,
            confidence=l1_l2_confidence * 0.9,  # Reduce confidence in neutral regime
            strength=0.7,
            source='hybrid_neutral_l1_l2',
            reason=f'L1/L2 signal in neutral regime {l3_regime} (L1/L2_conf: {l1_l2_confidence:.3f})',
            features={
                'path_mode': 'PATH2',
                'regime': l3_regime,
                'l3_confidence': l3_confidence,
                'l1_l2_confidence': l1_l2_confidence,
                'neutral_regime': True
            }
        )

    def _extract_l3_confidence(self, market_data: Dict[str, Any], portfolio_state: Dict[str, Any]) -> float:
        """Extract L3 confidence score - same as parent class method"""
        return super()._extract_l3_confidence(market_data, portfolio_state)


class FullL3DominanceGenerator(BaseSignalGenerator):
    """
    PATH3: Full L3 dominance generator
    - L3 has 100% control, blocks any opposing signals
    - No compromise with L1/L2 signals
    """

    async def generate_signal(self, symbol: str, market_data: Dict[str, Any],
                            l1_signals: List[Dict], l2_output: Dict,
                            l3_regime: str, portfolio_state: Dict[str, Any]) -> TacticalSignal:

        try:
            logger.info(f"🎯 FULL L3 DOMINANCE ({symbol}): L3 regime = {l3_regime}")

            # Extract L3 confidence
            l3_confidence = self._extract_l3_confidence(market_data, portfolio_state)

            # FULL L3 DOMINANCE DECISIONS - NO COMPROMISE
            if l3_regime.lower() in ['bull', 'bullish']:
                return self._create_signal(
                    symbol=symbol,
                    side='buy',
                    confidence=min(0.95, l3_confidence),
                    strength=1.0,  # Maximum strength
                    source='full_l3_dominance',
                    reason=f'FULL L3 DOMINANCE - Bull regime forces BUY (L3_regime: {l3_regime}, confidence: {l3_confidence:.3f})',
                    features={
                        'path_mode': 'PATH3',
                        'l3_regime': l3_regime,
                        'trend_direction': 'bullish',
                        'l3_confidence': l3_confidence,
                        'blocks_opposing_signals': True,
                        'dominance_level': 'full'
                    }
                )

            elif l3_regime.lower() in ['bear', 'bearish']:
                return self._create_signal(
                    symbol=symbol,
                    side='sell',
                    confidence=min(0.95, l3_confidence),
                    strength=1.0,  # Maximum strength
                    source='full_l3_dominance',
                    reason=f'FULL L3 DOMINANCE - Bear regime forces SELL (L3_regime: {l3_regime}, confidence: {l3_confidence:.3f})',
                    features={
                        'path_mode': 'PATH3',
                        'l3_regime': l3_regime,
                        'trend_direction': 'bearish',
                        'l3_confidence': l3_confidence,
                        'blocks_opposing_signals': True,
                        'dominance_level': 'full'
                    }
                )

            else:  # Neutral, range, unknown - HOLD with high conviction
                return self._create_signal(
                    symbol=symbol,
                    side='hold',
                    confidence=0.7,  # Higher confidence than other paths
                    strength=0.8,  # Higher strength
                    source='full_l3_dominance',
                    reason=f'FULL L3 DOMINANCE - Neutral/range holds all positions (L3_regime: {l3_regime}, confidence: {l3_confidence:.3f})',
                    features={
                        'path_mode': 'PATH3',
                        'l3_regime': l3_regime,
                        'trend_direction': 'neutral',
                        'l3_confidence': l3_confidence,
                        'blocks_opposing_signals': True,
                        'dominance_level': 'full'
                    }
                )

        except Exception as e:
            logger.error(f"❌ Error in FullL3DominanceGenerator for {symbol}: {e}")
            return self._create_signal(
                symbol=symbol,
                side='hold',
                confidence=0.4,
                strength=0.3,
                source='full_l3_dominance_error',
                reason=f'Error in full L3 dominance: {str(e)}'
            )

    def _extract_l3_confidence(self, market_data: Dict[str, Any], portfolio_state: Dict[str, Any]) -> float:
        """Extract L3 confidence score - same as parent class method"""
        return super()._extract_l3_confidence(market_data, portfolio_state)


# Generator factory function for easy instantiation
def create_generator(path_mode: str, config: Optional[Dict[str, Any]] = None) -> BaseSignalGenerator:
    """
    Factory function to create appropriate generator based on path mode.

    Args:
        path_mode: HRM_PATH_MODE value ('PATH1', 'PATH2', 'PATH3')
        config: Optional configuration dictionary

    Returns:
        Instance of appropriate generator class

    Raises:
        ValueError: If invalid path_mode is provided
    """
    path_mode = path_mode.upper()

    if path_mode == 'PATH1':
        return PureTrendFollowingGenerator(config)
    elif path_mode == 'PATH2':
        return HybridModeGenerator(config)
    elif path_mode == 'PATH3':
        return FullL3DominanceGenerator(config)
    else:
        raise ValueError(f"Invalid HRM_PATH_MODE: {path_mode}. Must be PATH1, PATH2, or PATH3")


# -----------------------------------------------------------------------------
# Compatibility shim for FinRLProcessor expecting SignalGenerators.prepare_observation
# and SignalGenerators.action_to_signal
# -----------------------------------------------------------------------------
class SignalGenerators:
    @staticmethod
    def prepare_observation(input_data: Dict[str, Any]) -> 'np.ndarray':
        """
        Legacy 13-dimensional observation builder expected by FinRLProcessor.
        Accepts a dict-like input_data with feature keys. Missing keys are zero-filled.
        """
        import numpy as np
        feature_names = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'rsi',
            'bollinger_upper', 'bollinger_lower',
            'ema_12', 'ema_26', 'macd'
        ]
        values: List[float] = []
        data = input_data or {}
        # Support pandas Series as well
        get_val = data.get if hasattr(data, 'get') else lambda k, d=None: None
        for name in feature_names:
            try:
                v = get_val(name, 0.0)
                if hasattr(v, 'item'):
                    v = v.item()
                v = safe_float(v) if v is not None else 0.0
            except Exception:
                v = 0.0
            values.append(v if pd.notna(v) else 0.0)
        # Ensure length 13
        if len(values) < 13:
            values.extend([0.0] * (13 - len(values)))
        elif len(values) > 13:
            values = values[:13]
        return np.array(values, dtype=np.float32)

    @staticmethod
    def action_to_signal(action_value: float, symbol: str, model_name: str, value: Any = None) -> TacticalSignal:
        """
        Maps a scalar action_value into a TacticalSignal. Aligns thresholds with finrl_wrapper._action_to_signal.
        """
        try:
            av = safe_float(action_value)
        except Exception:
            av = 0.5

        # Normalize to [0,1] if it's outside
        if av < 0.0:
            av = 0.0
        if av > 1.0:
            av = 1.0

        if av < 0.4:
            side = 'sell'
            confidence = 0.6 + (0.4 - av) * 0.5
            strength = 0.5 + (0.4 - av) * 0.5
        elif av > 0.6:
            side = 'buy'
            confidence = 0.6 + (av - 0.6) * 0.5
            strength = 0.5 + (av - 0.6) * 0.5
        else:
            side = 'hold'
            confidence = 0.5
            strength = 0.3

        confidence = max(0.3, min(0.9, confidence))
        strength = max(0.2, min(0.9, strength))

        return TacticalSignal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strength=strength,
            signal_type=f'finrl_{model_name}',
            source='finrl',
            timestamp=pd.Timestamp.utcnow(),
            features={'model': model_name, 'action_value': av}
        )
