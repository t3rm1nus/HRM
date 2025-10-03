"""
PATH MODE SIGNAL GENERATOR - ENHANCED FOR SETUP DETECTION
Handles PATH1, PATH2, PATH3 signal processing with setup awareness
"""

import logging
from typing import Dict, Any, Optional
from core.logging import logger

class PathModeSignalGenerator:
    """
    Intelligent signal generator for different trading paths with setup detection
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_signal(self, symbol: str, l1_l2_signal: str, l3_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal based on HRM_PATH_MODE with setup awareness

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            l1_l2_signal: Raw L1/L2 signal (BUY/SELL/HOLD)
            l3_context: L3 decision context including regime and setup info

        Returns:
            Processed trading signal with setup-aware logic
        """
        path_mode = l3_context.get('path_mode', 'PATH2')
        regime = l3_context.get('regime', 'neutral')
        l3_signal = l3_context.get('l3_signal', 'hold')
        l3_conf = l3_context.get('l3_confidence', 0.0)

        # Extract setup information from L3 decision
        l3_decision = l3_context.get('l3_decision', {})
        setup_type = l3_decision.get('setup_type')
        allow_l2_signals = l3_decision.get('allow_l2_signals', False)

        self.logger.info(f"🎯 {path_mode} SIGNAL GENERATION: {symbol}")
        self.logger.info(f"   L1/L2: {l1_l2_signal} | L3: {l3_signal} ({l3_conf:.2f}) | Regime: {regime}")
        self.logger.info(f"   Setup: {setup_type} | Allow L2: {allow_l2_signals}")

        # Process based on path mode
        if path_mode == 'PATH1':
            return self._process_path1_signal(symbol, l1_l2_signal, l3_context)
        elif path_mode == 'PATH2':
            return self._process_path2_signal(symbol, l1_l2_signal, l3_context, setup_type, allow_l2_signals)
        elif path_mode == 'PATH3':
            return self._process_path3_signal(symbol, l1_l2_signal, l3_context)
        else:
            self.logger.warning(f"Unknown path_mode: {path_mode}, defaulting to PATH2")
            return self._process_path2_signal(symbol, l1_l2_signal, l3_context, setup_type, allow_l2_signals)

    def _process_path1_signal(self, symbol: str, l1_l2_signal: str, l3_context: Dict[str, Any]) -> Dict[str, Any]:
        """PATH1: Pure Trend-Following - Regime driven only"""
        regime = l3_context.get('regime', 'neutral')
        l3_signal = l3_context.get('l3_signal', 'hold')
        l3_conf = l3_context.get('l3_confidence', 0.0)

        self.logger.info(f"   PATH1: Pure regime following - {regime.upper()}")

        # PATH1 ignores L1/L2 signals - purely regime driven
        return {
            'symbol': symbol,
            'action': l3_signal.upper(),
            'confidence': min(l3_conf, 0.85),  # Capped for safety
            'size_multiplier': 1.0,
            'path_mode': 'PATH1',
            'regime_driven': True,
            'reason': f'path1_{regime}_regime_following'
        }

    def _process_path2_signal(self, symbol: str, l1_l2_signal: str, l3_context: Dict[str, Any],
                            setup_type: Optional[str], allow_l2_signals: bool) -> Dict[str, Any]:
        """PATH2: Hybrid Intelligent - Balanced multi-signal with setup awareness"""
        regime = l3_context.get('regime', 'neutral')
        l3_signal = l3_context.get('l3_signal', 'hold')
        l3_conf = l3_context.get('l3_confidence', 0.0)

        self.logger.info(f"   PATH2: Hybrid intelligence - {regime.upper()} regime with L1/L2")

        # PRIORITY: Handle activated setups
        if setup_type == 'oversold' and l1_l2_signal.upper() == 'BUY':
            self.logger.info(f"   ✅ OVERSOLD SETUP: Allowing L2 BUY for {symbol} with 50% size")
            return {
                'symbol': symbol,
                'action': 'BUY',
                'confidence': min(l3_conf, 0.70),
                'size_multiplier': 0.50,
                'setup_trade': True,
                'path_mode': 'PATH2',
                'setup_type': setup_type,
                'reason': f'path2_oversold_setup_mean_reversion'
            }

        elif setup_type == 'overbought' and l1_l2_signal.upper() == 'SELL':
            self.logger.info(f"   ✅ OVERBOUGHT SETUP: Allowing L2 SELL for {symbol} with 50% size")
            return {
                'symbol': symbol,
                'action': 'SELL',
                'confidence': min(l3_conf, 0.70),
                'size_multiplier': 0.50,
                'setup_trade': True,
                'path_mode': 'PATH2',
                'setup_type': setup_type,
                'reason': f'path2_overbought_setup_mean_reversion'
            }

        # HIGH CONFIDENCE L3: Allow L2 signals when L3 has strong conviction
        if l3_conf > 0.75 and regime.lower() != 'range':
            self.logger.info(f"   ✅ STRONG REGIME: Allowing L2 signal for {symbol}")
            return {
                'symbol': symbol,
                'action': l1_l2_signal.upper(),
                'confidence': l3_conf,
                'size_multiplier': 1.0,
                'path_mode': 'PATH2',
                'l3_driven': True,
                'reason': f'path2_strong_{regime.lower()}_regime'
            }

        # RANGE REGIME: Strict control unless setup allows
        if regime.lower() == 'range':
            if allow_l2_signals:
                self.logger.info(f"   ⚠️ SETUP OVERRIDE: Allowing L2 signal for {symbol} despite range regime")
                return {
                    'symbol': symbol,
                    'action': l1_l2_signal.upper(),
                    'confidence': min(l3_conf, 0.60),
                    'size_multiplier': 0.75,
                    'setup_override': True,
                    'path_mode': 'PATH2',
                    'reason': f'path2_range_setup_override'
                }
            else:
                self.logger.info(f"   🚫 RANGE REGIME: Blocking L2 {l1_l2_signal} for {symbol}")
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': l3_conf,
                    'size_multiplier': 0.0,
                    'path_mode': 'PATH2',
                    'range_blocked': True,
                    'reason': f'path2_range_regime_block'
                }

        # DEFAULT: Conservative approach
        self.logger.info(f"   📊 PATH2 CONSERVATIVE: L3 priority over L2")
        return {
            'symbol': symbol,
            'action': l3_signal.upper(),
            'confidence': l3_conf,
            'size_multiplier': 0.8,  # Slightly reduced size
            'path_mode': 'PATH2',
            'l3_priority': True,
            'reason': f'path2_conservative_l3_priority'
        }

    def _process_path3_signal(self, symbol: str, l1_l2_signal: str, l3_context: Dict[str, Any]) -> Dict[str, Any]:
        """PATH3: Full L3 Dominance - L3 signals only"""
        l3_signal = l3_context.get('l3_signal', 'hold').upper()
        l3_conf = l3_context.get('l3_confidence', 0.0)

        self.logger.info(f"   PATH3: Full L3 dominance - L3 signals only")
        self.logger.info(f"   🚫 BLOCKED L1/L2 signal: {l1_l2_signal.upper()}")

        # PATH3: Complete L3 dominance - ignore L1/L2 completely
        return {
            'symbol': symbol,
            'action': l3_signal,
            'confidence': l3_conf,
            'size_multiplier': 1.0,
            'path_mode': 'PATH3',
            'l3_only': True,
            'l1_l2_blocked': True,
            'blocked_signal': l1_l2_signal.upper(),
            'reason': f'path3_full_l3_dominance'
        }
