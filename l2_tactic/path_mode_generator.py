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

        # Extract setup information from L3 context
        setup_type = l3_context.get('setup_type')
        allow_l2_signals = l3_context.get('allow_l2_signals', False)
        allow_setup_trades = l3_context.get('allow_setup_trades', False)

        self.logger.info(f"🎯 {path_mode} SIGNAL GENERATION: {symbol}")
        self.logger.info(f"   L1/L2: {l1_l2_signal} | L3: {l3_signal} ({l3_conf:.2f}) | Regime: {regime}")
        self.logger.info(f"   Setup: {setup_type} | Allow L2: {allow_l2_signals} | Allow Setup Trades: {allow_setup_trades}")

        # Process based on path mode
        if path_mode == 'PATH1':
            return self._process_path1_signal(symbol, l1_l2_signal, l3_context)
        elif path_mode == 'PATH2':
            return self._process_path2_signal(symbol, l1_l2_signal, l3_context, setup_type, allow_l2_signals, allow_setup_trades)
        elif path_mode == 'PATH3':
            return self._process_path3_signal(symbol, l1_l2_signal, l3_context)
        else:
            self.logger.warning(f"Unknown path_mode: {path_mode}, defaulting to PATH2")
            return self._process_path2_signal(symbol, l1_l2_signal, l3_context, setup_type, allow_l2_signals, allow_setup_trades)

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
                            setup_type: Optional[str], allow_l2_signals: bool, allow_setup_trades: bool) -> Dict[str, Any]:
        """PATH2: Hybrid Intelligent - Balanced multi-signal with setup awareness"""
        regime = l3_context.get('regime', 'neutral')
        l3_signal = l3_context.get('l3_signal', 'hold')
        l3_conf = l3_context.get('l3_confidence', 0.0)

        self.logger.info(f"   PATH2: Hybrid intelligence - {regime.upper()} regime with L1/L2")

        # PRIORITY: Handle activated setups - ONLY if allow_setup_trades is True
        if allow_setup_trades and setup_type == 'oversold' and l1_l2_signal.upper() == 'BUY':
            # Calculate confidence as per specification: max(0.5, min(l3_confidence, tactical_confidence + 0.15))
            # Using l3_conf as tactical_confidence for now (L2 confidence not directly available here)
            tactical_confidence = l3_conf
            final_confidence = max(0.5, min(l3_conf, tactical_confidence + 0.15))
            setup_max_allocation = l3_context.get('setup_max_allocation', 0.10)

            self.logger.info(f"   ✅ OVERSOLD SETUP: L2 BUY allowed for {symbol} (conf={final_confidence:.2f})")
            return {
                'symbol': symbol,
                'action': 'BUY',
                'confidence': final_confidence,
                'size_multiplier': setup_max_allocation,
                'setup_trade': True,
                'path_mode': 'PATH2',
                'setup_type': setup_type,
                'reason': 'L3 oversold setup - mean reversion'
            }

        elif allow_setup_trades and setup_type == 'overbought' and l1_l2_signal.upper() == 'SELL':
            # Calculate confidence as per specification: max(0.5, min(l3_confidence, tactical_confidence + 0.15))
            # Using l3_conf as tactical_confidence for now (L2 confidence not directly available here)
            tactical_confidence = l3_conf
            final_confidence = max(0.5, min(l3_conf, tactical_confidence + 0.15))
            setup_max_allocation = l3_context.get('setup_max_allocation', 0.10)

            self.logger.info(f"   ✅ OVERBOUGHT SETUP: L2 SELL allowed for {symbol} (conf={final_confidence:.2f})")
            return {
                'symbol': symbol,
                'action': 'SELL',
                'confidence': final_confidence,
                'size_multiplier': setup_max_allocation,
                'setup_trade': True,
                'path_mode': 'PATH2',
                'setup_type': setup_type,
                'reason': 'L3 overbought setup - mean reversion'
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

        # RANGE REGIME: Only HOLD if not allowing setup trades
        if regime.lower() == 'range' and not allow_setup_trades:
            self.logger.info(f"   🔄 RANGE REGIME: HOLD (capital preservation, no setup trades allowed)")
            return self._generate_range_signal(symbol, l3_context)

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

    def _generate_range_signal(self, symbol: str, l3_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate range-specific signal for RANGE regime when L2 signals are blocked
        with complete protection against None values

        Args:
            symbol: Trading symbol
            l3_context: L3 context with regime information

        Returns:
            Dictionary with range-specific signal
        """
        # === VALIDACIÓN Y EXTRACCIÓN SEGURA DE DATOS ===
        if not l3_context:
            logger.warning(f"⚠️ L3 context is None for {symbol}, using safe defaults")
            l3_context = {}

        l3_conf = float(l3_context.get('l3_confidence', 0.50))
        subtype = str(l3_context.get('subtype', 'normal_range')).lower()  # SAFE: Convert to string then lower
        regime = str(l3_context.get('regime', 'range')).lower()  # SAFE: Convert to string then lower
        setup_type = l3_context.get('setup_type')

        # === CONSTRUCCIÓN DE SEÑAL POR DEFECTO ===
        signal = {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': l3_conf,
            'size_multiplier': 0.0,  # Conservative sizing for range
            'path_mode': 'PATH2',
            'range_specific': True,
            'range_subtype': subtype,
            'regime': regime,
            'setup_type': str(setup_type).lower() if setup_type else None,
            'reason': f'range_{subtype}_hold_signal'
        }

        # === MANEJO ESPECÍFICO POR SUBTIPO CON SEGURIDAD ===
        if subtype == 'tight_range':
            # Tight range is more predictable - slightly more aggressive
            signal['confidence'] = min(l3_conf + 0.1, 0.85)  # Boost confidence slightly
            signal['size_multiplier'] = 0.0  # Still conservative but logging position
            signal['reason'] = 'tight_range_ready_hold'

        elif subtype == 'normal_range':
            # Normal range - standard hold
            signal['confidence'] = l3_conf
            signal['size_multiplier'] = 0.0
            signal['reason'] = 'normal_range_standard_hold'

        elif subtype == 'wide_range':
            # Wide range - might allow some action with very small size
            signal['confidence'] = l3_conf
            signal['size_multiplier'] = 0.0
            signal['reason'] = 'wide_range_cautious_hold'

        logger.info(f"   📊 RANGE SIGNAL GENERATED: {signal['action']} ({signal['confidence']:.2f}) for {signal['symbol']} - {signal['reason']}")

        return signal

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
