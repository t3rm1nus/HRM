#!/usr/bin/env python3
"""
Test script to verify L2 considers L3 setup trades correctly.
Tests the bug fix where L2 was ignoring L3 oversold setups in RANGE regime.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.models import TacticalSignal
import pandas as pd
import numpy as np
from datetime import datetime

def create_mock_market_data():
    """Create mock market data for testing."""
    # Create sample OHLCV data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)

    # Generate realistic price data with some trend
    base_price = 50000
    prices = []
    for i in range(100):
        change = np.random.normal(0, 0.02)  # 2% volatility
        base_price *= (1 + change)
        prices.append(base_price)

    # Create OHLCV DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(100, 1000) for _ in range(100)]
    }, index=dates)

    return {
        'BTCUSDT': {
            'historical_data': df
        },
        'ETHUSDT': {
            'historical_data': df.copy() * 0.1  # ETH at ~5000
        }
    }

def test_l2_setup_trade_logic():
    """Test that L2 generates BUY signals when L3 detects oversold setup in RANGE regime."""

    print("🧪 TESTING L2 SETUP TRADE LOGIC FIX")
    print("=" * 50)

    # Initialize L2 processor
    processor = L2TacticProcessor()

    # Create mock market data
    market_data = create_mock_market_data()

    # Test Case 1: RANGE regime WITHOUT setup (should return HOLD)
    print("\n📊 Test Case 1: RANGE regime without setup (should HOLD)")
    l3_context_range_no_setup = {
        'regime': 'RANGE',
        'confidence': 0.5,
        'allow_l2_signals': False,
        'setup_type': None,
        'allow_setup_trades': False
    }

    signals = processor.generate_signals_conservative(market_data, l3_context_range_no_setup)

    for signal in signals:
        symbol = getattr(signal, 'symbol', 'UNKNOWN')
        side = getattr(signal, 'side', 'unknown')
        print(f"  {symbol}: {side.upper()} (expected: HOLD)")

        if side != 'hold':
            print(f"  ❌ ERROR: Expected HOLD but got {side.upper()}")
            return False

    # Test Case 2: RANGE regime WITH oversold setup (should generate BUY)
    print("\n📊 Test Case 2: RANGE regime with oversold setup (should BUY)")
    l3_context_range_oversold = {
        'regime': 'RANGE',
        'confidence': 0.6,
        'allow_l2_signals': True,
        'setup_type': 'oversold',
        'allow_setup_trades': True,
        'max_allocation_for_setup': 0.10
    }

    signals = processor.generate_signals_conservative(market_data, l3_context_range_oversold)

    buy_signals_found = 0
    for signal in signals:
        symbol = getattr(signal, 'symbol', 'UNKNOWN')
        side = getattr(signal, 'side', 'unknown')
        confidence = getattr(signal, 'confidence', 0.0)
        metadata = getattr(signal, 'metadata', {})

        print(f"  {symbol}: {side.upper()} (conf={confidence:.2f})")

        if side == 'buy':
            buy_signals_found += 1
            # Check metadata indicates setup trade
            if 'l3_setup_type' in metadata and metadata['l3_setup_type'] == 'oversold':
                print("    ✅ Correctly identified as oversold setup trade")
            else:
                print(f"    ❌ Missing setup trade metadata: {metadata}")
                return False
        elif side == 'hold':
            print("    ℹ️ HOLD signal (acceptable if conditions not met)")
    if buy_signals_found == 0:
        print("  ❌ ERROR: Expected at least one BUY signal for oversold setup")
        return False

    # Test Case 3: BULL regime (should work normally)
    print("\n📊 Test Case 3: BULL regime (should work normally)")
    l3_context_bull = {
        'regime': 'BULL',
        'confidence': 0.9,
        'allow_l2_signals': True,
        'setup_type': None
    }

    signals = processor.generate_signals_conservative(market_data, l3_context_bull)

    for signal in signals:
        symbol = getattr(signal, 'symbol', 'UNKNOWN')
        side = getattr(signal, 'side', 'unknown')
        print(f"  {symbol}: {side.upper()} (BULL regime - can be BUY/SELL/HOLD)")

    print("\n✅ ALL TESTS PASSED: L2 correctly considers L3 setup trades!")
    return True

if __name__ == "__main__":
    success = test_l2_setup_trade_logic()
    if success:
        print("\n🎉 BUG FIX VERIFIED: L2 now properly handles L3 oversold setups in RANGE regime")
    else:
        print("\n❌ TESTS FAILED: Bug fix may not be working correctly")
        sys.exit(1)
