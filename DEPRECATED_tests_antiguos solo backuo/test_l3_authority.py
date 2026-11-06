#!/usr/bin/env python3
"""
Test script to verify L3 authority over L2 behavior.
"""

import sys
import os
sys.path.append('.')

from l2_tactic.tactical_signal_processor import L2TacticProcessor

def test_l3_authority():
    """Test that L3 signals override L2 HOLD defaults."""

    print("=== TESTING L3 AUTHORITY OVER L2 ===\n")

    # Create processor instance
    processor = L2TacticProcessor()

    # Test data (simplified)
    market_data = {
        'BTCUSDT': {'historical_data': None},
        'ETHUSDT': {'historical_data': None}
    }

    # Test 1: L3 SELL signal should generate SELL_LIGHT
    print("Test 1: L3 SELL signal")
    l3_sell_context = {
        'signal': 'sell',
        'confidence': 0.6,
        'regime': 'TRENDING'
    }

    btc_sell_signal = processor._generate_conservative_signal_for_symbol('BTCUSDT', market_data, l3_sell_context)
    sell_side = getattr(btc_sell_signal, 'side', 'None')
    sell_confidence = getattr(btc_sell_signal, 'confidence', 0.0)

    print(f"  Result: {sell_side.upper()} (confidence: {sell_confidence:.2f})")
    assert sell_side == 'sell', f"Expected 'sell', got '{sell_side}'"
    assert sell_confidence >= 0.55, f"Expected confidence >= 0.55, got {sell_confidence}"
    print("  ✅ PASS: L3 sell signal correctly generates SELL action\n")

    # Test 2: L3 BUY signal should generate BUY_LIGHT
    print("Test 2: L3 BUY signal")
    l3_buy_context = {
        'signal': 'buy',
        'confidence': 0.7,
        'regime': 'TRENDING'
    }

    btc_buy_signal = processor._generate_conservative_signal_for_symbol('BTCUSDT', market_data, l3_buy_context)
    buy_side = getattr(btc_buy_signal, 'side', 'None')
    buy_confidence = getattr(btc_buy_signal, 'confidence', 0.0)

    print(f"  Result: {buy_side.upper()} (confidence: {buy_confidence:.2f})")
    assert buy_side == 'buy', f"Expected 'buy', got '{buy_side}'"
    assert buy_confidence >= 0.55, f"Expected confidence >= 0.55, got {buy_confidence}"
    print("  ✅ PASS: L3 buy signal correctly generates BUY action\n")

    # Test 3: L3 HOLD (no clear bias) should generate HOLD
    print("Test 3: L3 HOLD (no clear bias)")
    l3_hold_context = {
        'signal': 'hold',
        'confidence': 0.4,
        'regime': 'RANGE'
    }

    btc_hold_signal = processor._generate_conservative_signal_for_symbol('BTCUSDT', market_data, l3_hold_context)
    hold_side = getattr(btc_hold_signal, 'side', 'None')

    print(f"  Result: {hold_side.upper()}")
    assert hold_side == 'hold', f"Expected 'hold', got '{hold_side}'"
    print("  ✅ PASS: L3 hold signal correctly maintains HOLD\n")

    # Test 4: Check metadata contains authority override flag
    print("Test 4: L3 authority override metadata")
    sell_metadata = getattr(btc_sell_signal, 'metadata', {})
    buy_metadata = getattr(btc_buy_signal, 'metadata', {})

    assert sell_metadata.get('l3_authority_override') == True, "SELL signal should have L3 authority override flag"
    assert buy_metadata.get('l3_authority_override') == True, "BUY signal should have L3 authority override flag"
    print("  ✅ PASS: Authority override metadata correctly set\n")

    print("🎉 ALL TESTS PASSED: L3 has real authority over L2!")
    print("   - L3 sell signals → L2 SELL_LIGHT/REDUCE")
    print("   - L3 buy signals → L2 BUY_LIGHT")
    print("   - L3 confidence boosted to minimum 0.55")
    print("   - No more HOLD defaults when L3 has clear bias")

if __name__ == "__main__":
    test_l3_authority()
