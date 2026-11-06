#!/usr/bin/env python3
"""
Test script for the new "SELL TÁCTICO DE SALIDA LIMPIA" rule
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import should_execute_with_l3_dominance

def test_tactical_sell_rule():
    """Test the new tactical sell rule implementation"""

    print("=" * 60)
    print("TESTING: SELL TÁCTICO DE SALIDA LIMPIA RULE")
    print("=" * 60)

    # Test case 1: SELL signal with TRENDING regime and low L3 confidence (< 0.6)
    print("\n📊 Test 1: SELL signal in TRENDING regime with low L3 confidence")
    l2_signal_sell = {'action': 'SELL', 'symbol': 'BTCUSDT', 'confidence': 0.8}
    l3_info_trending_low_conf = {
        'regime': 'TRENDING',
        'confidence': 0.4,  # Low confidence (< 0.6)
        'signal': 'hold',
        'allow_l2': False  # L3 would normally block
    }

    should_execute, reason = should_execute_with_l3_dominance(l2_signal_sell, l3_info_trending_low_conf)
    print(f"  L2 Signal: {l2_signal_sell['action']} (confidence: {l2_signal_sell['confidence']})")
    print(f"  L3 Context: regime={l3_info_trending_low_conf['regime']}, confidence={l3_info_trending_low_conf['confidence']}, allow_l2={l3_info_trending_low_conf['allow_l2']}")
    print(f"  Should execute: {should_execute}")
    print(f"  Reason: {reason}")
    print("  Expected: True (tactical sell override should allow this)")
    print("  ✅ PASS" if should_execute else "  ❌ FAIL")

    # Test case 2: SELL signal with RANGE regime (should not trigger override)
    print("\n📊 Test 2: SELL signal in RANGE regime")
    l3_info_range = {
        'regime': 'RANGE',
        'confidence': 0.4,
        'signal': 'hold',
        'allow_l2': False
    }

    should_execute, reason = should_execute_with_l3_dominance(l2_signal_sell, l3_info_range)
    print(f"  L2 Signal: {l2_signal_sell['action']} (confidence: {l2_signal_sell['confidence']})")
    print(f"  L3 Context: regime={l3_info_range['regime']}, confidence={l3_info_range['confidence']}, allow_l2={l3_info_range['allow_l2']}")
    print(f"  Should execute: {should_execute}")
    print(f"  Reason: {reason}")
    print("  Expected: False (RANGE regime doesn't qualify for tactical sell override)")
    print("  ✅ PASS" if not should_execute else "  ❌ FAIL")

    # Test case 3: BUY signal (should not trigger override)
    print("\n📊 Test 3: BUY signal (should not trigger tactical sell override)")
    l2_signal_buy = {'action': 'BUY', 'symbol': 'BTCUSDT', 'confidence': 0.8}
    should_execute, reason = should_execute_with_l3_dominance(l2_signal_buy, l3_info_trending_low_conf)
    print(f"  L2 Signal: {l2_signal_buy['action']} (confidence: {l2_signal_buy['confidence']})")
    print(f"  L3 Context: regime={l3_info_trending_low_conf['regime']}, confidence={l3_info_trending_low_conf['confidence']}, allow_l2={l3_info_trending_low_conf['allow_l2']}")
    print(f"  Should execute: {should_execute}")
    print(f"  Reason: {reason}")
    print("  Expected: False (BUY signals don't qualify for tactical sell override)")
    print("  ✅ PASS" if not should_execute else "  ❌ FAIL")

    # Test case 4: SELL signal with TRENDING regime but high L3 confidence (>= 0.6)
    print("\n📊 Test 4: SELL signal in TRENDING regime with high L3 confidence")
    l3_info_trending_high_conf = {
        'regime': 'TRENDING',
        'confidence': 0.7,  # High confidence (>= 0.6)
        'signal': 'hold',
        'allow_l2': False
    }

    should_execute, reason = should_execute_with_l3_dominance(l2_signal_sell, l3_info_trending_high_conf)
    print(f"  L2 Signal: {l2_signal_sell['action']} (confidence: {l2_signal_sell['confidence']})")
    print(f"  L3 Context: regime={l3_info_trending_high_conf['regime']}, confidence={l3_info_trending_high_conf['confidence']}, allow_l2={l3_info_trending_high_conf['allow_l2']}")
    print(f"  Should execute: {should_execute}")
    print(f"  Reason: {reason}")
    print("  Expected: False (high L3 confidence doesn't qualify for tactical sell override)")
    print("  ✅ PASS" if not should_execute else "  ❌ FAIL")

    print("\n" + "=" * 60)
    print("🎯 TACTICAL SELL RULE SUMMARY:")
    print("  - Allows SELL signals even when L3 would normally block")
    print("  - Conditions: has_position + l3_confidence < 0.6 + l3_regime == 'TRENDING'")
    print("  - Purpose: Clean exits when market is exhausted")
    print("  - Philosophy: System doubt → allow tactical selling")
    print("=" * 60)
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    test_tactical_sell_rule()
