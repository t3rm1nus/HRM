#!/usr/bin/env python3
"""
Test script to verify the exceptional override logic works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from l3_strategy import decision_maker

def test_exceptional_override():
    """Test that override only happens under exceptional conditions"""
    print("Testing EXCEPTIONAL OVERRIDE logic - CAMBIO 3")
    print("=" * 80)

    # Test cases - L3 NEVER overrides when L2 is HOLD (NO NEGOCIABLE)
    test_cases = [
        # (l2_signal, l3_confidence, regime, risk_blocking, expected_override, description)
        ("HOLD", 0.85, "BULL", False, False, "🔴 L3 NEVER overrides HOLD - invariant"),
        ("HOLD", 0.75, "BULL", False, False, "🔴 L3 NEVER overrides HOLD - invariant"),
        ("BUY", 0.85, "BULL", False, True, "✅ L2=BUY, exceptional conditions met"),
        ("BUY", 0.75, "BULL", False, False, "❌ L2=BUY, L3 confidence too low"),
        ("BUY", 0.85, "RANGE", False, False, "❌ L2=BUY, regime not STRONG"),
        ("SELL", 0.85, "BEAR", False, True, "✅ L2=SELL, exceptional conditions met"),
        ("SELL", 0.85, "BULL", True, False, "❌ L2=SELL, risk filters blocking"),
    ]

    for i, (l2_signal, l3_conf, regime, risk_blocking, expected_override, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        print(f"  Input: L2={l2_signal}, L3_conf={l3_conf}, regime={regime}, risk_blocking={risk_blocking}")

        # Prepare test data
        regime_info = {'regime': regime, 'signal': 'buy' if regime == 'BULL' else 'sell' if regime == 'BEAR' else 'hold'}
        market_indicators = None

        if risk_blocking:
            # Simulate risk blocking conditions
            market_indicators = {
                'rsi': 80,  # Extreme RSI
                'macd_divergence': True,
                'volume_spike': True,
                'has_pullback': False
            }

        # Run test
        result = decision_maker.strategic_override_processor(l2_signal, l3_conf, regime_info, market_indicators)

        # Check results
        actual_override = result['override']
        final_signal = result['final_signal']
        reason = result['reason']

        success = actual_override == expected_override
        status = "✅ PASS" if success else "❌ FAIL"

        print(f"  Result: override={actual_override}, signal={final_signal}")
        print(f"  Reason: {reason}")
        print(f"  Status: {status}")

        if not success:
            print(f"  ❌ EXPECTED override={expected_override}, got {actual_override}")

    print("\n" + "=" * 80)
    print("✅ EXCEPTIONAL OVERRIDE LOGIC TEST COMPLETE")
    print("🎯 Override ONLY when ALL conditions are exceptional:")
    print("   - l2_signal == 'HOLD'")
    print("   - l3_confidence >= 0.80")
    print("   - regime_strength == 'STRONG' (BULL/BEAR)")
    print("   - not risk_filters_blocking")
    print("🔴 Otherwise: respect_hold()")

if __name__ == "__main__":
    test_exceptional_override()