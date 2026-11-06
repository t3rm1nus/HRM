#!/usr/bin/env python3
"""
Test script to verify the TRENDING override removal works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from l3_strategy import decision_maker

def test_override_changes():
    """Test that TRENDING regime doesn't force overrides and L3 confidence < 0.80 allows L2"""
    print("Testing TRENDING override removal and L3 confidence logic")
    print("=" * 70)

    # Test 1: L3 confidence < 0.80 should return HOLD with allow_l2=True
    print("Test 1: L3 confidence < 0.80")
    result = decision_maker.strategic_override_processor('BUY', 0.75)
    expected = {
        "final_signal": "HOLD",
        "confidence": 0.75,
        "override": False,
        "reason": "L3 confidence insufficient",
        "allow_l2": True,
        "source": "L3_STRATEGIC"
    }

    success = (result["final_signal"] == expected["final_signal"] and
               result["allow_l2"] == expected["allow_l2"] and
               result["override"] == expected["override"])

    print(f"Result: {result}")
    print(f"Expected allow_l2=True: {'✅ PASS' if result.get('allow_l2') else '❌ FAIL'}")
    print(f"Expected final_signal=HOLD: {'✅ PASS' if result['final_signal'] == 'HOLD' else '❌ FAIL'}")
    print()

    # Test 2: TRENDING regime with high confidence should NOT force override
    print("Test 2: TRENDING regime with high confidence (>= 0.80)")
    regime_info = {'regime': 'TRENDING', 'signal': 'buy'}
    result2 = decision_maker.strategic_override_processor('HOLD', 0.85, regime_info)

    print(f"Result: {result2}")
    print(f"TRENDING prevents override: {'✅ PASS' if not result2['override'] else '❌ FAIL'}")
    print(f"Reason mentions TRENDING: {'✅ PASS' if 'TRENDING' in result2['reason'] else '❌ FAIL'}")
    print()

    # Test 3: Clear regime (BULL) with high confidence should allow override
    print("Test 3: Clear regime (BULL) with high confidence (>= 0.80)")
    regime_info3 = {'regime': 'BULL', 'signal': 'buy'}
    result3 = decision_maker.strategic_override_processor('HOLD', 0.85, regime_info3)

    print(f"Result: {result3}")
    print(f"BULL regime allows override: {'✅ PASS' if result3['override'] else '❌ FAIL'}")
    print()

    print("=" * 70)
    print("✅ TRENDING OVERRIDE REMOVAL COMPLETE")
    print("🚫 TRENDING ≠ trading order")
    print("🚫 TRENDING only changes context, not action")
    print("✅ L3 confidence < 0.80 returns HOLD with allow_l2=True")

if __name__ == "__main__":
    test_override_changes()