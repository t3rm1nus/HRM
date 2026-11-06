#!/usr/bin/env python3
"""
Test script for TRENDING regime low confidence fix
"""

from l3_strategy.decision_maker import make_decision

def test_trending_low_confidence():
    """Test that TRENDING with confidence < 0.5 allows setup trades"""

    print("🧪 Testing TRENDING Low Confidence Rule")
    print("=" * 50)

    # Test case 1: TRENDING with confidence < 0.5 and oversold setup
    regime_decision_low_conf = {
        'regime': 'TRENDING',
        'confidence': 0.4,
        'signal': 'hold',
        'setup_type': 'oversold'
    }

    decision = make_decision({}, regime_decision=regime_decision_low_conf)
    print('Test 1 - TRENDING confidence 0.4 with oversold setup:')
    print(f'  allow_l2_signals: {decision["allow_l2_signals"]}')
    print(f'  strategic_hold_active: {decision["strategic_hold_active"]}')
    print(f'  setup_type: {decision.get("setup_type")}')
    print(f'  strategic_control allow_setup_trades: {decision["strategic_control"].get("allow_setup_trades", False)}')
    print(f'  strategic_control low_confidence_trending: {decision["strategic_control"].get("low_confidence_trending", False)}')
    print()

    # Test case 2: TRENDING with confidence > 0.5 (should use default logic)
    regime_decision_high_conf = {
        'regime': 'TRENDING',
        'confidence': 0.6,
        'signal': 'hold',
        'setup_type': None
    }

    decision2 = make_decision({}, regime_decision=regime_decision_high_conf)
    print('Test 2 - TRENDING confidence 0.6 without setup:')
    print(f'  allow_l2_signals: {decision2["allow_l2_signals"]}')
    print(f'  strategic_hold_active: {decision2["strategic_hold_active"]}')
    print(f'  strategic_control allow_setup_trades: {decision2["strategic_control"].get("allow_setup_trades", False)}')
    print(f'  strategic_control low_confidence_trending: {decision2["strategic_control"].get("low_confidence_trending", False)}')
    print()

    # Test case 3: TRENDING with confidence < 0.5 but no setup
    regime_decision_low_conf_no_setup = {
        'regime': 'TRENDING',
        'confidence': 0.3,
        'signal': 'hold',
        'setup_type': None
    }

    decision3 = make_decision({}, regime_decision=regime_decision_low_conf_no_setup)
    print('Test 3 - TRENDING confidence 0.3 without setup:')
    print(f'  allow_l2_signals: {decision3["allow_l2_signals"]}')
    print(f'  strategic_hold_active: {decision3["strategic_hold_active"]}')
    print(f'  strategic_control allow_setup_trades: {decision3["strategic_control"].get("allow_setup_trades", False)}')
    print(f'  strategic_control low_confidence_trending: {decision3["strategic_control"].get("low_confidence_trending", False)}')
    print()

    # Validation
    print("✅ VALIDATION:")
    test1_pass = (
        decision["allow_l2_signals"] == True and
        decision["strategic_hold_active"] == False and
        decision["strategic_control"].get("allow_setup_trades") == True and
        decision["strategic_control"].get("low_confidence_trending") == True
    )
    print(f"  Test 1 (TRENDING 0.4 + oversold): {'✅ PASS' if test1_pass else '❌ FAIL'}")

    test2_pass = (
        decision2["strategic_control"].get("low_confidence_trending") != True  # Should not trigger low confidence rule
    )
    print(f"  Test 2 (TRENDING 0.6 no setup): {'✅ PASS' if test2_pass else '❌ FAIL'}")

    test3_pass = (
        decision3["strategic_control"].get("low_confidence_trending") == True and
        decision3["allow_l2_signals"] == True  # Even without setup, should allow L2 signals
    )
    print(f"  Test 3 (TRENDING 0.3 no setup): {'✅ PASS' if test3_pass else '❌ FAIL'}")

    all_pass = test1_pass and test2_pass and test3_pass
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")

if __name__ == "__main__":
    test_trending_low_confidence()
