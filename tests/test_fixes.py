#!/usr/bin/env python3
"""
Test script to verify BLIND MODE and INV-5 rule fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_blind_mode():
    """Test BLIND MODE fix"""
    print("🧪 Testing BLIND MODE fix...")

    from l3_strategy.decision_maker import make_decision

    # Test BLIND MODE (balances_synced=False)
    blind_decision = make_decision({}, balances_synced=False)

    # Verify BLIND MODE properties
    assert blind_decision['market_opinion'] == 'hold', f"Expected 'hold', got {blind_decision['market_opinion']}"
    assert blind_decision['opinion_confidence'] == 0.0, f"Expected 0.0, got {blind_decision['opinion_confidence']}"
    assert blind_decision['allow_l2_signals'] == False, f"Expected False, got {blind_decision['allow_l2_signals']}"
    assert blind_decision['strategic_control']['blind_mode_active'] == True, "Expected True for blind_mode_active"
    assert blind_decision['strategic_control']['force_hold_signals'] == True, "Expected True for force_hold_signals"

    print("✅ BLIND MODE fix verified - generates HOLD with confidence 0.0")
    return True

def test_inv5_rule():
    """Test INV-5 rule: doubt → HOLD"""
    print("🧪 Testing INV-5 rule (doubt → HOLD)...")

    from l2_tactic.tactical_signal_processor import L2TacticProcessor

    processor = L2TacticProcessor()

    # Test with low confidence L3 context (< 0.6)
    l3_context_low_conf = {
        'confidence': 0.3,  # Below 0.6 threshold
        'regime': 'RANGE',
        'signal': 'hold'
    }

    # Mock market data
    market_data = {
        'BTCUSDT': {
            'historical_data': None  # Will trigger early return
        }
    }

    # This should trigger the INV-5 rule and return HOLD
    signal = processor._generate_conservative_signal_for_symbol('BTCUSDT', market_data, l3_context_low_conf)

    assert signal is not None, "Expected HOLD signal from INV-5 rule"
    assert getattr(signal, 'side', None) == 'hold', f"Expected 'hold', got {getattr(signal, 'side', None)}"
    assert signal.metadata.get('inv5_rule_applied') == True, "Expected inv5_rule_applied=True"

    print("✅ INV-5 rule verified - generates HOLD when confidence < 0.6")
    return True

def test_normal_operation():
    """Test that normal operation still works"""
    print("🧪 Testing normal operation...")

    from l3_strategy.decision_maker import make_decision

    # Test normal mode (balances_synced=True)
    normal_decision = make_decision({}, balances_synced=True)

    # Should not be in BLIND MODE
    assert normal_decision.get('strategic_control', {}).get('blind_mode_active') != True, "Should not be in BLIND MODE"

    print("✅ Normal operation verified")
    return True

if __name__ == "__main__":
    print("🧪 Running HRM System Fixes Test Suite")
    print("=" * 50)

    try:
        # Run tests
        test_blind_mode()
        test_inv5_rule()
        test_normal_operation()

        print("\n🎉 ALL TESTS PASSED!")
        print("✅ BLIND MODE fix: L3 generates HOLD when balance sync fails")
        print("✅ INV-5 rule fix: System generates HOLD when confidence < 0.6")
        print("✅ Normal operation: System works correctly when balances synced")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
