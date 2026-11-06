#!/usr/bin/env python3
"""
Test script to verify L3 strategic control in RANGE regimes
"""

from l3_strategy.decision_maker import make_decision
import json

def test_l3_strategic_control():
    """Test L3 strategic control with RANGE regimes and confidence overrides"""

    print("🧪 Testing L3 Strategic Control in RANGE Regimes")
    print("=" * 60)

    # Test Case 1: RANGE regime with low confidence (< 0.7)
    print("\n📊 Test Case 1: RANGE regime with confidence = 0.5 (< 0.7)")
    regime_decision_low_conf = {
        'regime': 'RANGE',
        'subtype': 'NORMAL_RANGE',
        'signal': 'hold',
        'confidence': 0.5,
        'setup_type': None,
        'allow_l2_signal': True  # Initial value
    }

    decision_low_conf = make_decision(
        inputs={},
        regime_decision=regime_decision_low_conf
    )

    print(f"  allow_l2_signals: {decision_low_conf['allow_l2_signals']}")
    print(f"  strategic_hold_active: {decision_low_conf['strategic_control']['strategic_hold_active']}")
    print(f"  strategic_hold_type: {decision_low_conf['strategic_control']['strategic_hold_type']}")
    print(f"  capital_preservation_mode: {decision_low_conf['strategic_control']['capital_preservation_mode']}")

    # Test Case 2: RANGE regime with high confidence (>= 0.7)
    print("\n📊 Test Case 2: RANGE regime with confidence = 0.8 (>= 0.7)")
    regime_decision_high_conf = {
        'regime': 'RANGE',
        'subtype': 'NORMAL_RANGE',
        'signal': 'hold',
        'confidence': 0.8,
        'setup_type': None,
        'allow_l2_signal': True  # Initial value
    }

    decision_high_conf = make_decision(
        inputs={},
        regime_decision=regime_decision_high_conf
    )

    print(f"  allow_l2_signals: {decision_high_conf['allow_l2_signals']}")
    print(f"  strategic_hold_active: {decision_high_conf['strategic_control']['strategic_hold_active']}")
    print(f"  strategic_hold_type: {decision_high_conf['strategic_control']['strategic_hold_type']}")
    print(f"  capital_preservation_mode: {decision_high_conf['strategic_control']['capital_preservation_mode']}")

    # Test Case 3: RANGE regime with oversold setup
    print("\n📊 Test Case 3: RANGE regime with OVERSOLD setup")
    regime_decision_oversold = {
        'regime': 'RANGE',
        'subtype': 'OVERSOLD_SETUP',
        'signal': 'buy',
        'confidence': 0.6,
        'setup_type': 'oversold',
        'allow_l2_signal': False  # Initial value
    }

    decision_oversold = make_decision(
        inputs={},
        regime_decision=regime_decision_oversold
    )

    print(f"  allow_l2_signals: {decision_oversold['allow_l2_signals']}")
    print(f"  strategic_hold_active: {decision_oversold['strategic_control']['strategic_hold_active']}")
    print(f"  setup_detected: {decision_oversold['setup_detected']}")
    print(f"  setup_type: {decision_oversold['setup_type']}")

    # Test Case 4: TRENDING regime (should not trigger strategic hold)
    print("\n📊 Test Case 4: TRENDING regime (should not trigger strategic hold)")
    regime_decision_trending = {
        'regime': 'TRENDING',
        'subtype': 'STRONG_BULL',
        'signal': 'buy',
        'confidence': 0.5,
        'setup_type': None,
        'allow_l2_signal': True
    }

    decision_trending = make_decision(
        inputs={},
        regime_decision=regime_decision_trending
    )

    print(f"  allow_l2_signals: {decision_trending['allow_l2_signals']}")
    print(f"  strategic_hold_active: {decision_trending['strategic_control']['strategic_hold_active']}")
    print(f"  market_regime: {decision_trending['market_regime']}")

    print("\n" + "=" * 60)
    print("📋 VERIFICATION RESULTS:")

    # Verify acceptance criteria
    criteria_met = []

    # Criterion 1: L3 generates explicit HOLD in RANGE with low confidence
    if decision_low_conf['strategic_control']['strategic_hold_active'] and \
       decision_low_conf['strategic_control']['strategic_hold_type'] == "HOLD_L3_CONFIDENCE_OVERRIDE":
        criteria_met.append("✅ L3 generates HOLD_L3_CONFIDENCE_OVERRIDE in RANGE < 0.7 confidence")
    else:
        criteria_met.append("❌ L3 does not generate HOLD_L3_CONFIDENCE_OVERRIDE in RANGE < 0.7 confidence")

    # Criterion 2: allow_l2_signals = False in RANGE regimes (unless setup)
    if not decision_low_conf['allow_l2_signals'] and decision_low_conf['market_regime'] == 'RANGE':
        criteria_met.append("✅ allow_l2_signals = False in RANGE regime (low confidence)")
    else:
        criteria_met.append("❌ allow_l2_signals not properly blocked in RANGE regime")

    if not decision_high_conf['allow_l2_signals'] and decision_high_conf['market_regime'] == 'RANGE':
        criteria_met.append("✅ allow_l2_signals = False in RANGE regime (high confidence)")
    else:
        criteria_met.append("❌ allow_l2_signals not properly blocked in RANGE regime")

    # Criterion 3: Setup override works
    if decision_oversold['allow_l2_signals'] and decision_oversold['setup_detected']:
        criteria_met.append("✅ Setup override allows L2 signals in RANGE regime")
    else:
        criteria_met.append("❌ Setup override not working properly")

    # Criterion 4: TRENDING regime not affected
    if not decision_trending['strategic_control']['strategic_hold_active'] and decision_trending['market_regime'] == 'TRENDING':
        criteria_met.append("✅ TRENDING regime not affected by strategic hold logic")
    else:
        criteria_met.append("❌ TRENDING regime incorrectly affected by strategic hold logic")

    print("\n".join(criteria_met))

    all_criteria_met = all("✅" in criterion for criterion in criteria_met)
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL CRITERIA MET' if all_criteria_met else '❌ SOME CRITERIA FAILED'}")

    return all_criteria_met

if __name__ == "__main__":
    success = test_l3_strategic_control()
    exit(0 if success else 1)