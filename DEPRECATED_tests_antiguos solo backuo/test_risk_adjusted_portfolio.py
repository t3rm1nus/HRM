#!/usr/bin/env python3
"""
Test script for risk-adjusted portfolio management
Tests the integration between risk_manager.py output and portfolio_manager.py
"""

import os
import json
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append('.')

from core.portfolio_manager import PortfolioManager

def create_test_risk_data(risk_appetite: str):
    """Create test risk data file"""
    os.makedirs("data/datos_inferencia", exist_ok=True)

    risk_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "inputs": {
            "volatility": 0.45,
            "sentiment": 0.2,
            "regime": "bull"
        },
        "risk_appetite": risk_appetite
    }

    with open("data/datos_inferencia/risk.json", "w") as f:
        json.dump(risk_data, f, indent=2)

    print(f"✅ Created test risk data with appetite: {risk_appetite}")

def test_risk_adjusted_capital():
    """Test risk-adjusted capital deployment"""
    print("🧪 TESTING RISK-ADJUSTED PORTFOLIO MANAGEMENT")
    print("=" * 60)

    # Test different risk appetites
    test_cases = [
        ("low", 0.40),
        ("moderate", 0.60),
        ("high", 0.80),
        ("aggressive", 0.90)
    ]

    for risk_appetite, expected_deployment in test_cases:
        print(f"\n🎯 Testing Risk Appetite: {risk_appetite.upper()}")

        # Create test risk data
        create_test_risk_data(risk_appetite)

        # Initialize portfolio manager
        pm = PortfolioManager(mode="simulated", initial_balance=3000.0)

        # Test risk appetite loading
        loaded_appetite = pm.load_risk_appetite()
        print(f"   Loaded risk appetite: {loaded_appetite}")

        assert loaded_appetite == risk_appetite, f"Expected {risk_appetite}, got {loaded_appetite}"

        # Test capital deployment (mock market data)
        market_data = {
            "BTCUSDT": {"close": 50000.0},
            "ETHUSDT": {"close": 3000.0}
        }

        available_capital = pm.get_available_trading_capital(market_data)
        expected_capital = 3000.0 * expected_deployment

        print(f"   Available capital: ${available_capital:.2f} (expected: ${expected_capital:.2f})")

        # Allow small floating point differences
        assert abs(available_capital - expected_capital) < 0.01, f"Capital mismatch: {available_capital} vs {expected_capital}"

        # Test deployment status
        status = pm.get_capital_deployment_status(market_data)
        print(f"   Deployment percentage: {status['deployment_percentage']:.1%}")
        print(f"   Can deploy more: {status['can_deploy_more']}")

        assert status['risk_appetite'] == risk_appetite
        assert abs(status['deployment_percentage'] - expected_deployment) < 0.001

        print(f"   ✅ {risk_appetite.upper()} test passed")

    print("\n" + "=" * 60)
    print("🎉 ALL RISK-ADJUSTED PORTFOLIO TESTS PASSED!")
    print("\n📊 Risk Appetite Deployment Tiers:")
    print("   Low: 40% of USDT available for trading")
    print("   Moderate: 60% of USDT available for trading")
    print("   High: 80% of USDT available for trading")
    print("   Aggressive: 90% of USDT available for trading")

def test_position_sizing():
    """Test risk-adjusted position sizing"""
    print("\n🧪 TESTING RISK-ADJUSTED POSITION SIZING")
    print("=" * 60)

    # Test position sizing with different risk levels
    test_cases = [
        ("low", 0.5),       # 50% of base size
        ("moderate", 0.8),  # 80% of base size
        ("high", 1.0),      # 100% of base size
        ("aggressive", 1.2) # 120% of base size
    ]

    base_position_size = 1000.0  # $1000 base position
    signal_strength = 0.8        # Strong signal

    market_data = {
        "BTCUSDT": {"close": 50000.0},
        "ETHUSDT": {"close": 3000.0}
    }

    for risk_appetite, expected_multiplier in test_cases:
        print(f"\n🎯 Testing Position Sizing - Risk: {risk_appetite.upper()}")

        # Create test risk data
        create_test_risk_data(risk_appetite)

        # Initialize portfolio manager
        pm = PortfolioManager(mode="simulated", initial_balance=3000.0)

        # Test position sizing
        adjusted_size = pm.get_risk_adjusted_position_size(
            signal_strength=signal_strength,
            base_position_size=base_position_size,
            market_data=market_data
        )

        # Calculate expected size: base_size * risk_multiplier * signal_adjustment
        signal_adjustment = 0.5 + (signal_strength * 0.5)  # 0.5 + (0.8 * 0.5) = 0.9
        expected_size = base_position_size * expected_multiplier * signal_adjustment

        print(f"   Base size: ${base_position_size:.2f}")
        print(f"   Signal strength: {signal_strength:.2f} (adjustment: {signal_adjustment:.2f})")
        print(f"   Risk multiplier: {expected_multiplier:.1f}")
        print(f"   Adjusted size: ${adjusted_size:.2f} (expected: ${expected_size:.2f})")

        # Allow small differences due to minimum size constraints
        assert abs(adjusted_size - expected_size) < 1.0, f"Position size mismatch: {adjusted_size} vs {expected_size}"

        print(f"   ✅ Position sizing for {risk_appetite.upper()} passed")

    print("\n" + "=" * 60)
    print("🎉 ALL POSITION SIZING TESTS PASSED!")

if __name__ == "__main__":
    try:
        test_risk_adjusted_capital()
        test_position_sizing()

        print("\n" + "=" * 60)
        print("🎊 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n💡 Risk-adjusted portfolio management is working correctly.")
        print("   The system will now dynamically adjust capital deployment")
        print("   based on the risk appetite determined by the L3 strategy layer.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
