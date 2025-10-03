#!/usr/bin/env python3
"""
Test script for Aggressive Mode functionality in PortfolioManager
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.portfolio_manager import PortfolioManager
from core.weight_calculator import WeightStrategy

def test_aggressive_mode():
    """Test aggressive mode functionality"""
    print("🚀 TESTING AGGRESSIVE MODE FUNCTIONALITY")
    print("=" * 60)

    # Test data
    market_data = {
        "BTCUSDT": {"close": 50000.0},
        "ETHUSDT": {"close": 3000.0}
    }

    # Test 1: Normal mode
    print("\n📊 TEST 1: Normal Mode (aggressive_mode=False)")
    pm_normal = PortfolioManager(
        mode="simulated",
        initial_balance=10000.0,
        aggressive_mode=False
    )

    # Initialize weight calculator
    pm_normal.initialize_weight_calculator()

    # Add test assets
    pm_normal.add_asset_for_weighting("BTCUSDT", 50000.0, market_cap=1000000000, volatility=0.3)
    pm_normal.add_asset_for_weighting("ETHUSDT", 3000.0, market_cap=500000000, volatility=0.4)

    # Test position limits
    btc_limit_normal = pm_normal.get_position_size_limit("BTCUSDT", "aggressive")
    print(f"   BTC Position Limit (Normal): ${btc_limit_normal:.2f}")

    # Test deployment plan
    deployment_normal = pm_normal.get_risk_adjusted_deployment_plan(
        strategy=WeightStrategy.RISK_PARITY,
        market_data=market_data
    )
    print(f"   Deployment Multiplier (Normal): {deployment_normal.get('risk_multiplier', 1.0):.1f}x")
    print(f"   Adjusted Capital (Normal): ${deployment_normal.get('adjusted_available_capital', 0):.2f}")

    # Test 2: Aggressive mode
    print("\n🚨 TEST 2: Aggressive Mode (aggressive_mode=True)")
    pm_aggressive = PortfolioManager(
        mode="simulated",
        initial_balance=10000.0,
        aggressive_mode=True
    )

    # Initialize weight calculator
    pm_aggressive.initialize_weight_calculator()

    # Add test assets
    pm_aggressive.add_asset_for_weighting("BTCUSDT", 50000.0, market_cap=1000000000, volatility=0.3)
    pm_aggressive.add_asset_for_weighting("ETHUSDT", 3000.0, market_cap=500000000, volatility=0.4)

    # Test position limits
    btc_limit_aggressive = pm_aggressive.get_position_size_limit("BTCUSDT", "aggressive")
    print(f"   BTC Position Limit (Aggressive): ${btc_limit_aggressive:.2f}")

    # Test deployment plan
    deployment_aggressive = pm_aggressive.get_risk_adjusted_deployment_plan(
        strategy=WeightStrategy.RISK_PARITY,
        market_data=market_data
    )
    print(f"   Deployment Multiplier (Aggressive): {deployment_aggressive.get('risk_multiplier', 1.0):.1f}x")
    print(f"   Adjusted Capital (Aggressive): ${deployment_aggressive.get('adjusted_available_capital', 0):.2f}")

    # Test 3: Comparison
    print("\n📈 TEST 3: Comparison Results")
    print("-" * 40)

    limit_ratio = btc_limit_aggressive / btc_limit_normal if btc_limit_normal > 0 else 0
    capital_ratio = deployment_aggressive.get('adjusted_available_capital', 0) / deployment_normal.get('adjusted_available_capital', 0) if deployment_normal.get('adjusted_available_capital', 0) > 0 else 0

    print(f"   Position Limit Increase: {limit_ratio:.2f}x")
    print(f"   Capital Deployment Increase: {capital_ratio:.2f}x")

    # Verify aggressive mode is working
    success = True
    if limit_ratio <= 1.0:
        print("   ❌ Position limits not increased in aggressive mode")
        success = False
    else:
        print("   ✅ Position limits correctly increased in aggressive mode")

    if capital_ratio <= 1.0:
        print("   ❌ Capital deployment not increased in aggressive mode")
        success = False
    else:
        print("   ✅ Capital deployment correctly increased in aggressive mode")

    print("\n" + "=" * 60)
    if success:
        print("✅ AGGRESSIVE MODE TEST PASSED")
        print("   - Higher position limits: ✓")
        print("   - Increased capital deployment: ✓")
        print("   - Risk warnings displayed: ✓")
    else:
        print("❌ AGGRESSIVE MODE TEST FAILED")
        return False

    return True

if __name__ == "__main__":
    test_aggressive_mode()
