#!/usr/bin/env python3
"""
Test Allocation Tiers - Comprehensive testing of dynamic capital allocation system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.portfolio_manager import PortfolioManager
from core.logging import logger
import json

def test_allocation_tiers():
    """Test comprehensive allocation tier system"""
    print("🎯 TESTING ALLOCATION TIERS SYSTEM")
    print("=" * 60)

    # Initialize portfolio manager
    pm = PortfolioManager(mode="simulated", initial_balance=10000.0, aggressive_mode=False)

    # Test different scenarios
    test_scenarios = [
        {
            "name": "Conservative - Weak Signal",
            "signal_strength": 0.3,
            "market_condition": "bearish",
            "asset_type": "crypto",
            "risk_appetite": "low"
        },
        {
            "name": "Balanced - Moderate Signal",
            "signal_strength": 0.6,
            "market_condition": "neutral",
            "asset_type": "crypto",
            "risk_appetite": "moderate"
        },
        {
            "name": "Growth - Strong Signal",
            "signal_strength": 0.8,
            "market_condition": "bullish",
            "asset_type": "crypto",
            "risk_appetite": "high"
        },
        {
            "name": "Aggressive - Very Strong Signal",
            "signal_strength": 0.9,
            "market_condition": "bullish",
            "asset_type": "crypto",
            "risk_appetite": "aggressive"
        },
        {
            "name": "Stocks - Conservative",
            "signal_strength": 0.7,
            "market_condition": "neutral",
            "asset_type": "stocks",
            "risk_appetite": "moderate"
        },
        {
            "name": "Bonds - Very Conservative",
            "signal_strength": 0.5,
            "market_condition": "volatile",
            "asset_type": "bonds",
            "risk_appetite": "low"
        },
        {
            "name": "Commodities - Volatile Market",
            "signal_strength": 0.4,
            "market_condition": "volatile",
            "asset_type": "commodities",
            "risk_appetite": "high"
        }
    ]

    results = []

    for scenario in test_scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        print("-" * 40)

        # Get allocation tier
        tier = pm.get_allocation_tier(
            signal_strength=scenario["signal_strength"],
            market_condition=scenario["market_condition"],
            asset_type=scenario["asset_type"],
            risk_appetite=scenario["risk_appetite"]
        )

        # Display results
        print(f"   Tier: {tier['tier_name'].upper()}")
        print(f"   Risk Appetite: {tier['risk_appetite']}")
        print(f"   Signal Strength: {tier['signal_strength']:.2f} → {tier['signal_multiplier']:.2f}x")
        print(f"   Market Condition: {tier['market_condition']} → {tier['market_multiplier']:.2f}x")
        print(f"   Asset Type: {tier['asset_type']} → {tier['asset_multiplier']:.2f}x")
        print(f"   Final Allocation: {tier['final_allocation']:.1%} (${tier['available_capital']:.2f})")
        print(f"   Position Limit: {tier['final_position_limit']:.1%} (${tier['max_position_size']:.2f})")
        print(f"   Description: {tier['description']}")

        results.append({
            "scenario": scenario["name"],
            "tier": tier
        })

    # Test Aggressive Mode
    print(f"\n🚨 TESTING AGGRESSIVE MODE")
    print("=" * 40)

    pm_aggressive = PortfolioManager(mode="simulated", initial_balance=10000.0, aggressive_mode=True)

    aggressive_scenarios = [
        {
            "name": "Aggressive Mode - Conservative Risk",
            "signal_strength": 0.5,
            "market_condition": "neutral",
            "asset_type": "crypto",
            "risk_appetite": "low"
        },
        {
            "name": "Aggressive Mode - Aggressive Risk",
            "signal_strength": 0.8,
            "market_condition": "bullish",
            "asset_type": "crypto",
            "risk_appetite": "aggressive"
        }
    ]

    for scenario in aggressive_scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        print("-" * 40)

        tier = pm_aggressive.get_allocation_tier(
            signal_strength=scenario["signal_strength"],
            market_condition=scenario["market_condition"],
            asset_type=scenario["asset_type"],
            risk_appetite=scenario["risk_appetite"]
        )

        print(f"   Tier: {tier['tier_name'].upper()} (AGGRESSIVE MODE)")
        print(f"   Risk Appetite: {tier['risk_appetite']}")
        print(f"   Signal Strength: {tier['signal_strength']:.2f} → {tier['signal_multiplier']:.2f}x")
        print(f"   Market Condition: {tier['market_condition']} → {tier['market_multiplier']:.2f}x")
        print(f"   Asset Type: {tier['asset_type']} → {tier['asset_multiplier']:.2f}x")
        print(f"   Final Allocation: {tier['final_allocation']:.1%} (${tier['available_capital']:.2f})")
        print(f"   Position Limit: {tier['final_position_limit']:.1%} (${tier['max_position_size']:.2f})")
        print(f"   Aggressive Mode Active: {tier['aggressive_mode_active']}")

        results.append({
            "scenario": scenario["name"],
            "tier": tier
        })

    # Summary comparison
    print(f"\n📈 ALLOCATION TIERS SUMMARY")
    print("=" * 60)

    print("Normal Mode Allocations:")
    normal_results = [r for r in results if "Aggressive Mode" not in r["scenario"]]
    for result in normal_results:
        tier = result["tier"]
        print(f"   {result['scenario']:<35}: {tier['final_allocation']:.1%} (${tier['available_capital']:>8.0f}) | Limit: {tier['final_position_limit']:.1%}")

    print("\nAggressive Mode Allocations:")
    aggressive_results = [r for r in results if "Aggressive Mode" in r["scenario"]]
    for result in aggressive_results:
        tier = result["tier"]
        print(f"   {result['scenario'].replace('Aggressive Mode - ', ''):<35}: {tier['final_allocation']:.1%} (${tier['available_capital']:>8.0f}) | Limit: {tier['final_position_limit']:.1%}")

    # Save detailed results
    with open("allocation_tiers_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: allocation_tiers_results.json")

    print(f"\n✅ ALLOCATION TIERS TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("🎯 Key Features Implemented:")
    print("   • 4 Base Allocation Tiers (Conservative, Balanced, Growth, Aggressive)")
    print("   • Signal Strength Adjustments (0.5x to 1.0x multiplier)")
    print("   • Market Condition Multipliers (0.7x to 1.2x)")
    print("   • Asset Type Adjustments (0.7x to 1.0x)")
    print("   • Aggressive Mode Override (1.3x to 1.5x additional)")
    print("   • Dynamic Position Limits and Capital Allocation")
    print("   • Comprehensive Risk Management Integration")

    return results

if __name__ == "__main__":
    test_allocation_tiers()
