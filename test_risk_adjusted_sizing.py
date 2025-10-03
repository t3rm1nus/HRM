#!/usr/bin/env python3
"""
Test Risk-Adjusted Position Sizing - Comprehensive testing of multi-factor risk management
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.portfolio_manager import PortfolioManager
from core.logging import logger
import json

def test_risk_adjusted_sizing():
    """Test comprehensive risk-adjusted position sizing system"""
    print("🎯 TESTING COMPREHENSIVE RISK-ADJUSTED POSITION SIZING")
    print("=" * 70)

    # Initialize portfolio manager with some existing positions to test concentration
    pm = PortfolioManager(mode="simulated", initial_balance=10000.0, aggressive_mode=False)

    # Add some existing positions to test concentration limits
    pm.portfolio['BTCUSDT'] = {'position': 0.2, 'free': 0.2}  # $10,000 position at $50,000/BTC
    pm.portfolio['USDT'] = {'free': 5000.0}  # $5,000 cash remaining

    # Test scenarios with different risk conditions
    test_scenarios = [
        {
            "name": "Strong Bull Market - High Confidence",
            "symbol": "BTCUSDT",
            "base_position_size": 2000.0,
            "signal_data": {
                "strength": 0.9,
                "confidence": 0.85,
                "technical_score": 0.8,
                "market_regime": "bullish"
            },
            "market_data": {
                "BTCUSDT": {"close": 50000.0, "volatility": 0.15}  # Low volatility
            },
            "risk_metrics": {"var_95": 0.02},  # Low VaR
            "expected_adjustments": "Should increase position size due to strong conditions"
        },
        {
            "name": "Bear Market - Weak Signal",
            "symbol": "ETHUSDT",
            "base_position_size": 1500.0,
            "signal_data": {
                "strength": 0.3,
                "confidence": 0.4,
                "technical_score": 0.35,
                "market_regime": "bearish"
            },
            "market_data": {
                "ETHUSDT": {"close": 3000.0, "volatility": 0.35}  # High volatility
            },
            "risk_metrics": {"var_95": 0.08},  # High VaR
            "expected_adjustments": "Should significantly reduce position size"
        },
        {
            "name": "Volatile Market - Moderate Signal",
            "symbol": "BTCUSDT",
            "base_position_size": 1000.0,
            "signal_data": {
                "strength": 0.6,
                "confidence": 0.65,
                "technical_score": 0.55,
                "market_regime": "volatile"
            },
            "market_data": {
                "BTCUSDT": {"close": 50000.0, "volatility": 0.45}  # Very high volatility
            },
            "risk_metrics": {"var_95": 0.06},  # Moderate VaR
            "expected_adjustments": "Should reduce size due to high volatility"
        },
        {
            "name": "Portfolio Drawdown Scenario",
            "symbol": "BTCUSDT",
            "base_position_size": 1200.0,
            "signal_data": {
                "strength": 0.7,
                "confidence": 0.6,
                "technical_score": 0.65,
                "market_regime": "neutral"
            },
            "market_data": {
                "BTCUSDT": {"close": 50000.0, "volatility": 0.25}
            },
            "risk_metrics": {"var_95": 0.04},
            "drawdown_setup": 0.12,  # 12% drawdown
            "expected_adjustments": "Should reduce size due to portfolio drawdown"
        },
        {
            "name": "High Concentration Warning",
            "symbol": "BTCUSDT",
            "base_position_size": 800.0,
            "signal_data": {
                "strength": 0.8,
                "confidence": 0.75,
                "technical_score": 0.7,
                "market_regime": "neutral"
            },
            "market_data": {
                "BTCUSDT": {"close": 50000.0, "volatility": 0.20}
            },
            "risk_metrics": {"var_95": 0.03},
            "expected_adjustments": "Should reduce size due to BTC concentration"
        }
    ]

    results = []

    for scenario in test_scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        print("-" * 50)
        print(f"   Base Position Size: ${scenario['base_position_size']:.2f}")
        print(f"   Expected: {scenario['expected_adjustments']}")

        # Setup special conditions
        if "drawdown_setup" in scenario:
            # Temporarily set peak value to create drawdown
            original_peak = pm.peak_value
            pm.peak_value = pm.get_total_value(scenario['market_data']) / (1 - scenario['drawdown_setup'])

        # Calculate comprehensive risk-adjusted position size
        sizing_result = pm.get_comprehensive_risk_adjusted_position_size(
            symbol=scenario['symbol'],
            base_position_size=scenario['base_position_size'],
            signal_data=scenario['signal_data'],
            market_data=scenario['market_data'],
            risk_metrics=scenario.get('risk_metrics')
        )

        # Restore original peak value if modified
        if "drawdown_setup" in scenario:
            pm.peak_value = original_peak

        # Display results
        print(f"   Final Position Size: ${sizing_result['final_size']:.2f}")
        print(f"   Risk Score: {sizing_result['position_risk_score']:.2f}")
        print(f"   Risk-Adjusted Return: ${sizing_result['risk_adjusted_return']:.2f}")

        if sizing_result['risk_warnings']:
            print(f"   ⚠️ Risk Warnings: {len(sizing_result['risk_warnings'])}")
            for warning in sizing_result['risk_warnings']:
                print(f"      - {warning}")

        if sizing_result['applied_limits']:
            print(f"   📏 Applied Limits: {len(sizing_result['applied_limits'])}")
            for limit in sizing_result['applied_limits']:
                print(f"      - {limit}")

        # Show key multipliers
        print(f"   Multipliers:")
        print(f"      Risk Appetite: {sizing_result['risk_appetite_multiplier']:.2f}x")
        print(f"      Signal Strength: {sizing_result['signal_strength_multiplier']:.2f}x")
        print(f"      Technical: {sizing_result['technical_multiplier']:.2f}x")
        print(f"      Market Regime: {sizing_result['market_regime_multiplier']:.2f}x")
        print(f"      Volatility: {sizing_result['volatility_multiplier']:.2f}x")
        print(f"      VaR: {sizing_result['var_multiplier']:.2f}x")
        print(f"      Drawdown: {sizing_result['drawdown_multiplier']:.2f}x")
        print(f"      Concentration: {sizing_result['concentration_multiplier']:.2f}x")

        results.append({
            "scenario": scenario["name"],
            "sizing_result": sizing_result
        })

    # Test Aggressive Mode comparison
    print(f"\n🚨 TESTING AGGRESSIVE MODE COMPARISON")
    print("=" * 50)

    pm_aggressive = PortfolioManager(mode="simulated", initial_balance=10000.0, aggressive_mode=True)
    pm_aggressive.portfolio['BTCUSDT'] = {'position': 0.2, 'free': 0.2}
    pm_aggressive.portfolio['USDT'] = {'free': 5000.0}

    comparison_scenario = {
        "symbol": "BTCUSDT",
        "base_position_size": 1500.0,
        "signal_data": {
            "strength": 0.75,
            "confidence": 0.7,
            "technical_score": 0.65,
            "market_regime": "bullish"
        },
        "market_data": {
            "BTCUSDT": {"close": 50000.0, "volatility": 0.22}
        },
        "risk_metrics": {"var_95": 0.035}
    }

    print(f"\n📊 Aggressive Mode vs Normal Mode Comparison")
    print("-" * 50)

    # Normal mode result
    normal_result = pm.get_comprehensive_risk_adjusted_position_size(**comparison_scenario)

    # Aggressive mode result
    aggressive_result = pm_aggressive.get_comprehensive_risk_adjusted_position_size(**comparison_scenario)

    print(f"   Base Position Size: ${comparison_scenario['base_position_size']:.2f}")
    print(f"   Normal Mode Final: ${normal_result['final_size']:.2f}")
    print(f"   Aggressive Mode Final: ${aggressive_result['final_size']:.2f}")
    print(f"   Difference: ${(aggressive_result['final_size'] - normal_result['final_size']):+.2f}")
    print(f"   Multiplier: {aggressive_result['final_size'] / normal_result['final_size']:.2f}x")

    results.append({
        "scenario": "Aggressive Mode Comparison",
        "normal_result": normal_result,
        "aggressive_result": aggressive_result
    })

    # Summary statistics
    print(f"\n📈 RISK-ADJUSTED SIZING SUMMARY")
    print("=" * 50)

    final_sizes = [r['sizing_result']['final_size'] for r in results[:-1]]  # Exclude comparison
    base_sizes = [r['sizing_result']['base_size'] for r in results[:-1]]

    avg_multiplier = sum(f/b for f, b in zip(final_sizes, base_sizes)) / len(final_sizes)
    risk_warnings_total = sum(len(r['sizing_result']['risk_warnings']) for r in results[:-1])
    limits_applied_total = sum(len(r['sizing_result']['applied_limits']) for r in results[:-1])

    print(f"   Scenarios Tested: {len(test_scenarios)}")
    print(f"   Average Size Multiplier: {avg_multiplier:.2f}x")
    print(f"   Total Risk Warnings: {risk_warnings_total}")
    print(f"   Total Limits Applied: {limits_applied_total}")
    print(f"   Risk Management Effectiveness: {'✅ HIGH' if risk_warnings_total >= 3 else '⚠️ MODERATE'}")

    # Save detailed results
    with open("risk_adjusted_sizing_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: risk_adjusted_sizing_results.json")

    print(f"\n✅ COMPREHENSIVE RISK-ADJUSTED POSITION SIZING TEST COMPLETED")
    print("=" * 70)
    print("🎯 Key Risk Factors Implemented:")
    print("   • Risk Appetite Adjustment (Conservative/Moderate/High/Aggressive)")
    print("   • Signal Strength & Confidence Multipliers")
    print("   • Volatility-based Position Scaling")
    print("   • VaR (Value at Risk) Integration")
    print("   • Portfolio Drawdown Protection")
    print("   • Technical Strength Scoring")
    print("   • Market Regime Adaptation")
    print("   • Portfolio Concentration Limits")
    print("   • Correlation-adjusted Sizing")
    print("   • Allocation Tier Integration")
    print("   • Aggressive Mode Override")
    print("   • Comprehensive Risk Scoring")
    print("   • Real-time Risk Warnings")

    return results

if __name__ == "__main__":
    test_risk_adjusted_sizing()
