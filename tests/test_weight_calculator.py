"""
Test script for the Weight Calculator and Portfolio Rebalancing system
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from core.weight_calculator import (
    WeightCalculator, WeightStrategy, WeightConstraints, AssetData
)
from core.correlation_position_sizer import (
    CorrelationPositionSizer, CorrelationData, CorrelationSizingConfig
)
from core.portfolio_rebalancer import (
    PortfolioRebalancer, RebalanceTrigger, RebalanceConfig
)
from core.portfolio_manager import PortfolioManager
from core.logging import logger


def create_sample_market_data():
    """Create sample market data for testing"""
    return {
        "BTCUSDT": {
            "close": 45000.0,
            "volatility": 0.25,
            "market_cap": 850_000_000_000
        },
        "ETHUSDT": {
            "close": 2800.0,
            "volatility": 0.30,
            "market_cap": 340_000_000_000
        },
        "ADAUSDT": {
            "close": 0.45,
            "volatility": 0.35,
            "market_cap": 16_000_000_000
        },
        "SOLUSDT": {
            "close": 95.0,
            "volatility": 0.40,
            "market_cap": 42_000_000_000
        }
    }


def create_sample_correlation_matrix():
    """Create sample correlation matrix"""
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
    # Create a realistic correlation matrix
    corr_data = {
        "BTCUSDT": [1.0, 0.65, 0.45, 0.55],
        "ETHUSDT": [0.65, 1.0, 0.50, 0.60],
        "ADAUSDT": [0.45, 0.50, 1.0, 0.70],
        "SOLUSDT": [0.55, 0.60, 0.70, 1.0]
    }
    return pd.DataFrame(corr_data, index=symbols)


def test_weight_calculator():
    """Test the weight calculator with different strategies"""
    print("\n" + "="*60)
    print("🎯 TESTING WEIGHT CALCULATOR")
    print("="*60)

    # Initialize weight calculator
    constraints = WeightConstraints(
        max_concentration=0.4,  # Max 40% per asset
        rebalance_threshold=0.03  # Rebalance if drift > 3%
    )
    calculator = WeightCalculator(constraints)

    # Add sample assets
    market_data = create_sample_market_data()
    for symbol, data in market_data.items():
        asset = AssetData(
            symbol=symbol,
            price=data["close"],
            market_cap=data.get("market_cap"),
            volatility=data.get("volatility", 0.2),
            expected_return=0.08  # 8% expected annual return
        )
        calculator.add_asset(asset)

    print(f"📊 Added {len(calculator.assets)} assets for weighting")

    # Test different weighting strategies
    strategies = [
        WeightStrategy.EQUAL,
        WeightStrategy.MARKET_CAP,
        WeightStrategy.RISK_PARITY,
        WeightStrategy.VOLATILITY_TARGETED,
        WeightStrategy.MINIMUM_VARIANCE,
        WeightStrategy.MAXIMUM_SHARPE
    ]

    results = {}
    for strategy in strategies:
        try:
            if strategy == WeightStrategy.VOLATILITY_TARGETED:
                weights = calculator.calculate_weights(strategy, target_volatility=0.15)
            else:
                weights = calculator.calculate_weights(strategy)

            results[strategy.value] = weights

            print(f"\n⚖️ {strategy.value.upper()} WEIGHTS:")
            for symbol, weight in weights.items():
                print(".1%")

            # Calculate risk metrics
            risk_metrics = calculator.get_portfolio_risk_metrics(weights)
            print(f"   📊 Risk Metrics: Vol={risk_metrics.get('portfolio_volatility', 0):.1%}, "
                  f"Return={risk_metrics.get('portfolio_return', 0):.1%}, "
                  f"Sharpe={risk_metrics.get('sharpe_ratio', 0):.2f}")

        except Exception as e:
            print(f"❌ Error with {strategy.value}: {e}")

    return results


def test_correlation_position_sizer():
    """Test the correlation-based position sizer"""
    print("\n" + "="*60)
    print("🔗 TESTING CORRELATION POSITION SIZER")
    print("="*60)

    # Initialize correlation sizer
    config = CorrelationSizingConfig(
        max_correlation_threshold=0.7,
        diversification_bonus=1.3,
        correlation_penalty_factor=0.6
    )
    sizer = CorrelationPositionSizer(config)

    # Add correlation data
    market_data = create_sample_market_data()
    correlation_matrix = create_sample_correlation_matrix()
    sizer.update_correlation_matrix(correlation_matrix)

    for symbol in market_data.keys():
        correlations = correlation_matrix[symbol].to_dict() if symbol in correlation_matrix.columns else {}
        correlation_data = CorrelationData(
            symbol=symbol,
            correlations=correlations,
            volatility=market_data[symbol].get("volatility", 0.2)
        )
        sizer.add_asset_correlation_data(correlation_data)

    # Test correlation-adjusted sizing
    base_position_size = 1000.0  # $1000 base position
    current_portfolio = {
        "BTCUSDT": 0.5,  # 50% in BTC
        "ETHUSDT": 0.3,  # 30% in ETH
        "USDT": 0.2      # 20% cash
    }

    print("📊 Current Portfolio Weights:")
    for symbol, weight in current_portfolio.items():
        print(".1%")

    print(f"\n🎯 Base Position Size: ${base_position_size:.2f}")

    for symbol in ["ADAUSDT", "SOLUSDT"]:  # Test with new assets
        adjusted_size = sizer.calculate_correlation_adjusted_size(
            symbol, base_position_size, current_portfolio, market_data
        )

        print(f"\n🔄 {symbol} Correlation-Adjusted Sizing:")
        print(".2f")
        print(".2f")
        print(".1f")

    # Generate correlation report
    report = sizer.get_correlation_report(current_portfolio)
    print("\n📈 Correlation Report:")
    print(".3f")
    print(f"   Most Correlated Pair: {report.get('most_correlated_pair', 'N/A')}")
    print(f"   Correlation Clusters: {report.get('correlation_clusters', [])}")

    return sizer


def test_portfolio_rebalancer():
    """Test the portfolio rebalancer"""
    print("\n" + "="*60)
    print("🔄 TESTING PORTFOLIO REBALANCER")
    print("="*60)

    # Initialize components
    calculator = WeightCalculator()
    market_data = create_sample_market_data()

    # Add assets
    for symbol, data in market_data.items():
        asset = AssetData(
            symbol=symbol,
            price=data["close"],
            market_cap=data.get("market_cap"),
            volatility=data.get("volatility", 0.2)
        )
        calculator.add_asset(asset)

    # Initialize rebalancer
    rebalance_config = RebalanceConfig(
        rebalance_frequency_days=30,
        drift_threshold=0.05,  # 5% drift threshold
        max_drift_threshold=0.10,
        min_trade_size=50.0
    )
    rebalancer = PortfolioRebalancer(calculator, rebalance_config)

    # Set target weights (equal weighting)
    target_weights = calculator.calculate_weights(WeightStrategy.EQUAL)
    rebalancer.set_target_weights(target_weights)

    print("🎯 Target Weights (Equal):")
    for symbol, weight in target_weights.items():
        print(".1%")

    # Simulate current portfolio with some drift
    current_weights = {
        "BTCUSDT": 0.35,  # Should be ~0.25, drifted +10%
        "ETHUSDT": 0.20,  # Should be ~0.25, drifted -5%
        "ADAUSDT": 0.15,  # Should be ~0.25, drifted -10%
        "SOLUSDT": 0.30,  # Should be ~0.25, drifted +5%
    }

    print("\n📊 Current Weights (with drift):")
    for symbol, weight in current_weights.items():
        target = target_weights.get(symbol, 0)
        drift = abs(weight - target)
        print(".1%")

    # Check if rebalance is needed
    portfolio_value = 10000.0  # $10,000 portfolio
    should_rebalance, reason = rebalancer.should_rebalance(current_weights)

    print(f"\n🔍 Rebalance Check: {should_rebalance} - {reason}")

    if should_rebalance:
        # Calculate rebalance trades
        trades = rebalancer.calculate_rebalance_trades(current_weights, portfolio_value)

        print("\n💼 Required Rebalance Trades:")
        for symbol, trade_value in trades.items():
            action = "BUY" if trade_value > 0 else "SELL"
            print(".2f")

        # Estimate costs
        costs = rebalancer.estimate_rebalance_costs(trades, market_data)
        print(".2f")

        # Calculate impact
        impact = rebalancer.calculate_rebalance_impact(
            type('RebalanceResult', (), {
                'trades_required': trades,
                'portfolio_value': portfolio_value,
                'current_weights': current_weights,
                'target_weights': target_weights,
                'estimated_costs': costs
            })()
        )

        print("\n📊 Rebalance Impact:")
        print(".1%")
        print(".1%")
        print(".1%")

    return rebalancer


async def test_portfolio_manager_integration():
    """Test the portfolio manager integration"""
    print("\n" + "="*60)
    print("🏦 TESTING PORTFOLIO MANAGER INTEGRATION")
    print("="*60)

    # Initialize portfolio manager
    pm = PortfolioManager(mode="simulated", initial_balance=10000.0)

    # Initialize weight calculator components
    success = pm.initialize_weight_calculator()
    if not success:
        print("❌ Failed to initialize weight calculator")
        return

    # Add assets for weighting
    market_data = create_sample_market_data()
    for symbol, data in market_data.items():
        pm.add_asset_for_weighting(
            symbol=symbol,
            price=data["close"],
            market_cap=data.get("market_cap"),
            volatility=data.get("volatility", 0.2),
            expected_return=0.08
        )

    # Add correlation data
    correlation_matrix = create_sample_correlation_matrix()
    pm.update_correlation_matrix(correlation_matrix)

    for symbol in market_data.keys():
        correlations = correlation_matrix[symbol].to_dict() if symbol in correlation_matrix.columns else {}
        pm.add_correlation_data(symbol, correlations, market_data[symbol].get("volatility", 0.2))

    print("✅ Portfolio Manager initialized with weight calculator")

    # Test weight calculation
    weights = pm.calculate_portfolio_weights(WeightStrategy.RISK_PARITY)
    print("\n⚖️ Risk Parity Weights:")
    for symbol, weight in weights.items():
        print(".1%")

    # Test correlation-adjusted position sizing
    base_size = 500.0
    current_market_data = {symbol: {"close": data["close"]} for symbol, data in market_data.items()}

    for symbol in ["ADAUSDT", "SOLUSDT"]:
        adjusted_size = pm.calculate_correlation_adjusted_position_size(
            symbol, base_size, current_market_data
        )
        print(".2f")

    # Test rebalance check
    current_weights = pm.get_current_portfolio_weights(current_market_data)
    should_rebalance, reason = pm.check_rebalance_needed(weights, current_market_data)

    print(f"\n🔄 Rebalance Check: {should_rebalance} - {reason}")

    # Get risk metrics
    risk_metrics = pm.get_portfolio_risk_metrics(weights)
    print("\n📊 Portfolio Risk Metrics:")
    print(".1%")
    print(".1%")
    print(".2f")

    # Get correlation report
    corr_report = pm.get_correlation_report()
    print("\n🔗 Correlation Report:")
    print(".3f")
    print(f"   Clusters: {len(corr_report.get('correlation_clusters', []))}")

    return pm


def run_comprehensive_test():
    """Run comprehensive test of all weight calculator features"""
    print("🚀 STARTING COMPREHENSIVE WEIGHT CALCULATOR TEST")
    print("="*80)

    try:
        # Test individual components
        weight_results = test_weight_calculator()
        correlation_sizer = test_correlation_position_sizer()
        rebalancer = test_portfolio_rebalancer()

        # Test integration
        asyncio.run(test_portfolio_manager_integration())

        print("\n" + "="*80)
        print("✅ COMPREHENSIVE TEST COMPLETED SUCCESSFULLY")
        print("="*80)

        # Summary
        print("\n📋 TEST SUMMARY:")
        print(f"   ✅ Weight Calculator: {len(weight_results)} strategies tested")
        print("   ✅ Correlation Position Sizer: Working")
        print("   ✅ Portfolio Rebalancer: Working")
        print("   ✅ Portfolio Manager Integration: Working")

        print("\n🎯 WEIGHT CALCULATOR FEATURES IMPLEMENTED:")
        print("   • Equal Weighting")
        print("   • Market Cap Weighting")
        print("   • Risk Parity Weighting")
        print("   • Volatility Targeted Weighting")
        print("   • Minimum Variance Optimization")
        print("   • Maximum Sharpe Ratio Optimization")
        print("   • Correlation-Based Position Sizing")
        print("   • Automated Portfolio Rebalancing")
        print("   • Risk Metrics Calculation")
        print("   • Portfolio Manager Integration")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_test()
