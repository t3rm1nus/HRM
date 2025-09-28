#!/usr/bin/env python3
"""
Test script for the three implemented solutions:
1. Validación Mejorada de Órdenes
2. Gestión Mejorada del Capital
3. Configuración Recomendada
"""

from l1_operational.order_manager import OrderManager
from core.portfolio_manager import PortfolioManager
from comms.config import config

def test_order_validation():
    """Test the new validate_order_size method"""
    print("🧪 Testing OrderManager.validate_order_size...")

    order_manager = OrderManager()

    # Test portfolio state
    portfolio = {
        'BTCUSDT': {'position': 0.0, 'free': 0.0},
        'ETHUSDT': {'position': 0.0, 'free': 0.0},
        'USDT': {'free': 3000.0}
    }

    # Test 1: Valid order (meets minimum size and has sufficient funds)
    result = order_manager.validate_order_size('BTCUSDT', 0.0002, 50000, portfolio)
    assert result['valid'] == True, f"Expected valid order, got: {result}"
    assert result['order_value'] == 10.0, f"Expected order value 10.0, got: {result['order_value']}"
    print("✅ Test 1 passed: Valid order accepted")

    # Test 2: Invalid order (too small)
    result = order_manager.validate_order_size('BTCUSDT', 0.00001, 50000, portfolio)
    assert result['valid'] == False, f"Expected invalid order, got: {result}"
    assert "below minimum" in result['reason'], f"Expected minimum size error, got: {result['reason']}"
    print("✅ Test 2 passed: Too small order rejected")

    # Test 3: Invalid order (insufficient funds)
    result = order_manager.validate_order_size('BTCUSDT', 0.1, 50000, portfolio)
    assert result['valid'] == False, f"Expected invalid order, got: {result}"
    assert "Insufficient capital" in result['reason'], f"Expected insufficient funds error, got: {result['reason']}"
    print("✅ Test 3 passed: Insufficient funds order rejected")

    # Test 4: Sell order validation
    portfolio_with_position = {
        'BTCUSDT': {'position': 0.01, 'free': 0.01},
        'USDT': {'free': 1000.0}
    }
    result = order_manager.validate_order_size('BTCUSDT', -0.005, 50000, portfolio_with_position)
    assert result['valid'] == True, f"Expected valid sell order, got: {result}"
    print("✅ Test 4 passed: Valid sell order accepted")

    # Test 5: Invalid sell order (no position)
    result = order_manager.validate_order_size('BTCUSDT', -0.001, 50000, portfolio)
    assert result['valid'] == False, f"Expected invalid sell order, got: {result}"
    assert "No position to sell" in result['reason'], f"Expected no position error, got: {result['reason']}"
    print("✅ Test 5 passed: Sell order without position rejected")

def test_portfolio_allocation():
    """Test the new update_portfolio_allocation method"""
    print("\n🧪 Testing PortfolioManager.update_portfolio_allocation...")

    portfolio_manager = PortfolioManager(mode='simulated', initial_balance=3000.0)

    # Test allocation update
    available_trading_capital, max_per_symbol = portfolio_manager.update_portfolio_allocation()

    # Verify calculations
    expected_trading_capital = 3000.0 * 0.80  # 80% of USDT balance
    assert available_trading_capital == expected_trading_capital, f"Expected {expected_trading_capital}, got {available_trading_capital}"

    expected_max_per_symbol = 3000.0 * 0.3  # 30% of total portfolio
    assert max_per_symbol == expected_max_per_symbol, f"Expected {expected_max_per_symbol}, got {max_per_symbol}"

    print("✅ Portfolio allocation test passed")

def test_trading_config():
    """Test the new TRADING_CONFIG in config"""
    print("\n🧪 Testing TRADING_CONFIG...")

    trading_config = config['TRADING_CONFIG']

    # Verify required keys exist
    required_keys = [
        'MIN_ORDER_SIZE_USD', 'MAX_ALLOCATION_PER_SYMBOL_PCT', 'AVAILABLE_TRADING_CAPITAL_PCT',
        'CASH_RESERVE_PCT', 'TRADING_FEE_RATE', 'MAX_DAILY_TRADES', 'RISK_LIMITS',
        'VALIDATION', 'ALLOCATION'
    ]

    for key in required_keys:
        assert key in trading_config, f"Missing required key: {key}"

    # Verify values
    assert trading_config['MIN_ORDER_SIZE_USD'] == 10.0, f"Expected 10.0, got {trading_config['MIN_ORDER_SIZE_USD']}"
    assert trading_config['MAX_ALLOCATION_PER_SYMBOL_PCT'] == 30.0, f"Expected 30.0, got {trading_config['MAX_ALLOCATION_PER_SYMBOL_PCT']}"
    assert trading_config['AVAILABLE_TRADING_CAPITAL_PCT'] == 80.0, f"Expected 80.0, got {trading_config['AVAILABLE_TRADING_CAPITAL_PCT']}"
    assert trading_config['CASH_RESERVE_PCT'] == 20.0, f"Expected 20.0, got {trading_config['CASH_RESERVE_PCT']}"
    assert trading_config['TRADING_FEE_RATE'] == 0.001, f"Expected 0.001, got {trading_config['TRADING_FEE_RATE']}"

    # Verify nested structures
    risk_limits = trading_config['RISK_LIMITS']
    assert 'MAX_DRAWDOWN_PCT' in risk_limits, "Missing MAX_DRAWDOWN_PCT"
    assert 'MAX_POSITION_SIZE_PCT' in risk_limits, "Missing MAX_POSITION_SIZE_PCT"
    assert 'MIN_CAPITAL_REQUIREMENT_USD' in risk_limits, "Missing MIN_CAPITAL_REQUIREMENT_USD"

    validation = trading_config['VALIDATION']
    assert validation['ENABLE_ORDER_SIZE_CHECK'] == True, "ENABLE_ORDER_SIZE_CHECK should be True"
    assert validation['ENABLE_CAPITAL_CHECK'] == True, "ENABLE_CAPITAL_CHECK should be True"
    assert validation['ENABLE_POSITION_CHECK'] == True, "ENABLE_POSITION_CHECK should be True"

    allocation = trading_config['ALLOCATION']
    assert allocation['DYNAMIC_REBALANCING'] == True, "DYNAMIC_REBALANCING should be True"
    assert allocation['CONCENTRATION_LIMIT_PCT'] == 30.0, "CONCENTRATION_LIMIT_PCT should be 30.0"

    print("✅ TRADING_CONFIG test passed")

def main():
    """Run all tests"""
    print("🚀 Testing HRM Trading System Improvements")
    print("=" * 50)

    try:
        test_order_validation()
        test_portfolio_allocation()
        test_trading_config()

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Three solutions successfully implemented:")
        print("   1. Validación Mejorada de Órdenes")
        print("   2. Gestión Mejorada del Capital")
        print("   3. Configuración Recomendada")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
