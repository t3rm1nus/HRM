import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import TEMPORARY_AGGRESSIVE_MODE, set_config_value
from core.l3_processor import get_l3_decision
from l1_operational.simulated_exchange_client import SimulatedExchangeClient
from core.portfolio_manager import PortfolioManager

async def test_aggressive_mode():
    """Test temporary aggressive mode functionality"""
    
    # Test 1: Verify default mode is conservative
    print("1. Testing default conservative mode:")
    if not TEMPORARY_AGGRESSIVE_MODE:
        print("✅ Default mode is conservative (TEMPORARY_AGGRESSIVE_MODE = False)")
    else:
        print("❌ Default mode should be conservative")
    
    # Create test market data
    test_market_data = {
        "BTCUSDT": [],
        "ETHUSDT": []
    }
    
    # Test 2: Get L3 decision in conservative mode
    print("\n2. Testing L3 decision in conservative mode:")
    l3_decision_conservative = get_l3_decision(test_market_data)
    print(f"   Regime: {l3_decision_conservative.get('regime', 'unknown')}")
    print(f"   Signal: {l3_decision_conservative.get('signal', 'unknown')}")
    print(f"   Allow L2 signals: {l3_decision_conservative.get('allow_l2_signals', False)}")
    
    # Enable aggressive mode temporarily
    print("\n3. Enabling temporary aggressive mode:")
    import core.config
    core.config.TEMPORARY_AGGRESSIVE_MODE = True
    print("✅ Temporary aggressive mode enabled")
    
    # Test 3: Get L3 decision in aggressive mode
    print("\n4. Testing L3 decision in aggressive mode:")
    l3_decision_aggressive = get_l3_decision(test_market_data)
    print(f"   Regime: {l3_decision_aggressive.get('regime', 'unknown')}")
    print(f"   Signal: {l3_decision_aggressive.get('signal', 'unknown')}")
    print(f"   Allow L2 signals: {l3_decision_aggressive.get('allow_l2_signals', False)}")
    
    assert l3_decision_aggressive.get('allow_l2_signals') == True
    print("✅ Allow L2 signals is True in aggressive mode")
    
    # Test 4: Verify strategic_hold is False
    assert l3_decision_aggressive.get('strategic_hold') == False
    print("✅ Strategic hold is False in aggressive mode")
    
    # Disable aggressive mode
    print("\n5. Disabling temporary aggressive mode:")
    core.config.TEMPORARY_AGGRESSIVE_MODE = False
    print("✅ Temporary aggressive mode disabled")
    
    print("\n📊 Test completed successfully!")
    print("\n📝 Summary of changes:")
    print("   - Default mode is conservative")
    print("   - Aggressive mode overrides L3 decision to allow L2 signals")
    print("   - Aggressive mode sets strategic_hold to False")
    print("   - Mode can be easily toggled on/off")

async def test_portfolio_behavior():
    """Test portfolio behavior in aggressive mode"""
    print("\n6. Testing portfolio behavior:")
    
    # Create simulated exchange client
    client = SimulatedExchangeClient({"USDT": 10000, "BTC": 0.1, "ETH": 1.0})
    pm = PortfolioManager(client=client, mode="simulated")
    
    # Test initial state
    print(f"   Initial BTC: {pm.get_balance('BTCUSDT'):.6f}")
    print(f"   Initial ETH: {pm.get_balance('ETHUSDT'):.6f}")
    print(f"   Initial USDT: {pm.get_balance('USDT'):.2f}")
    
    assert pm.get_balance('BTCUSDT') == 0.1
    assert pm.get_balance('ETHUSDT') == 1.0
    assert pm.get_balance('USDT') == 10000
    
    print("✅ Portfolio initialized correctly")

async def main():
    await test_aggressive_mode()
    await test_portfolio_behavior()

if __name__ == "__main__":
    asyncio.run(main())