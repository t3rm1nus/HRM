#!/usr/bin/env python3
"""
Test script to verify portfolio persistence and state consistency
"""
import os
import sys
import json
import asyncio
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.portfolio_manager import PortfolioManager

async def test_portfolio_persistence():
    """Test portfolio persistence functionality"""
    print("🧪 Testing Portfolio Persistence...")

    # Test 1: Initialize PortfolioManager
    print("\n1. Testing PortfolioManager initialization...")
    pm = PortfolioManager(
        mode="live",
        initial_balance=3000.0,
        symbols=['BTCUSDT', 'ETHUSDT']
    )

    print(f"   Initial state: BTC={pm.get_balance('BTCUSDT'):.6f}, ETH={pm.get_balance('ETHUSDT'):.3f}, USDT={pm.get_balance('USDT'):.2f}")

    # Test 2: Save state
    print("\n2. Testing save functionality...")
    pm.save_to_json()
    print("   State saved to JSON")

    # Test 3: Load state
    print("\n3. Testing load functionality...")
    pm2 = PortfolioManager(
        mode="live",
        initial_balance=3000.0,
        symbols=['BTCUSDT', 'ETHUSDT']
    )

    if pm2.load_from_json():
        print("   State loaded from JSON successfully")
        print(f"   Loaded state: BTC={pm2.get_balance('BTCUSDT'):.6f}, ETH={pm2.get_balance('ETHUSDT'):.3f}, USDT={pm2.get_balance('USDT'):.2f}")
    else:
        print("   ❌ Failed to load state from JSON")

    # Test 4: Simulate order processing
    print("\n4. Testing order processing...")
    mock_orders = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.001,
            "status": "filled",
            "filled_price": 50000.0
        }
    ]

    mock_market_data = {
        "BTCUSDT": {"close": 50000.0},
        "ETHUSDT": {"close": 3000.0}
    }

    await pm.update_from_orders_async(mock_orders, mock_market_data)
    print(f"   After order: BTC={pm.get_balance('BTCUSDT'):.6f}, ETH={pm.get_balance('ETHUSDT'):.3f}, USDT={pm.get_balance('USDT'):.2f}")

    # Test 5: Save updated state
    print("\n5. Testing updated state save...")
    pm.save_to_json()
    print("   Updated state saved")

    # Test 6: Load updated state in new instance
    print("\n6. Testing updated state load...")
    pm3 = PortfolioManager(
        mode="live",
        initial_balance=3000.0,
        symbols=['BTCUSDT', 'ETHUSDT']
    )

    if pm3.load_from_json():
        print("   Updated state loaded successfully")
        print(f"   Final state: BTC={pm3.get_balance('BTCUSDT'):.6f}, ETH={pm3.get_balance('ETHUSDT'):.3f}, USDT={pm3.get_balance('USDT'):.2f}")

        # Verify consistency
        if (abs(pm3.get_balance('BTCUSDT') - pm.get_balance('BTCUSDT')) < 0.000001 and
            abs(pm3.get_balance('ETHUSDT') - pm.get_balance('ETHUSDT')) < 0.000001 and
            abs(pm3.get_balance('USDT') - pm.get_balance('USDT')) < 0.01):
            print("   ✅ State persistence is consistent!")
        else:
            print("   ❌ State persistence is inconsistent!")
    else:
        print("   ❌ Failed to load updated state")

    print("\n🎯 Portfolio persistence test completed!")

if __name__ == "__main__":
    asyncio.run(test_portfolio_persistence())
