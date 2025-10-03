#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify portfolio initialization fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.portfolio_manager import PortfolioManager

def test_portfolio_initialization():
    """Test that portfolio initializes correctly with 3000 USDT"""
    print("🧪 Testing PortfolioManager initialization...")

    # Create PortfolioManager in live mode without client (should use fallback)
    pm = PortfolioManager(
        mode="live",
        initial_balance=3000.0,
        client=None,  # No client to avoid BinanceClient initialization issues
        symbols=['BTCUSDT', 'ETHUSDT']
    )

    # Force clean reset
    pm.force_clean_reset()

    # Check balances
    btc_balance = pm.get_balance("BTCUSDT")
    eth_balance = pm.get_balance("ETHUSDT")
    usdt_balance = pm.get_balance("USDT")

    print("📊 Portfolio state after initialization:")
    print(f"   BTC: {btc_balance:.6f}")
    print(f"   ETH: {eth_balance:.3f}")
    print(f"   USDT: {usdt_balance:.2f}")

    # Verify USDT balance is correct
    if abs(usdt_balance - 3000.0) < 0.01:
        print("✅ SUCCESS: USDT balance is correct (3000.00)")
        return True
    else:
        print(f"❌ FAILURE: USDT balance is {usdt_balance}, expected 3000.00")
        return False

if __name__ == "__main__":
    success = test_portfolio_initialization()
    sys.exit(0 if success else 1)
